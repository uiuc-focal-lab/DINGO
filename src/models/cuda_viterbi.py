"""Optional fused float32 CUDA Viterbi; Triton is imported on demand."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _step_kernel():
    import triton

    # JIT resolves this module global while compiling the lazily defined kernel.
    global tl
    import triton.language as tl

    @triton.jit(do_not_specialize=["TIME"])
    def step(
        SRC,
        TOK,
        IDS,
        ROWS,
        CURR,
        LOGP,
        NEXT,
        BT,
        BS,
        VOCAB: tl.constexpr,
        STATES: tl.constexpr,
        TIME,
        BLOCK: tl.constexpr,
    ):
        destination = tl.program_id(0)
        position_offset = TIME.to(tl.int64)
        begin = tl.load(ROWS + destination)
        end = tl.load(ROWS + destination + 1)
        lanes = tl.arange(0, BLOCK)
        best_score = tl.full((BLOCK,), -float("inf"), tl.float32)
        best_id = tl.full((BLOCK,), 2147483647, tl.int32)
        best_position = tl.full((BLOCK,), 2147483647, tl.int32)
        saw_nan = tl.full((BLOCK,), False, tl.int1)
        for offset in range(begin, end, BLOCK):
            edge = offset + lanes
            present = edge < end
            source = tl.load(SRC + edge, present, other=0)
            token = tl.load(TOK + edge, present, other=0)
            original_id = tl.load(IDS + edge, present, other=2147483647)
            score = tl.load(CURR + source) + tl.load(LOGP + position_offset * VOCAB + token)
            nan = present & (score != score)
            saw_nan = saw_nan | nan
            score = tl.where(present & ~nan, score, -float("inf"))
            improve = (score > best_score) | ((score == best_score) & (original_id < best_id))
            best_score = tl.where(improve, score, best_score)
            best_id = tl.where(improve, original_id, best_id)
            best_position = tl.where(improve, edge.to(tl.int32), best_position)
        maximum = tl.max(best_score, 0)
        nan_row = tl.sum(saw_nan.to(tl.int32), 0) > 0
        finite = (maximum > -float("inf")) & (maximum < float("inf")) & ~nan_row
        winner = tl.min(tl.where(best_score == maximum, best_id, 2147483647), 0)
        # Keep the grouped position paired with the original winning edge ID.
        position = tl.min(tl.where(best_id == winner, best_position, 2147483647), 0)
        source = tl.load(SRC + position, finite, other=-1)
        token = tl.load(TOK + position, finite, other=-1)
        tl.store(NEXT + destination, tl.where(nan_row, float("nan"), maximum))
        tl.store(BT + position_offset * STATES + destination, token)
        tl.store(BS + position_offset * STATES + destination, source)

    return step


@lru_cache(maxsize=1)
def _tiled_kernels():
    import triton

    global tl
    import triton.language as tl

    @triton.jit(do_not_specialize=["TIME"])
    def tiles(
        SRC,
        TOK,
        IDS,
        BEGIN,
        END,
        CURR,
        LOGP,
        SCORE,
        ID,
        POSITION,
        NAN,
        VOCAB: tl.constexpr,
        TIME,
        BLOCK: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        edge = tl.load(BEGIN + chunk) + tl.arange(0, BLOCK)
        present = edge < tl.load(END + chunk)
        source = tl.load(SRC + edge, present, other=0)
        token = tl.load(TOK + edge, present, other=0)
        original_id = tl.load(IDS + edge, present, other=2147483647)
        score = tl.load(CURR + source) + tl.load(LOGP + TIME.to(tl.int64) * VOCAB + token)
        nan = present & (score != score)
        score = tl.where(present & ~nan, score, -float("inf"))
        maximum = tl.max(score, 0)
        winner = tl.min(tl.where(score == maximum, original_id, 2147483647), 0)
        position = tl.min(
            tl.where(present & (original_id == winner), edge.to(tl.int32), 2147483647), 0
        )
        tl.store(SCORE + chunk, maximum)
        tl.store(ID + chunk, winner)
        tl.store(POSITION + chunk, position)
        tl.store(NAN + chunk, tl.sum(nan.to(tl.int32), 0) > 0)

    @triton.jit(do_not_specialize=["TIME"])
    def merge(
        SRC,
        TOK,
        ROWS,
        SCORE,
        ID,
        POSITION,
        NAN,
        NEXT,
        BT,
        BS,
        STATES: tl.constexpr,
        TIME,
        BLOCK: tl.constexpr,
    ):
        destination = tl.program_id(0)
        begin = tl.load(ROWS + destination)
        end = tl.load(ROWS + destination + 1)
        lanes = tl.arange(0, BLOCK)
        best_score = tl.full((BLOCK,), -float("inf"), tl.float32)
        best_id = tl.full((BLOCK,), 2147483647, tl.int32)
        best_position = tl.full((BLOCK,), 2147483647, tl.int32)
        saw_nan = tl.full((BLOCK,), False, tl.int1)
        for offset in range(begin, end, BLOCK):
            chunk = offset + lanes
            present = chunk < end
            score = tl.load(SCORE + chunk, present, other=-float("inf"))
            original_id = tl.load(ID + chunk, present, other=2147483647)
            position = tl.load(POSITION + chunk, present, other=2147483647)
            nan = tl.load(NAN + chunk, present, other=False)
            saw_nan = saw_nan | nan
            improve = (score > best_score) | ((score == best_score) & (original_id < best_id))
            best_score = tl.where(improve, score, best_score)
            best_id = tl.where(improve, original_id, best_id)
            best_position = tl.where(improve, position, best_position)
        maximum = tl.max(best_score, 0)
        nan_row = tl.sum(saw_nan.to(tl.int32), 0) > 0
        finite = (maximum > -float("inf")) & (maximum < float("inf")) & ~nan_row
        winner = tl.min(tl.where(best_score == maximum, best_id, 2147483647), 0)
        position = tl.min(tl.where(best_id == winner, best_position, 2147483647), 0)
        source = tl.load(SRC + position, finite, other=-1)
        token = tl.load(TOK + position, finite, other=-1)
        position_offset = TIME.to(tl.int64)
        tl.store(NEXT + destination, tl.where(nan_row, float("nan"), maximum))
        tl.store(BT + position_offset * STATES + destination, token)
        tl.store(BS + position_offset * STATES + destination, source)

    return tiles, merge


class FusedViterbi:
    """Immutable CSR graph with graph-capturable, fixed-shape block execution.

    Supports contiguous float32 scores/costs and CUDA long edge tensors with
    fewer than 2**31 edges. Setup sorting and validation stay outside capture.
    Returned block tensors belong to that invocation (or its captured graph).
    Automatic tiling splits large rows over four times the mean incoming degree;
    tile_size=None explicitly keeps the single-CTA reduction.
    """

    def __init__(self, edge_src, edge_dst, edge_tok, num_states, block_size=256, tile_size="auto"):
        if edge_src.device.type != "cuda":
            raise ValueError("FusedViterbi requires CUDA edge tensors")
        if not all(
            e.device == edge_src.device and e.dtype == torch.long and e.ndim == 1
            for e in (edge_src, edge_dst, edge_tok)
        ):
            raise ValueError("Edges must be one-dimensional CUDA long tensors on one device")
        if not (edge_src.numel() == edge_dst.numel() == edge_tok.numel() < 2**31):
            raise ValueError("Edges must have equal lengths below 2**31")
        if num_states <= 0:
            raise ValueError("num_states must be positive")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        if tile_size not in (None, "auto") and (tile_size <= 0 or tile_size & (tile_size - 1)):
            raise ValueError("tile_size must be a positive power of two")
        if edge_src.numel() and (
            bool((edge_src < 0).any())
            or bool((edge_src >= num_states).any())
            or bool((edge_dst < 0).any())
            or bool((edge_dst >= num_states).any())
            or bool((edge_tok < 0).any())
        ):
            raise ValueError("Edges contain invalid state or token IDs")
        self.device = edge_src.device
        self.num_states = num_states
        self.block_size = block_size
        order = torch.argsort(edge_dst, stable=True)
        self.src = edge_src[order].contiguous()
        self.tok = edge_tok[order].contiguous()
        self.ids = order.to(torch.int32)
        counts = torch.bincount(edge_dst, minlength=num_states)
        if tile_size == "auto":
            maximum_degree = int(counts.max())
            tile_size = (
                512
                if maximum_degree > 4096 and maximum_degree * num_states > 4 * edge_src.numel()
                else None
            )
        self.tile_size = tile_size
        self.rows = torch.cat((counts.new_zeros(1), counts.cumsum(0))).contiguous()
        self.max_token = int(edge_tok.max()) if edge_tok.numel() else -1
        self.kernel = _step_kernel()
        if tile_size is not None:
            boundaries = self.rows.cpu().tolist()
            begins, ends, chunk_rows = [], [], [0]
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                for begin in range(start, stop, tile_size):
                    begins.append(begin)
                    ends.append(min(begin + tile_size, stop))
                chunk_rows.append(len(begins))
            self.num_tiles = len(begins)
            self.tile_begin = torch.tensor(begins, device=self.device, dtype=torch.long)
            self.tile_end = torch.tensor(ends, device=self.device, dtype=torch.long)
            self.tile_rows = torch.tensor(chunk_rows, device=self.device, dtype=torch.long)
            self.tiles_kernel, self.merge_kernel = _tiled_kernels()

    def fused_block(self, logp_block, init_cost):
        if (
            logp_block.device != self.device
            or init_cost.device != self.device
            or logp_block.dtype != torch.float32
            or init_cost.dtype != torch.float32
            or logp_block.ndim != 2
            or init_cost.shape != (self.num_states,)
            or not logp_block.is_contiguous()
            or not init_cost.is_contiguous()
        ):
            raise ValueError(
                "Scores and costs must be contiguous CUDA float32 tensors with matching shapes"
            )
        length, vocab = logp_block.shape
        if vocab <= self.max_token:
            raise ValueError("Score vocabulary is smaller than the graph token IDs")
        current = init_cost.clone()
        following = torch.empty_like(current)
        tokens = torch.empty((length, self.num_states), device=self.device, dtype=torch.long)
        sources = torch.empty_like(tokens)
        if self.tile_size is not None:
            partial_score = torch.empty(self.num_tiles, dtype=torch.float32, device=self.device)
            partial_id = torch.empty(self.num_tiles, dtype=torch.int32, device=self.device)
            partial_position = torch.empty_like(partial_id)
            partial_nan = torch.empty(self.num_tiles, dtype=torch.bool, device=self.device)
        for position in range(length):
            if self.tile_size is None:
                self.kernel[(self.num_states,)](
                    self.src,
                    self.tok,
                    self.ids,
                    self.rows,
                    current,
                    logp_block,
                    following,
                    tokens,
                    sources,
                    vocab,
                    self.num_states,
                    position,
                    self.block_size,
                    num_warps=4,
                )
            else:
                if self.num_tiles:
                    self.tiles_kernel[(self.num_tiles,)](
                        self.src,
                        self.tok,
                        self.ids,
                        self.tile_begin,
                        self.tile_end,
                        current,
                        logp_block,
                        partial_score,
                        partial_id,
                        partial_position,
                        partial_nan,
                        vocab,
                        position,
                        self.tile_size,
                        num_warps=4,
                    )
                self.merge_kernel[(self.num_states,)](
                    self.src,
                    self.tok,
                    self.tile_rows,
                    partial_score,
                    partial_id,
                    partial_position,
                    partial_nan,
                    following,
                    tokens,
                    sources,
                    self.num_states,
                    position,
                    128,
                    num_warps=4,
                )
            current, following = following, current
        return current, tokens, sources
