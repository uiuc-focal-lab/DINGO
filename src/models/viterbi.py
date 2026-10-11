from typing import Optional

import torch


@torch.jit.script
def viterbi_block(
    edge_src: torch.Tensor,  # [E]
    edge_dst: torch.Tensor,  # [E]
    edge_tok: torch.Tensor,  # [E]
    logp_block: torch.Tensor,  # [B, V]  (B = block_length)
    init_cost: torch.Tensor,  # [S]
    compact_backpointers: Optional[bool] = None,
):
    B, V = logp_block.shape
    S = init_cost.numel()
    E = edge_src.numel()
    # Compact winners avoid a second full-edge atomic reduction on large
    # eager workloads. They synchronize once per position via nonzero(), so
    # small blocks and CUDA graph capture use fixed-size reductions instead.
    compact = False if compact_backpointers is None else compact_backpointers

    # Buffers that will be swapped in‑place
    curr_cost = init_cost.to(dtype=logp_block.dtype).clone()
    next_cost = torch.empty_like(curr_cost)

    back_tokens = torch.full((B, S), -1, dtype=torch.long, device=init_cost.device)
    back_src = torch.full((B, S), -1, dtype=torch.long, device=init_cost.device)
    edge_ids = torch.arange(E, device=init_cost.device)
    best_edge = torch.empty(S, dtype=torch.long, device=init_cost.device)

    for t in range(B):
        # 1) score each edge
        edge_score = curr_cost[edge_src] + logp_block[t, edge_tok]

        # 2) scatter‑max onto destinations
        next_cost.fill_(-float("inf"))
        next_cost.scatter_reduce_(0, edge_dst, edge_score, reduce="amax")

        # 3) record argmax edges
        # Pick exactly one edge per destination. Advanced-index assignments
        # with duplicate destinations race on CUDA and can mix token/source
        # backpointers. Fixed-size reductions also permit CUDA graph capture
        # (nonzero() has a dynamic output shape and synchronizes the host).
        winners = (next_cost[edge_dst] == edge_score) & torch.isfinite(edge_score)
        best_edge.fill_(E)
        if compact:
            winning_ids = winners.nonzero().flatten()
            best_edge.scatter_reduce_(0, edge_dst[winning_ids], winning_ids, reduce="amin")
        else:
            best_edge.scatter_reduce_(0, edge_dst, torch.where(winners, edge_ids, E), reduce="amin")
        if E > 0:
            safe_edge = best_edge.clamp(max=E - 1)
            back_tokens[t] = torch.where(best_edge < E, edge_tok[safe_edge], -1)
            back_src[t] = torch.where(best_edge < E, edge_src[safe_edge], -1)

        # 4) prepare for next position
        curr_cost, next_cost = next_cost, curr_cost

    return curr_cost, back_tokens, back_src


@torch.jit.script
def backtrack(back_tokens: torch.Tensor, back_src: torch.Tensor, end_state: torch.Tensor):
    """Reconstruct on device without a host synchronization per token.

    The caller must first establish that end_state has a finite path cost.
    """
    length = back_tokens.shape[0]
    path = torch.empty(length, dtype=torch.long, device=back_tokens.device)
    state = end_state.reshape(1)
    for i in range(length - 1, -1, -1):
        path[i] = back_tokens[i].gather(0, state)[0]
        state = back_src[i].gather(0, state).clamp_min(0)
    return path


@torch.jit.script
def constrained_path(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_tok: torch.Tensor,
    scores: torch.Tensor,
    init: torch.Tensor,
    targets: torch.Tensor,
    greedy: bool = False,
):
    """Fixed-shape decoding, including safe outputs when no path is supported."""
    if not greedy:
        cost, bt, bs = viterbi_block(edge_src, edge_dst, edge_tok, scores, init, False)
        cost = cost.masked_fill(~targets, -float("inf"))
        end = cost.argmax()
        valid = torch.isfinite(cost.gather(0, end.reshape(1))).all()
        return backtrack(bt, bs, end), end, valid
    length = scores.shape[0]
    states = init.numel()
    can_finish = [targets]
    for i in range(length - 1, -1, -1):
        supported = torch.isfinite(scores[i, edge_tok]) & can_finish[-1][edge_dst]
        can = torch.zeros(states, dtype=torch.long, device=scores.device)
        can.scatter_reduce_(0, edge_src, supported.long(), reduce="amax")
        can_finish.append(can.to(dtype=torch.bool))
    current = torch.isfinite(init)
    path = torch.empty(length, dtype=torch.long, device=scores.device)
    valid = torch.ones((), dtype=torch.bool, device=scores.device)
    for i in range(length):
        viable = current[edge_src] & can_finish[length - i - 1][edge_dst]
        edge_scores = scores[i, edge_tok].masked_fill(~viable, -float("inf"))
        if edge_tok.numel() == 0:
            path[i] = -1
            valid = valid & False
        else:
            best = edge_scores.argmax().reshape(1)
            token = edge_tok.gather(0, best)[0]
            path[i] = token
            valid = valid & torch.isfinite(edge_scores.gather(0, best)).all()
            chosen = viable & (edge_tok == token) & torch.isfinite(edge_scores)
            counts = torch.zeros(states, dtype=torch.long, device=scores.device)
            counts.scatter_reduce_(0, edge_dst, chosen.long(), reduce="amax")
            current = counts.to(dtype=torch.bool)
    feasible = current & targets
    end = feasible.long().argmax()
    return path, end, valid & feasible.any()


class ViterbiGraph:
    """
    Replays fixed-shape decoding, using fused CUDA DP when Triton is available.
    Re‑usable for every block, provided shapes remain constant.
    """

    def __init__(
        self,
        edge_src,
        edge_dst,
        edge_tok,
        block_length,
        vocab_size,
        num_states,
        device,
        decode=False,
        greedy=False,
    ):

        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_tok = edge_tok
        B = block_length
        S = num_states
        V = vocab_size
        self.device = device
        self.decode = decode
        self.greedy = greedy
        if torch.device(device).type != "cuda":
            raise ValueError("ViterbiGraph requires a CUDA device")
        self.fused = None
        if not greedy:
            try:
                from models.cuda_viterbi import FusedViterbi

                self.fused = FusedViterbi(edge_src, edge_dst, edge_tok, S)
            except ImportError:
                # PyTorch remains usable on CUDA installations without Triton.
                pass
        self.backend = "triton" if self.fused is not None else "torch"

        # Static input/output “shadow” tensors (must live for the graph's lifetime)
        self.logp_buf = torch.empty((B, V), device=device, dtype=torch.float32)  # in
        self.init_cost = torch.empty(S, device=device, dtype=torch.float32)  # in
        self.out_cost = torch.empty_like(self.init_cost)  # out
        self.btok_buf = torch.empty((B, S), dtype=torch.long, device=device)
        self.bsrc_buf = torch.empty((B, S), dtype=torch.long, device=device)
        self.targets_buf = torch.ones(S, dtype=torch.bool, device=device)
        self.path_buf = torch.empty(B, dtype=torch.long, device=device)
        self.end_buf = torch.empty((), dtype=torch.long, device=device)
        self.valid_buf = torch.empty((), dtype=torch.bool, device=device)

        # Warm up the exact captured path on a side stream.
        self.logp_buf.zero_()
        self.init_cost.zero_()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._compute()

    def _compute(self):
        if self.fused is not None:
            cost, bt, bs = self.fused.fused_block(self.logp_buf, self.init_cost)
            if self.decode:
                cost = cost.masked_fill(~self.targets_buf, -float("inf"))
                end = cost.argmax()
                valid = torch.isfinite(cost.gather(0, end.reshape(1))).all()
                self.path_buf.copy_(backtrack(bt, bs, end))
                self.end_buf.copy_(end)
                self.valid_buf.copy_(valid)
            else:
                self.out_cost.copy_(cost)
                self.btok_buf.copy_(bt)
                self.bsrc_buf.copy_(bs)
        elif self.decode:
            path, end, valid = constrained_path(
                self.edge_src,
                self.edge_dst,
                self.edge_tok,
                self.logp_buf,
                self.init_cost,
                self.targets_buf,
                self.greedy,
            )
            self.path_buf.copy_(path)
            self.end_buf.copy_(end)
            self.valid_buf.copy_(valid)
        else:
            cost, bt, bs = viterbi_block(
                self.edge_src, self.edge_dst, self.edge_tok, self.logp_buf, self.init_cost, False
            )
            self.out_cost.copy_(cost)
            self.btok_buf.copy_(bt)
            self.bsrc_buf.copy_(bs)

    @torch.no_grad()
    def run(self, logp_block, init_cost, targets=None):
        """
        logp_block : [B, V]  (F.log_softmax output for a whole block)
        init_cost  : [S]     (cost vector from previous token)
        Returns    : (final_cost[S], back_tokens[B,S], back_src[B,S])
        """
        if logp_block.shape != self.logp_buf.shape or init_cost.shape != self.init_cost.shape:
            raise ValueError("CUDA graph inputs must match the captured shapes")
        self.logp_buf.copy_(logp_block)  # host<‑‑>device copy is cheap (already on device)
        self.init_cost.copy_(init_cost)
        if self.decode:
            if targets is None or targets.shape != self.targets_buf.shape:
                raise ValueError("Decode graphs require a target mask with the captured shape")
            self.targets_buf.copy_(targets)
        self.graph.replay()  # GPU executes the captured sequence
        if self.decode:
            # The committed block state must survive subsequent replays.
            return self.path_buf, self.end_buf.clone(), self.valid_buf
        return self.out_cost, self.btok_buf, self.bsrc_buf
