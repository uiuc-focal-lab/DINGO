import torch

@torch.jit.script
def viterbi_block(edge_src: torch.Tensor,      # [E]
                  edge_dst: torch.Tensor,      # [E]
                  edge_tok: torch.Tensor,      # [E]
                  logp_block: torch.Tensor,    # [B, V]  (B = block_length)
                  init_cost: torch.Tensor      # [S]
):
    B, V = logp_block.shape
    S    = init_cost.numel()
    E    = edge_src.numel()

    # Buffers that will be swapped in‑place
    curr_cost = init_cost.clone()
    next_cost = torch.empty_like(curr_cost)

    back_tokens = torch.full((B, S), -1, dtype=torch.long, device=init_cost.device)
    back_src    = torch.full((B, S), -1, dtype=torch.long, device=init_cost.device)

    for t in range(B):
        # 1) score each edge
        edge_score = curr_cost[edge_src] + logp_block[t, edge_tok]

        # 2) scatter‑max onto destinations
        next_cost.fill_(-float('inf'))
        next_cost.scatter_reduce_(0, edge_dst, edge_score, reduce='amax')

        # 3) record argmax edges
        mask       = next_cost[edge_dst] == edge_score
        best_edges = mask.nonzero().squeeze(1)

        back_tokens[t, edge_dst[best_edges]] = edge_tok[best_edges]
        back_src   [t, edge_dst[best_edges]] = edge_src[best_edges]

        # 4) prepare for next position
        curr_cost, next_cost = next_cost, curr_cost

    return curr_cost, back_tokens, back_src

class ViterbiGraph:
    """
    Wraps a CUDA graph around the scripted viterbi_block.
    Re‑usable for every block, provided shapes remain constant.
    """
    def __init__(self, edge_src, edge_dst, edge_tok,
                 block_length, vocab_size, num_states, device):

        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_tok = edge_tok
        B             = block_length
        S             = num_states
        V             = vocab_size
        self.device   = device

        # Static input/output “shadow” tensors (must live for the graph's lifetime)
        self.logp_buf   = torch.empty((B, V),  device=device)      # in
        self.init_cost  = torch.empty(S,        device=device)     # in
        self.out_cost   = torch.empty_like(self.init_cost)         # out
        self.btok_buf   = torch.empty((B, S), dtype=torch.long, device=device)
        self.bsrc_buf   = torch.empty((B, S), dtype=torch.long, device=device)

        # Warm‑up call to allocate everything once
        _ = viterbi_block(edge_src, edge_dst, edge_tok,
                          self.logp_buf.normal_(), self.init_cost.zero_())

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        stream     = torch.cuda.Stream()
        with torch.cuda.graph(self.graph, stream=stream):
            oc, bt, bs = viterbi_block(edge_src, edge_dst, edge_tok,
                                       self.logp_buf, self.init_cost)
            # copy into persistent outputs
            self.out_cost.copy_(oc)
            self.btok_buf.copy_(bt)
            self.bsrc_buf.copy_(bs)

        # After capture, all writes will go directly into out_cost / btok_buf / bsrc_buf

    @torch.no_grad()
    def run(self, logp_block, init_cost):
        """
        logp_block : [B, V]  (F.log_softmax output for a whole block)
        init_cost  : [S]     (cost vector from previous token)
        Returns    : (final_cost[S], back_tokens[B,S], back_src[B,S])
        """
        self.logp_buf.copy_(logp_block)   # host<‑‑>device copy is cheap (already on device)
        self.init_cost.copy_(init_cost)
        self.graph.replay()               # ➊ GPU executes the captured sequence
        return self.out_cost, self.btok_buf, self.bsrc_buf