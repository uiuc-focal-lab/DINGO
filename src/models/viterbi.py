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