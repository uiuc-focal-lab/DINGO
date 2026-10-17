import itertools

import pytest
import torch

from models.viterbi import ViterbiGraph, viterbi_block


def enumerate_costs(src, dst, tok, scores, init):
    """Independent sequence enumeration, including all NFA branches."""
    best = [-float("inf")] * len(init)
    for sequence in itertools.product(range(scores.shape[1]), repeat=len(scores)):
        paths = [(s, float(c)) for s, c in enumerate(init) if torch.isfinite(c)]
        for i, token in enumerate(sequence):
            paths = [
                (d, c + float(scores[i, token]))
                for state, c in paths
                for s, d, t in zip(src, dst, tok, strict=True)
                if s == state and t == token
            ]
        for state, score in paths:
            best[state] = max(best[state], score)
    return torch.tensor(best, dtype=scores.dtype)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("compact", [False, True])
def test_sparse_dp_matches_exhaustive_nfa_search(seed, compact):
    g = torch.Generator().manual_seed(seed)
    edges = [
        (s, d, t)
        for s in range(3)
        for d in range(3)
        for t in range(3)
        if torch.rand((), generator=g) > 0.5
    ]
    src, dst, tok = map(torch.tensor, zip(*edges, strict=True))
    scores = torch.randn((3, 3), generator=g, dtype=torch.float64).log_softmax(-1)
    # One point-mass masked position; token 2 can have several destinations.
    scores[1] = torch.tensor([-torch.inf, -torch.inf, 0.0])
    init = torch.tensor([-torch.inf, 0.0, -torch.inf], dtype=torch.float64)
    costs, bt, bs = viterbi_block(src, dst, tok, scores, init, compact)
    torch.testing.assert_close(costs, enumerate_costs(src, dst, tok, scores, init))
    for end in range(3):
        if not torch.isfinite(costs[end]):
            continue
        state, total = end, 0.0
        for i in range(2, -1, -1):
            token, prev = bt[i, state].item(), bs[i, state].item()
            assert (prev, state, token) in edges
            total += float(scores[i, token])
            state = prev
        assert state == 1
        assert total == pytest.approx(float(costs[end]))


def test_unreachable_states_have_no_backpointers():
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 1])
    tok = torch.tensor([0, 1])
    _, bt, bs = viterbi_block(
        src, dst, tok, torch.full((2, 2), -torch.inf), torch.tensor([0.0, -torch.inf])
    )
    assert torch.all(bt == -1)
    assert torch.all(bs == -1)


def test_ties_choose_one_consistent_edge():
    src, dst, tok = (torch.tensor([0, 1, 0]), torch.tensor([2, 2, 2]), torch.tensor([1, 0, 0]))
    _, bt, bs = viterbi_block(src, dst, tok, torch.zeros(1, 2), torch.zeros(3))
    # First edge is the deterministic tie break; token and source stay paired.
    assert (bs[0, 2].item(), bt[0, 2].item()) == (0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replay_and_ties():
    src = torch.tensor([0, 1, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    tok = torch.tensor([0, 1, 1, 0], device="cuda")
    graph = ViterbiGraph(src, dst, tok, 3, 2, 2, "cuda")
    for seed in range(4):
        torch.manual_seed(seed)
        scores = (
            torch.zeros(3, 2, device="cuda")
            if seed == 0
            else torch.randn(3, 2, device="cuda").log_softmax(-1)
        )
        init = torch.tensor([0.0, -torch.inf], device="cuda")
        expected = viterbi_block(src, dst, tok, scores, init)
        actual = graph.run(scores, init)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e)
