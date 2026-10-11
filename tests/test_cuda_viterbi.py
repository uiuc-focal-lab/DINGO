import importlib

import pytest
import torch

from models.viterbi import viterbi_block


def test_optional_fused_module_imports_without_cuda_or_triton():
    module = importlib.import_module("models.cuda_viterbi")
    with pytest.raises(ValueError, match="CUDA"):
        module.FusedViterbi(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("edges", [0, 3, 47, 4097])
@pytest.mark.parametrize("block_size", [256, 1024])
def test_fused_sparse_dp_matches_fixed_and_compact_reference(seed, edges, block_size):
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    torch.manual_seed(seed)
    states, vocab, length = 5, 11, 4
    src = torch.randint(states, (edges,), device="cuda")
    dst = torch.randint(states, (edges,), device="cuda")
    tok = torch.randint(vocab, (edges,), device="cuda")
    scores = torch.randn(length, vocab, device="cuda")
    scores[1].fill_(0)
    scores[2, ::2] = -torch.inf
    init = torch.full((states,), -torch.inf, device="cuda")
    init[0] = 0
    actual = FusedViterbi(src, dst, tok, states, block_size=block_size).fused_block(scores, init)
    for compact in (False, True):
        expected = viterbi_block(src, dst, tok, scores, init, compact)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_preserves_original_edge_ties_and_nonfinite_semantics():
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    src = torch.tensor([2, 1, 0, 2, 0, 1, 0], device="cuda")
    dst = torch.tensor([1, 0, 1, 0, 2, 2, 2], device="cuda")
    tok = torch.tensor([1, 0, 0, 1, 0, 1, 2], device="cuda")
    fused = FusedViterbi(src, dst, tok, 3)
    for values in (
        [0.0, -0.0, 0.0],
        [torch.inf, 0.0, -torch.inf],
        [float("nan"), 0.0, -torch.inf],
        [-torch.inf] * 3,
    ):
        scores = torch.tensor([values, values], device="cuda")
        init = torch.zeros(3, device="cuda")
        expected = viterbi_block(src, dst, tok, scores, init, False)
        actual = fused.fused_block(scores, init)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_full_block_cuda_graph_capture_and_replay():
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    src = torch.tensor([0, 1, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    tok = torch.tensor([0, 1, 1, 0], device="cuda")
    fused = FusedViterbi(src, dst, tok, 2)
    scores = torch.zeros(7, 2, device="cuda")
    init = torch.tensor([0.0, -torch.inf], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.fused_block(scores, init)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused.fused_block(scores, init)
    for seed in range(4):
        torch.manual_seed(seed)
        scores.copy_(torch.randn_like(scores))
        expected = viterbi_block(src, dst, tok, scores, init, False)
        graph.replay()
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tile_size", [256, 1024])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_tiled_skewed_rows_preserve_score_ties_and_capture(tile_size, nonfinite):
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    torch.manual_seed(21)
    edges, states, vocab, length = 8193, 5, 13, 4
    src = torch.randint(states, (edges,), device="cuda")
    dst = torch.full((edges,), 2, device="cuda")
    dst[::67] = 0
    tok = torch.randint(vocab, (edges,), device="cuda")
    scores = torch.randn(length, vocab, device="cuda")
    scores[0].zero_()
    scores[2, ::2] = -torch.inf
    if nonfinite:
        scores[1, 1] = float("nan")
        scores[1, 2] = torch.inf
    init = torch.zeros(states, device="cuda")
    fused = FusedViterbi(src, dst, tok, states, tile_size=tile_size)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.fused_block(scores, init)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused.fused_block(scores, init)
    expected = viterbi_block(src, dst, tok, scores, init, False)
    graph.replay()
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tiled_empty_edges_keep_all_states_unreachable():
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    edges = torch.empty(0, dtype=torch.long, device="cuda")
    init = torch.zeros(3, device="cuda")
    scores = torch.zeros(2, 1, device="cuda")
    cost, bt, bs = FusedViterbi(edges, edges, edges, 3, tile_size=256).fused_block(scores, init)
    assert torch.isneginf(cost).all()
    assert (bt == -1).all()
    assert (bs == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "skewed,edges,want_tile", [(True, 8193, 512), (False, 8193, None), (True, 40, None)]
)
def test_automatic_tiling_only_splits_large_concentrated_rows(skewed, edges, want_tile):
    pytest.importorskip("triton")
    from models.cuda_viterbi import FusedViterbi

    src = torch.arange(edges, device="cuda") % 5
    dst = torch.zeros_like(src) if skewed else src
    tok = src.clone()
    engine = FusedViterbi(src, dst, tok, 5)
    assert engine.tile_size == want_tile
    scores = torch.zeros(2, 5, device="cuda")
    init = torch.zeros(5, device="cuda")
    expected = viterbi_block(src, dst, tok, scores, init, False)
    actual = engine.fused_block(scores, init)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
