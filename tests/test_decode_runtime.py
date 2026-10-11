import pytest
import torch

from models import constrained_decode
from models.constrained_decode import decode_block


def test_deferred_validation_rejects_an_earlier_impossible_step(store_factory):
    store = store_factory("a")
    decode_block(
        store,
        torch.full((1, 4), -torch.inf),
        torch.tensor([False]),
        store.initial_state,
        0,
        defer_validation=True,
    )
    decode_block(
        store,
        torch.tensor([[9.0, 1.0, -10.0, -10.0]]),
        torch.tensor([False]),
        store.initial_state,
        0,
        defer_validation=True,
    )
    with pytest.raises(ValueError, match="supported constrained path"):
        constrained_decode.validate_decode_block(store, torch.device("cpu"))
    # A failed request must not poison the next one.
    decode_block(
        store,
        torch.tensor([[9.0, 1.0, -10.0, -10.0]]),
        torch.tensor([False]),
        store.initial_state,
        0,
        defer_validation=True,
    )
    constrained_decode.validate_decode_block(store, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("opportunistic", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_production_graph_replay_preserves_state_and_support(store_factory, opportunistic, greedy):
    store = store_factory("ab", opportunistic=opportunistic)
    device = torch.device("cuda")
    logits = torch.tensor([[9.0, 1.0, -10.0, -10.0], [1.0, 9.0, -10.0, -10.0]], device=device)
    masked = torch.tensor([True, False], device=device)
    for _ in range(2):
        tokens, state = decode_block(
            store, logits, masked, store.initial_state, 0, greedy=greedy, defer_validation=True
        )
        constrained_decode.validate_decode_block(store, device)
        assert tokens.tolist() == [3, 1]
        assert state.device.type == "cuda"
    assert len(store._runtime_decoders) == 1
    # Reuse the graph with changed mask support; final output must be fully revealed.
    tokens, _ = decode_block(
        store,
        logits,
        torch.zeros(2, dtype=torch.bool, device=device),
        store.initial_state,
        0,
        greedy=greedy,
    )
    assert tokens.tolist() == [0, 1]
    with pytest.raises(ValueError, match="supported constrained path"):
        decode_block(
            store,
            torch.full_like(logits, -torch.inf),
            torch.zeros_like(masked),
            store.initial_state,
            0,
            greedy=greedy,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warm_graph_decode_has_no_host_scalar_or_dynamic_shape_operations(store_factory):
    store = store_factory("a|b", opportunistic=True)
    logits = torch.tensor([[9.0, 1.0, -10.0, -10.0]], device="cuda")
    masked = torch.tensor([False], device="cuda")
    _, state = decode_block(store, logits, masked, store.initial_state, 0, defer_validation=True)
    constrained_decode.validate_decode_block(store, logits.device)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        decode_block(store, logits, masked, state, 0, defer_validation=True)
    keys = {event.key for event in prof.key_averages()}
    assert "aten::_local_scalar_dense" not in keys
    assert "aten::nonzero" not in keys


def test_unindexed_cuda_validation_uses_current_device_key(store_factory, monkeypatch):
    store = store_factory("a")
    store._runtime_validity = {"cuda:0": torch.tensor(False)}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ValueError, match="supported constrained path"):
        constrained_decode.validate_decode_block(store, torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fallback", [False, True])
def test_production_graph_selects_fused_backend_or_optional_fallback(
    store_factory, monkeypatch, fallback
):
    from models import cuda_viterbi

    if fallback:

        def unavailable(*args, **kwargs):
            raise ImportError("Triton is unavailable")

        monkeypatch.setattr(cuda_viterbi, "FusedViterbi", unavailable)
    else:
        pytest.importorskip("triton")
    store = store_factory("a")
    logits = torch.tensor([[9.0, 1.0, -10.0, -10.0]], device="cuda")
    tokens, _ = decode_block(
        store, logits, torch.tensor([False], device="cuda"), store.initial_state, 0
    )
    assert tokens.tolist() == [0]
    decoder = next(iter(store._runtime_decoders.values()))
    assert decoder.backend == ("torch" if fallback else "triton")
