import dill
import pytest
import torch
from helpers import TinyTokenizer

from dfa import DFAMatrixStore


def test_mask_consumes_one_token_not_an_empty_transition(store_factory):
    store = store_factory()
    assert store.check_is_reachable([3, 1], is_final_block=True)
    assert not store.check_is_reachable([3, 0, 1], is_final_block=True)
    assert store.initial_state not in store.token_transitions.get((store.initial_state, 3), [])


def test_missing_eot_must_not_remove_an_ordinary_unknown_token(store_factory):
    store = store_factory(tokenizer=TinyTokenizer(unknown_eot=True))
    assert store.check_is_reachable([0, 1], is_final_block=True)


def test_cache_distinguishes_schema_vocabulary_and_runtime_options(store_factory):
    store_factory("a", opportunistic=False)
    b = store_factory("b", opportunistic=True)
    assert b.check_is_reachable([1], is_final_block=True)
    assert not b.check_is_reachable([0], is_final_block=True)
    assert b.enable_oppurtunistic
    assert store_factory("a", opportunistic=True).enable_oppurtunistic
    swapped = TinyTokenizer(vocab={"a": 1, "b": 0, "<eos>": 2, "<mask>": 3})
    assert store_factory("a", swapped).check_is_reachable([1], is_final_block=True)


def test_no_tokenizable_completion_is_explicit(store_factory):
    tok = TinyTokenizer(vocab={"a": 0, "<eos>": 2, "<mask>": 3})
    with pytest.raises(ValueError, match="token|completion|language"):
        store_factory("ab", tok)


def test_serialized_store_retains_regex_and_transitions(store_factory):
    store = store_factory()
    restored = dill.loads(dill.dumps(store))
    assert restored.check_is_reachable([0, 1], is_final_block=True)
    assert restored.dfa.matches("ab")


def test_back_dist_handles_states_that_cannot_reach_acceptance():
    dist = DFAMatrixStore("test").precompute_back_dist(3, torch.tensor([0]), torch.tensor([1]), [1])
    assert dist[:2].tolist() == [1, 0]
    assert dist[2] > 3


def test_empty_sparse_edge_set_has_valid_shape():
    edges = DFAMatrixStore("test").build_sparse_edges({}, 3)
    assert all(t.shape == (0,) for t in edges)


def test_ar_logits_process_masks_all_but_reachable_tokens(store_factory):
    store = store_factory("ab")
    scores = store.ar_logits_process(torch.tensor([[0]]), torch.zeros(1, 4))
    assert torch.isfinite(scores).tolist() == [[False, True, False, True]]


def test_completion_targets_reuses_mask_on_requested_device(store_factory):
    store = store_factory("ab")
    first = store.completion_targets(1, device="cpu")
    assert first.dtype == torch.bool
    assert first[store.initial_state].item() is False
    assert store.completion_targets(2)[store.initial_state].item() is True
    assert store.completion_targets(1, device=torch.device("cpu")).data_ptr() == first.data_ptr()


def test_exact_completion_targets_cache_preserves_length_and_cycles():
    store = DFAMatrixStore("test")
    store.num_states = 3
    store.final_states = [2]
    store.mask_token_id = 9
    store.acceptance_can_pad = False
    store.dst_to_final = torch.tensor([2, 1, 0])
    store.token_transitions = {(0, 0): [1], (1, 1): [2], (2, 0): [1], (0, 9): [2]}
    expected = {
        0: [False, False, True],
        1: [False, True, False],
        2: [True, False, True],
        3: [False, True, False],
    }
    for length in (3, 0, 2, 1):
        mask = store.completion_targets(length, device="cpu")
        assert mask.tolist() == expected[length]
        assert store.completion_targets(length).data_ptr() == mask.data_ptr()
    with pytest.raises(ValueError, match="nonnegative"):
        store.completion_targets(-1)


def test_completion_runtime_cache_is_excluded_from_serialization(store_factory):
    store = store_factory("ab")
    cached = store.completion_targets(2, device="cpu")
    restored = dill.loads(dill.dumps(store))
    assert restored.completion_targets(2).tolist() == cached.tolist()
    assert restored.completion_targets(2).data_ptr() != cached.data_ptr()
    assert restored.completion_targets(2).data_ptr() == restored.completion_targets(2).data_ptr()


def test_all_runtime_device_caches_are_excluded_from_serialization(store_factory):
    store = store_factory("ab")
    store._runtime_decoder = object()
    restored = dill.loads(dill.dumps(store))
    assert not hasattr(restored, "_runtime_decoder")


def test_rebuilding_store_invalidates_completion_masks(store_factory):
    store = store_factory("a")
    assert store.completion_targets(1)[store.initial_state].item() is True
    rebuilt = store.build("ab", 1, TinyTokenizer())
    mask = rebuilt.completion_targets(1)
    assert mask.shape == (rebuilt.num_states,)
    assert mask[rebuilt.initial_state].item() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_completion_targets_default_tracks_distance_tensor_device(store_factory):
    store = store_factory("ab")
    store.dst_to_final = store.dst_to_final.cuda()
    mask = store.completion_targets(2)
    assert mask.device == store.dst_to_final.device
    assert mask[store.initial_state].item() is True
    assert store.completion_targets(2, device="cpu").device.type == "cpu"
