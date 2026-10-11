import pytest
import torch
from helpers import TinyTokenizer

from models.constrained_decode import decode_block, masked_log_scores


def test_masked_positions_are_point_masses_and_revealed_masks_are_impossible():
    logits = torch.tensor([[1000.0, 999.0, 998.0, 1001.0], [2.0, 3.0, 4.0, 5.0]])
    scores = masked_log_scores(logits, torch.tensor([False, True]), 3)
    assert scores[0, 0] == 1000.0
    assert scores[0, 3] == -torch.inf
    assert scores[1, 3] == 0.0
    assert torch.all(scores[1, :3] == -torch.inf)


def test_valid_but_suboptimal_candidate_cannot_bypass_dp(store_factory):
    store = store_factory("a|b", opportunistic=True)
    # An existing 'a' is valid, but the current output distribution prefers 'b'.
    logits = torch.tensor([[1.0, 9.0, -100.0, -100.0]])
    tokens, state = decode_block(
        store, logits, torch.tensor([False]), store.initial_state, remaining_tokens=0
    )
    assert tokens.tolist() == [1]
    assert state in store.final_states


def test_impossible_support_raises_instead_of_backtracking_negative_ids(store_factory):
    store = store_factory("ab")
    with pytest.raises(ValueError, match="No.*path"):
        decode_block(
            store,
            torch.full((1, 4), -torch.inf),
            torch.tensor([False]),
            store.initial_state,
            remaining_tokens=0,
        )


def test_prefix_objective_and_complete_objective_are_explicit(store_factory):
    store = store_factory("ab")
    logits = torch.tensor([[4.0, 1.0, -100.0, -100.0]])
    tokens, _ = decode_block(
        store,
        logits,
        torch.tensor([False]),
        store.initial_state,
        remaining_tokens=0,
        require_complete=False,
    )
    assert tokens.tolist() == [0]
    with pytest.raises(ValueError, match="No.*path"):
        decode_block(
            store,
            logits,
            torch.tensor([False]),
            store.initial_state,
            remaining_tokens=0,
            require_complete=True,
        )


def test_completion_budget_avoids_a_prefix_that_cannot_finish(store_factory):
    store = store_factory("aaa|b")
    # Highest-probability prefix 'a' cannot finish in the one token left.
    logits = torch.tensor([[9.0, 1.0, -100.0, -100.0]])
    tokens, _ = decode_block(
        store, logits, torch.tensor([False]), store.initial_state, remaining_tokens=1
    )
    assert tokens.tolist() == [1]


def test_mask_branch_and_new_block_start_state(store_factory):
    store = store_factory("ab")
    logits = torch.tensor([[3.0, 1.0, -100.0, -100.0], [1.0, 3.0, -100.0, -100.0]])
    tokens, _ = decode_block(
        store, logits, torch.tensor([True, False]), store.initial_state, remaining_tokens=0
    )
    assert tokens.tolist() == [3, 1]
    mid, error = store.traverse_token_path([0], store.initial_state)
    assert not error
    tokens, end = decode_block(store, logits[1:], torch.tensor([False]), mid, remaining_tokens=0)
    assert tokens.tolist() == [1]
    assert end in store.final_states


def test_completion_requires_exact_budget_without_padding_tokens(store_factory):
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = None
    store = store_factory("a|bb", tokenizer=tokenizer)
    tokens, _ = decode_block(
        store,
        torch.tensor([[9.0, 1.0, -100.0, -100.0]]),
        torch.tensor([False]),
        store.initial_state,
        remaining_tokens=1,
    )
    assert tokens.tolist() == [1]
