from types import SimpleNamespace

import torch
from helpers import TinyTokenizer

from models.gen_utils import completed_stop_prefix, find_target_after_phrase


def test_stop_phrase_and_delimiter_can_share_one_token():
    tok = SimpleNamespace(
        decode=lambda ids, **kwargs: "".join(
            ["THE FINAL ANSWER is <<a>>", " suffix"][int(i)] for i in ids
        )
    )
    assert find_target_after_phrase([0, 1], tok, "The final answer", ">>") == 0


def test_early_stop_keeps_original_ids_and_requires_acceptance(store_factory):
    x = torch.tensor([[2, 0, 1, 2]])
    stopped = completed_stop_prefix(x, 1, 4, TinyTokenizer(), 3, "a", "b", store_factory("ab"))
    assert stopped.tolist() == [[2, 0, 1]]
    assert completed_stop_prefix(x, 1, 4, TinyTokenizer(), 3, "", "a", store_factory("ab")) is None


def test_early_stop_waits_for_revealed_prefix(store_factory):
    assert (
        completed_stop_prefix(
            torch.tensor([[2, 0, 3]]), 1, 3, TinyTokenizer(), 3, "", "a", store_factory("a")
        )
        is None
    )


def test_static_reveal_schedule_preserves_budget_without_tensor_reads():
    from models import gen_utils

    assert gen_utils.reveal_schedule(16, 8) == [2] * 8
    assert gen_utils.reveal_schedule(5, 3) == [2, 2, 1]
    assert gen_utils.reveal_schedule(2, 4) == [1, 1, 0, 0]
