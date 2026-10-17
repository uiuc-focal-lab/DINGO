import pytest

from result_utils import align_results, better_result


def test_comparisons_align_dataset_ids_instead_of_completion_order():
    first = [{"idx": 42, "correct": True}, {"idx": 7, "correct": False}]
    second = [{"idx": 7, "correct": True}, {"idx": 42, "correct": False}]
    assert [(a["idx"], b["idx"]) for a, b in align_results(first, second)] == [(7, 7), (42, 42)]


def test_mismatched_and_duplicate_result_ids_are_rejected():
    with pytest.raises(ValueError, match="same"):
        align_results([{"idx": 1}], [{"idx": 2}])
    with pytest.raises(ValueError, match="duplicate"):
        align_results([{"idx": 1}, {"idx": 1}], [{"idx": 1}])


def test_oracle_retains_time_of_selected_result():
    correct = {"correct": True, "parses": True, "time": 10}
    fast = {"correct": False, "parses": True, "time": 1}
    assert better_result(correct, fast) == correct
