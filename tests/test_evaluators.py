import pytest
from z3 import RealVal, simplify

from evaluators.gsm_evaluator import Floor, validate_expression_equivalence
from evaluators.jsonmode_evaluator import JSONModeEvaluator


def json_result(response, schema):
    return JSONModeEvaluator("json_mode").evaluate_answer(
        {
            "llm_response": response,
            "schema": schema,
            "prompt": "test",
            "completion": "",
            "response_info": {"time": 0.0},
            "idx": 23,
        }
    )


def test_json_array_is_validated_as_a_whole():
    result = json_result("[1, 2]", '{"type":"array","items":{"type":"integer"}}')
    assert result["parses"] and result["correct"]
    assert not json_result("[]", '{"type":"object"}')["correct"]


def test_python_literals_are_not_json():
    result = json_result("{'a': 1}", '{"type":"object"}')
    assert not result["parses"]
    assert not result["correct"]


def test_resume_identity_is_preserved_in_evaluator_results():
    assert json_result("{}", '{"type":"object"}')["idx"] == 23


def test_floor_on_negative_reals_is_not_double_rounded():
    assert simplify(Floor(RealVal("-1.2"))).as_long() == -2


@pytest.mark.parametrize(
    "expr, gold, expected",
    [
        ("a // 2 + 1", "(a // 2) + 1", True),
        ("a // 2 + 1", "a // (2 + 1)", False),
        ("a % 3", "a - 3 * (a // 3)", True),
        ("int(-a / 2)", "-int(a / 2)", True),
        ("a / (b - b)", "a", False),
        ("a", "bad(a)", False),
        ("0 ** 0", "1", True),
        ("(a-a) ** 0", "1", True),
    ],
)
def test_symbolic_operations_preserve_python_precedence_and_definedness(expr, gold, expected):
    assert validate_expression_equivalence(expr, gold, {"a": "int", "b": "int"}) is expected
