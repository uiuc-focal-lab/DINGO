import re
from pathlib import Path

import pytest

from grammar_folder.grammar_to_regex import LarkToRegexConverter, lark_to_regex


def test_recursion_is_rejected_before_expanding_references():
    converter = LarkToRegexConverter()
    assert converter._is_recursive({"expr": 'expr "+" term | term', "term": "/[0-9]+/"})
    assert not converter._is_recursive({"a": '"a"'})


def test_standard_math_grammar_exists_and_accepts_the_prompt_format():
    path = Path(__file__).resolve().parents[1] / "src/grammar_folder/gsm_symbolic_std.lark"
    pattern = lark_to_regex(path.read_text(), "start")
    assert re.fullmatch(pattern, "<<a + b>>")
    assert not re.fullmatch(pattern, "<<a + >>")


def test_escaped_literals_and_regex_alternation_preserve_language():
    pattern = lark_to_regex('start: /a|b/ "c" "\\n"', "start")
    assert re.fullmatch(pattern, "ac\n")
    assert re.fullmatch(pattern, "bc\n")
    assert not re.fullmatch(pattern, "a")


def test_converter_reuse_does_not_retain_previous_rules():
    converter = LarkToRegexConverter()
    converter.convert('old: "a"\n')
    assert converter.convert('start: "b"\n') == {"start": "b"}


def test_undefined_rule_reference_is_rejected():
    with pytest.raises(ValueError, match="Undefined"):
        lark_to_regex("start: missing\n")
