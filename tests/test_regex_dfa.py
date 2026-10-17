import pytest

import rust_dfa


def make_dfa(pattern):
    dfa = rust_dfa.RegexDFA()
    dfa.initialize(pattern)
    return dfa


@pytest.mark.parametrize(
    "pattern, valid, invalid",
    [
        ("a", ["a"], ["", "aa", "ab", "ba"]),
        ("a|ab", ["a", "ab"], ["abc", "ac", "ba"]),
        ("a*", ["", "a", "aaa"], ["b", "aab"]),
        ("(?i:a)", ["a", "A"], ["Ab"]),
        ("é", ["é"], ["éx", "xé"]),
    ],
)
def test_entire_response_must_match(pattern, valid, invalid):
    dfa = make_dfa(pattern)
    assert all(dfa.matches(s) for s in valid)
    assert not any(dfa.matches(s) for s in invalid)
    assert not any(dfa.prefix_matches(s) for s in invalid if s)


def test_reinitialization_discards_old_states_and_tokens():
    dfa = make_dfa("abc")
    dfa.compute_token_transitions({b"a": 0, b"b": 1, b"c": 2})
    dfa.size()
    dfa.initialize("z")
    assert dfa.token_transitions_size() == 0
    dfa.compute_token_transitions({b"z": 3})
    transitions = dfa.get_all_token_transitions()
    assert transitions[(dfa.get_initial_state(), 3)] in dfa.get_final_states()
    assert dfa.matches("z")
    assert not dfa.matches("abc")


def test_token_transition_consumes_all_bytes():
    dfa = make_dfa("a")
    dfa.compute_token_transitions({b"a": 0, b"ab": 1})
    assert (dfa.get_initial_state(), 1) not in dfa.get_all_token_transitions()


def test_equal_byte_tokens_keep_every_token_id():
    dfa = make_dfa("a")
    dfa.compute_token_transitions({b"a": [0, 1]})
    transitions = dfa.get_all_token_transitions()
    assert transitions[(dfa.get_initial_state(), 0)] == transitions[(dfa.get_initial_state(), 1)]
