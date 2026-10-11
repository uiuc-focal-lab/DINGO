import re
import unittest

from grammar_folder.grammar_to_regex import lark_to_regex


class TestLarkToRegex(unittest.TestCase):
    def test_simple_literal(self):
        """Test conversion of a simple string literal rule."""
        grammar = 'greeting: "hello"\n'
        regex = lark_to_regex(grammar, "greeting")

        self.assertEqual(regex, "hello")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "hello"))
        self.assertFalse(re.fullmatch(regex, "hello world"))

    def test_alternative(self):
        """Test conversion of alternatives."""
        grammar = 'greeting: "hello" | "hi"\n'
        regex = lark_to_regex(grammar, "greeting")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "hello"))
        self.assertTrue(re.fullmatch(regex, "hi"))
        self.assertFalse(re.fullmatch(regex, "hey"))

    def test_sequence(self):
        """Test conversion of a sequence."""
        grammar = 'greeting: "hello" "world"\n'
        regex = lark_to_regex(grammar, "greeting")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "helloworld"))
        self.assertFalse(re.fullmatch(regex, "hello"))
        self.assertFalse(re.fullmatch(regex, "world"))

    def test_modifiers(self):
        """Test modifiers (?, *, +)."""
        # Optional modifier
        grammar = 'item: "a" "b"?\n'
        regex = lark_to_regex(grammar, "item")
        self.assertTrue(re.fullmatch(regex, "a"))
        self.assertTrue(re.fullmatch(regex, "ab"))
        self.assertFalse(re.fullmatch(regex, "abb"))

        # Zero or more modifier
        grammar = 'item: "a" "b"*\n'
        regex = lark_to_regex(grammar, "item")
        self.assertTrue(re.fullmatch(regex, "a"))
        self.assertTrue(re.fullmatch(regex, "ab"))
        self.assertTrue(re.fullmatch(regex, "abbb"))

        # One or more modifier
        grammar = 'item: "a" "b"+\n'
        regex = lark_to_regex(grammar, "item")
        self.assertFalse(re.fullmatch(regex, "a"))
        self.assertTrue(re.fullmatch(regex, "ab"))
        self.assertTrue(re.fullmatch(regex, "abbb"))

    def test_regexp(self):
        """Test regular expression terminals."""
        grammar = "num: /[0-9]+/\n"
        regex = lark_to_regex(grammar, "num")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "123"))
        self.assertFalse(re.fullmatch(regex, "abc"))

    def test_rule_references(self):
        """Test references to other rules."""
        grammar = """
            sentence: subject verb object
            subject: "I" | "you" | "they"
            verb: "like" | "love"
            object: "pizza" | "pasta"
        """
        regex = lark_to_regex(grammar, "sentence")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "Ilikepizza"))
        self.assertTrue(re.fullmatch(regex, "youlovepasta"))
        self.assertFalse(re.fullmatch(regex, "Ilikedogs"))

    def test_grouping(self):
        """Test grouping with parentheses."""
        grammar = 'expr: "a" ("b" | "c")\n'
        regex = lark_to_regex(grammar, "expr")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "ab"))
        self.assertTrue(re.fullmatch(regex, "ac"))
        self.assertFalse(re.fullmatch(regex, "ad"))

    def test_complex_grammar(self):
        """Test a more complex grammar."""
        grammar = """
            expr: term ("+" term | "-" term)*
            term: factor ("*" factor | "/" factor)*
            factor: number | "(" expr ")"
            number: /[0-9]+/
        """
        with self.assertRaises(ValueError) as context:
            lark_to_regex(grammar, "expr")

        # Check that it was detected as recursive
        self.assertIn("Recursive grammar detected", str(context.exception))

    def test_recursive_detection(self):
        """Test detection of recursive grammars."""
        # Direct recursion
        grammar = """
            expr: term | expr "+" term
            term: /[0-9]+/
        """
        with self.assertRaises(ValueError):
            lark_to_regex(grammar)

        # Indirect recursion
        grammar = """
            a: b "x"
            b: c "y"
            c: "z" | a
        """
        with self.assertRaises(ValueError):
            lark_to_regex(grammar)

    def test_special_chars(self):
        """Test handling of regex special characters in literals."""
        grammar = 'special: "a.b[c]d(e)f{g}h"\n'
        regex = lark_to_regex(grammar, "special")

        # The regex should escape special characters
        self.assertTrue(re.fullmatch(regex, "a.b[c]d(e)f{g}h"))

    def test_multiple_rules(self):
        """Test returning multiple rules as a dictionary."""
        grammar = """
            rule1: "abc"
            rule2: "def"
        """
        regex_dict = lark_to_regex(grammar)

        self.assertIn("rule1", regex_dict)
        self.assertIn("rule2", regex_dict)
        self.assertTrue(re.fullmatch(regex_dict["rule1"], "abc"))
        self.assertTrue(re.fullmatch(regex_dict["rule2"], "def"))

    def test_non_recursive_repetition(self):
        """Test grammar with repetition that is not truly recursive."""
        grammar = """
            list: item ("," item)*
            item: "a" | "b" | "c"
        """
        regex = lark_to_regex(grammar, "list")

        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, "a"))
        self.assertTrue(re.fullmatch(regex, "a,b"))
        self.assertTrue(re.fullmatch(regex, "a,b,c"))
        self.assertFalse(re.fullmatch(regex, "a,b,d"))
