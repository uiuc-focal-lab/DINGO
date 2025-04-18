# test_regex_dfa.py
import unittest
import rust_dfa

class TestRegexDFA(unittest.TestCase):
    def test_initialization(self):
        # Test with simple regex
        dfa = rust_dfa.RegexDFA()
        size = dfa.initialize("a(b|c)*d")
        print(f"DFA size for 'a(b|c)*d': {size}")
        
        # Test with more complex regex
        dfa2 = rust_dfa.RegexDFA()
        size2 = dfa2.initialize("(a|b)*c(d|e)+f?")
        print(f"DFA size for '(a|b)*c(d|e)+f?': {size2}")
        
        # Test with very specific pattern
        dfa3 = rust_dfa.RegexDFA()
        size3 = dfa3.initialize("a{3}b{2,4}c{1,}")
        print(f"DFA size for 'a{3}b{2,4}c{1,}': {size3}")

if __name__ == "__main__":
    unittest.main()