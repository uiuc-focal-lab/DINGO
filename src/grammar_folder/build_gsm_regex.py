import os
import re
import time
import unittest
from syncode.larkm import Lark
from grammar_to_regex import lark_to_regex
import rust_dfa


class GSMSymbolicRegex:
    """Class to handle expression matching using both regex approaches"""
    
    @staticmethod
    def build_regex_old():
        """Build a regex pattern using the direct nested approach"""
        # Define the operator subpattern for convenience
        op = r'(?:[+\-]|\/\/|\/|%|\*|\*\*)'
        
        # Build expressions of increasing nesting depth
        E0 = (
            r'(?:[a-z]|[0-9]{1,3})'
            r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}))*'
        )
        
        E1 = (
            r'(?:[a-z]|[0-9]{1,3}|\(' + E0 + r'\))'
            r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E0 + r'\)))*'
        )
        
        E2 = (
            r'(?:[a-z]|[0-9]{1,3}|\(' + E1 + r'\))'
            r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E1 + r'\)))*'
        )
        
        E3 = (
            r'(?:[a-z]|[0-9]{1,3}|\(' + E2 + r'\))'
            r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E2 + r'\)))*'
        )
        
        pattern_str = r'<<( )?' + E3 + r'( )?>>'
        cot = r'(' + r"([0-9]|[A-z]|[a-z]|,|\.|( )|{|}|\$|%|\'|\:)*" + pattern_str + r')+\.'
        
        return re.compile(cot)
    
    @staticmethod
    def grammar_regex():
        """Define a Lark grammar that represents the same pattern as the direct regex"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = f"{current_dir}/gsm_symbolic_simpl.lark"
        with open(grammar_path, 'r') as file:
            grammar_string = file.read()
        return grammar_string


class RegexTest(unittest.TestCase):
    """Test cases for expression regex matching"""
    
    # These will be set from main() before tests run
    direct_pattern = None
    lark_pattern = None
    dfa = None
    
    @classmethod
    def setUpClass(cls):
        # Test data
        cls.test_cases = [
            # Should match (True)
            "Let's think step by step: <<a+2*x>>.",
            "Let's think step by step. We know there are {b} girls in the {s1}. We also know there are {a} the number of boys in the {s1}. To find the total number of kids in the {s1}, we need to add the number of girls and the number of boys. Since the number of boys is {a} times the number of girls, the number of boys is <<a * b>>. Therefore, the total number of kids in the {s1} is <<b + a * b>>. The final answer is <<b + a * b>>.",
            # Should not match (False)
            "Let's think step by step: <<a+2x>> this one has issue.",
            "Let's think step by step: <<a+2*x = 3*2>> this one has issue."
        ]
    
    def test_direct_regex(self):
        """Test the direct regex approach"""
        results = []
        match_times = []
        
        for test_str in self.test_cases:
            start_time = time.time()
            match = bool(self.direct_pattern.search(test_str))
            match_time = time.time() - start_time
            match_times.append(match_time)
            results.append(match)
        
        # First two should match, last two should not
        self.assertTrue(results[0], f"Failed on case 0: {self.test_cases[0]}")
        self.assertTrue(results[1], f"Failed on case 1: {self.test_cases[1]}")
        self.assertFalse(results[2], f"Failed on case 2: {self.test_cases[2]}")
        self.assertFalse(results[3], f"Failed on case 3: {self.test_cases[3]}")
        
        print(f"\nDirect Regex Match Times: {[f'{t*1000:.2f}ms' for t in match_times]}")
    
    def test_dfa_regex(self):
        """Test the Lark-based DFA approach"""
        results = []
        match_times = []
        
        for test_str in self.test_cases:
            start_time = time.time()
            match = self.dfa.matches(test_str)
            match_time = time.time() - start_time
            match_times.append(match_time)
            results.append(match)
        
        # First two should match, last two should not
        self.assertTrue(results[0], f"Failed on case 0: {self.test_cases[0]}")
        self.assertTrue(results[1], f"Failed on case 1: {self.test_cases[1]}")
        self.assertFalse(results[2], f"Failed on case 2: {self.test_cases[2]}")
        self.assertFalse(results[3], f"Failed on case 3: {self.test_cases[3]}")
        
        print(f"\nDFA Match Times: {[f'{t*1000:.2f}ms' for t in match_times]}")


def main():
    # Create timing variables first
    direct_compile_time = 0
    lark_compile_time = 0
    dfa_init_time = 0
    dfa_size = 0
    
    # Measure direct regex compilation time
    start_time = time.time()
    direct_pattern = GSMSymbolicRegex.build_regex_old()
    direct_compile_time = time.time() - start_time
    
    # Measure Lark grammar to regex compilation time
    start_time = time.time()
    lark_grammar = GSMSymbolicRegex.grammar_regex()
    lark_pattern_dict = lark_to_regex(lark_grammar)
    lark_pattern = lark_pattern_dict['start']
    lark_compile_time = time.time() - start_time
    
    # Measure DFA initialization time
    dfa = rust_dfa.RegexDFA()
    start_time = time.time()
    dfa_size = dfa.initialize(lark_pattern)
    dfa_init_time = time.time() - start_time
    
    # Print compilation time comparison
    print("\n===== REGEX PATTERN COMPILATION BENCHMARKS =====")
    print(f"Direct Regex Compilation Time: {direct_compile_time*1000:.2f}ms")
    print(f"Lark Grammar to Regex Compilation Time: {lark_compile_time*1000:.2f}ms")
    print(f"DFA Initialization Time: {dfa_init_time*1000:.2f}ms")
    print(f"DFA Size: {dfa_size}")
    
    # Print pattern information
    print("\n===== REGEX PATTERNS =====")
    print(f"Direct Regex Pattern Length: {len(direct_pattern.pattern)} characters")
    print(f"Lark-derived Regex Pattern Length: {len(lark_pattern)} characters")
    
    # Now initialize the test class with the patterns
    RegexTest.direct_pattern = direct_pattern
    RegexTest.lark_pattern = lark_pattern
    RegexTest.dfa = dfa
    
    # Run tests
    print("\n===== RUNNING TESTS =====")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    main()