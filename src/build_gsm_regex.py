import re
from syncode.larkm import Lark
from validators.grammar_to_regex import lark_to_regex
import rust_dfa

# Approach 1: Direct Regex Construction
def build_regex_direct():
    # Define the operator subpattern for convenience
    op = r'(?:[+\-]|\/\/|\/|%|\*|\*\*)'
    
    # E0: expression with zero parentheses
    #     operand = single letter OR up to 3 digits
    #     then zero or more [ operator + operand ]
    E0 = (
        r'(?:[a-z]|[0-9]{1,3})'
        r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}))*'
    )
    
    # E1: expression allowing parentheses at depth 1
    #     operand = a letter OR digits OR ( E0 )
    E1 = (
        r'(?:[a-z]|[0-9]{1,3}|\(' + E0 + r'\))'
        r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E0 + r'\)))*'
    )
    
    # E2: expression allowing up to depth 2
    #     operand = a letter OR digits OR ( E1 )
    E2 = (
        r'(?:[a-z]|[0-9]{1,3}|\(' + E1 + r'\))'
        r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E1 + r'\)))*'
    )
    
    # E3: expression allowing up to depth 3
    #     operand = a letter OR digits OR ( E2 )
    E3 = (
        r'(?:[a-z]|[0-9]{1,3}|\(' + E2 + r'\))'
        r'(?:( )?' + op + r'( )?(?:[a-z]|[0-9]{1,3}|\(' + E2 + r'\)))*'
    )
    
    pattern_str = r'<<( )?' + E3 + r'( )?>>'
    cot = r'(' + r"([0-9]|[A-z]|[a-z]|,|\.|( )|{|}|\$|%|\'|\:)*" + pattern_str + r')+\.'
    
    return re.compile(cot)

# Definition of Lark grammar for the expression pattern
def define_lark_grammar():
    """Define a Lark grammar that represents the same pattern as the direct regex"""
    grammar = r"""
    start: cot
    
    cot: (char | delim_expr)+ "."
    
    delim_expr: "<<" space? expr space? ">>"
    
    expr: e3
    
    e3: operand3 (space? operator space? operand3)*
    e2: operand2 (space? operator space? operand2)*
    e1: operand1 (space? operator space? operand1)*
    e0: operand0 (space? operator space? operand0)*
    
    operand3: letter | number | "(" e2 ")"
    operand2: letter | number | "(" e1 ")"
    operand1: letter | number | "(" e0 ")"
    operand0: letter | number
    
    operator: "+" | "-" | "//" | "/" | "%" | "*" | "**"
    
    letter: /[a-z]/
    number: /[0-9]{1,3}/
    char: /[0-9A-Za-z,.\$%': {}]/
    
    space: " "
    """
    return grammar

# Test both approaches
def test_regex_patterns():
    # Create patterns using both approaches
    direct_pattern = build_regex_direct()
    
    # For the Lark approach, we're assuming the grammar string is passed to the existing lark_to_regex function
    lark_grammar = define_lark_grammar()
    lark_pattern = lark_to_regex(lark_grammar)
    
    print("Direct Regex Pattern:")
    print(direct_pattern.pattern)
    print("Lark Regex Pattern:")
    print(lark_pattern['start'])

    dfa = rust_dfa.RegexDFA()
    size = dfa.initialize(lark_pattern['start'])
    print(f"DFA size: {size}")

    # Define test cases
    tests = [
        "Let's think step by step: <<a+2*x>>.",
        "Let's think step by step. We know there are {b} girls in the {s1}. We also know there are {a} the number of boys in the {s1}. To find the total number of kids in the {s1}, we need to add the number of girls and the number of boys. Since the number of boys is {a} times the number of girls, the number of boys is <<{a}*{b}>>. Therefore, the total number of kids in the {s1} is <<{b}+{a}*{b}>>. The final answer is <<{b}+{a}*{b}>>."
    ]
    
    out = dfa.matches(tests[0])
    print(out)
    out = dfa.matches(tests[1])
    print(out)

if __name__ == "__main__":
    test_regex_patterns()