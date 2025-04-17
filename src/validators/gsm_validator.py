import re
from syncode.larkm import Lark

# Step 1: Define the Lark grammar parser
grammar = '''
start: space? "<<" space? expr space? ">>" space?
expr: expr space? "+" space? term   
     | expr space? "-" space? term   
     | term

term: term space? "*" space? factor 
     | term space? "/" space? factor 
     | term space? "//" space? factor 
     | term space? "%" space? factor  
     | factor space?

factor: "-" space? factor    
       | primary space?
primary: NUMBER        
        | VARIABLE      
        | "(" space? expr space? ")"

space: " "
%import common.CNAME -> VARIABLE
%import common.NUMBER
'''

# Create the parser
expr_parser = Lark(grammar, start='start')

# Step 2: Function to extract expressions
def extract_expressions(text):
    pattern = r'<<\s*(.*?)\s*>>'
    return re.findall(pattern, text, re.DOTALL)

# Step 3: Function to validate an expression with the Lark parser
def is_valid_expression(expr):
    try:
        # Wrap the expression in << >> for the parser
        expr_to_parse = f"<< {expr} >>"
        expr_parser.parse(expr_to_parse)
        return True
    except Exception as e:
        print(f"Invalid expression: '{expr}'. Error: {e}")
        return False

# Main function to validate text
def validate_gsm_symbolic(text):
    expressions = extract_expressions(text)
    
    if not expressions:
        return True  # No expressions found
    
    all_valid = True
    for expr in expressions:
        if not is_valid_expression(expr):
            all_valid = False
    
    return all_valid

# Usage example
def test_with_grammar(tests):
    for i, t in enumerate(tests):
        valid = validate_gsm_symbolic(t)
        print(f"Test {i+1}: {valid}")


# Run the tests
if __name__ == "__main__":
    # Test cases
    tests = [
        "Let me think step by step. The selling price of the property is ${c}. The brokerage fee is {a}% of the selling price, which is <<a/100 * c>>. The transfer fee is {d}% of the selling price, which is <<d/102 * c>>. The total price of the property including the brokerage and transfer fees is <<c + a/102 * c + d/102 * c>>. The final answer is <<(c + a/102 * c + d/102 * c) - b>>.",
        "Let's think step by step. The selling price of the property is ${c}. The brokerage fee is {a}% of the selling price, which is <<a/100 * c>>. The transfer fee is {d}% of the selling price, which is <<d/100 * c>>. The total price of the property including the brokerage and transfer fees is <<c + a/100 * c + d/100 * c>>. The final answer is <<(c + a/100 * c + d/100 * c) - b>>.",
        "This test has invalid expressions: <<a + b + 2c>>.",  # Invalid syntax
        "This test has no expressions.",  # No expressions
    ]

    test_with_grammar(tests)