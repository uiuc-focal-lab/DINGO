import re

# Define the operator subpattern for convenience:
op = r'(?:[+\-]|\/\/|\/|%|\*|\*\*)'

# E0: expression with zero parentheses
#     operand = single letter OR up to 3 digits
#     then zero or more [ operator + operand ]
E0 = (
    r'(?:[a-z]|[0-9]{1,3})'
    r'(?:( )?{}' + op + r'( )?(?:[a-z]|[0-9]{1,3}))*'
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


pattern = re.compile(cot)

# ------------------
# Test it:
tests = [
    "Let's think step by step: <<a/(100 * c)>>.",
   
]

for t in tests:
    m = bool(pattern.match(t))
    print(f"{m}")