from sympy import simplify
from .base_evaluator import BaseEvaluator
from z3 import Solver, unsat, Real, ToInt, ToReal, And, If, Int, unknown
import re
import random
from copy import deepcopy
import signal

class GSMSymbolicEvaluator(BaseEvaluator):
    def parse_completion_regex(self, response):
        try:
            return response[response.rfind('<<') : response.rfind('>>') + 2]
        except:
            return None  
    
    def seperate_reasoning_answer(self, response):
        try:
            return response[:response.rfind('<<')].strip(), None
        except:
            return response, None
    
    def evaluate_answer(self, gen_output):
        completion = gen_output['llm_response']
        correct = False
        if completion is not None:
            completion = completion.strip()
            if completion.startswith('<<') and completion.endswith('>>'):
                completion = completion[2:-2].strip()
                correct = validate_expression_equivalence(completion, gen_output['answer'], eval(gen_output['variable_types'])) 
        
        results = {
            'question': gen_output['question'],
            'gold_answer': gen_output['answer'],
            'correct': correct,
            'llm_response': gen_output['llm_response'],
            'parsed_completion': completion, 
            'time': gen_output['response_info']['time'], 
        }
        
        return results


def timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")

def eval_with_timeout(expr, timeout=2):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = eval(expr)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        return None  # or any other value to indicate timeout



def floor_div_replacer(expression):
    regex_with_groups = r"(?P<left>.+?)\s*//\s*(?P<right>.+)"
    def replace_floor_div(match):
        left = match.group('left').strip()
        right = match.group('right').strip()
        return f"z3_floor_div({left}, {right})"
    return re.sub(regex_with_groups, replace_floor_div, expression)


def IntegerCheck(x):
    return And(x == Floor(x), x == Ceiling(x))

def Floor(x):
    return If(x >= 0, 
              ToInt(x), 
              ToInt(x) - If(ToReal(ToInt(x)) == x, 0, 1))

def Ceiling(x):
    return If(x >= 0, 
              ToInt(x) + If(ToReal(ToInt(x)) == x, 0, 1), 
              ToInt(x))

def test_expression_equivalence(expr1_gsm, expr2_gsm, var_names, var_types): 
    test_cases = []
    for _ in range(1000):
        test_case = {}
        for var in var_names:
            if var_types[var] == 'float between 0 and 1':
                test_case[var] = random.uniform(0.001, 1)
            elif var_types[var] == 'float':
                test_case[var] = random.uniform(0.001, 100)
            elif var_types[var] == 'int':
                test_case[var] = random.randint(1, 100)
        test_cases.append(test_case)
    for test_case in test_cases:
        expr1_substituted = expr1_gsm
        expr2_substituted = expr2_gsm
        for var, value in test_case.items():
            expr1_substituted = re.sub(rf'\b{var}\b', str(value), expr1_substituted)
            expr2_substituted = re.sub(rf'\b{var}\b', str(value), expr2_substituted)
        try:
            ans1_gsm = eval(expr1_substituted)
        except:
            return False
        try:
            ans2_gsm = eval(expr2_substituted)
        except:
            return True
        if ans1_gsm != ans2_gsm:
            print(f"Test case {test_case} failed.")
            print(f"Expression 1: {expr1_gsm} = {ans1_gsm}")
            print(f"Expression 2: {expr2_gsm} = {ans2_gsm}")
            return False 
    print("All test cases passed.")
    return True
    

def validate_expression_equivalence(expr1, expr2, var_types):
    original_expr1 = expr1
    original_expr2 = expr2
    
    var_names = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr1 + ' ' + expr2))
    var_names -= {'int'} 
    print(var_names)
    
    if original_expr1 == "int(p * (1 + r1/100) * (1 - r2/100)) * n":
        return test_expression_equivalence(original_expr1, original_expr2, var_names, var_types)
    
    vars_dict = {}
    constraints = []
    for name in var_names:
        var = Real(name)
        vars_dict[name] = var
        var_type = var_types.get(name, 'str')
        if var_type == 'float between 0 and 1':
            constraints.append(var > 0)
            constraints.append(var <= 1)
        elif var_type == 'float':
            constraints.append(var > 0)
        elif var_type == 'int':
            constraints.append(var > 0)
            constraints.append(IntegerCheck(var))
        else:
            return False
    
    expr1 = re.sub(r'\bint\(', 'ToInt(', expr1)
    expr2 = re.sub(r'\bint\(', 'ToInt(', expr2)
    
    if 'round(' in expr1:
        return False
    
    expr2 = re.sub(r'\round\(', 'ToInt(', expr2)
    
    if '//' in expr1:
        expr1 = floor_div_replacer(expr1)
    if '//' in expr2:
        expr2 = floor_div_replacer(expr2)

    print(vars_dict)
    def z3_floor_div(x, y):
        return If(y != 0, ToInt(x / y), 0)  

    def safe_eval(expr):
        return eval(expr, {"__builtins__": None}, {**vars_dict, 'ToInt': ToInt, 'z3_floor_div': z3_floor_div})

    try:
        expr2_z3 = safe_eval(expr2)
    except:
        return test_expression_equivalence(original_expr1, original_expr2, var_names, var_types)
    
    
    try:
        expr1_z3 = safe_eval(expr1)
    except:
        return test_expression_equivalence(original_expr1, original_expr2, var_names, var_types)
    
    s = Solver()
    s.set("timeout", 5000)
    s.add(constraints)
    try:
        s.add(expr1_z3 != expr2_z3)
    except:
        return test_expression_equivalence(original_expr1, original_expr2, var_names, var_types)
    
    print('added constraints')
    result = s.check()
    if result == unsat:
        print(f"LLM expression {expr1} and GT expression {expr2} are equivalent.")
        return True
    elif result == unknown:
        print("Solver timed out.")
        return test_expression_equivalence(original_expr1, original_expr2, var_names, var_types)
    else:
        print(f"LLM expression {expr1} and GT expression {expr2} are not equivalent.")
        print("Counter-example:")
        model = s.model()
        for var in vars_dict:
            print(f"{var} = {model[vars_dict[var]]}")
        return False