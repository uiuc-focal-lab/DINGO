import ast
import os

from z3 import And, If, Int, Not, Or, Real, RealVal, Solver, ToInt, ToReal, is_int, sat, unsat

import rust_dfa
from grammar_folder.grammar_to_regex import lark_to_regex

from .base_evaluator import BaseEvaluator


class GSMSymbolicEvaluator(BaseEvaluator):
    def __init__(self, dataset, do_cot=False):
        super().__init__(dataset, do_cot)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        suffix = "cot" if do_cot else "std"
        grammar_path = f"{parent_dir}/grammar_folder/gsm_symbolic_{suffix}.lark"
        if os.path.exists(grammar_path):
            with open(grammar_path, "r") as f:
                grammar_string = f.read()
            pattern_dict = lark_to_regex(grammar_string)
            pattern = pattern_dict["start"]
            self.dfa = rust_dfa.RegexDFA()
            self.dfa.initialize(pattern)

    def parse_completion_regex(self, response):
        try:
            return response[response.rfind("<<") : response.rfind(">>") + 2]
        except (AttributeError, TypeError):
            return None

    def extract_question_trace(self, response):
        try:
            start_idx = response.find("final answer")
            if start_idx == -1:
                return response
            end_idx = response.find(">>", start_idx)
            if end_idx == -1:
                return response
            return response[: end_idx + len(">>")]
        except (AttributeError, TypeError):
            return response

    def evaluate_answer(self, gen_output):
        full_llm_response = gen_output["llm_response"]

        llm_response = full_llm_response

        if self.do_cot:
            llm_response = self.extract_question_trace(full_llm_response)

        llm_response = llm_response.rstrip("\n.")
        parses = self.dfa.matches(llm_response) if llm_response is not None else False
        if self.do_cot:
            completion = self.parse_completion_regex(llm_response)
        else:
            completion = llm_response

        correct = False
        if parses and completion is not None:
            completion = completion.strip()
            if completion.startswith("<<") and completion.endswith(">>"):
                completion = completion[2:-2].strip()
                correct = validate_expression_equivalence(
                    completion,
                    gen_output["answer"],
                    ast.literal_eval(gen_output["variable_types"])
                    if isinstance(gen_output["variable_types"], str)
                    else gen_output["variable_types"],
                )

        results = {
            "question": gen_output["question"],
            "gold_answer": gen_output["answer"],
            "correct": correct,
            "parses": parses,
            "llm_response": llm_response,
            "full_llm_response": full_llm_response,
            "parsed_completion": completion,
            "time": gen_output["response_info"]["time"],
        }

        if "idx" in gen_output:
            results["idx"] = gen_output["idx"]
        return results


def Floor(x):
    # Z3 ToInt already rounds towards negative infinity.
    return x if is_int(x) else ToInt(x)


def Ceiling(x):
    return -Floor(-x)


def IntegerCheck(x):
    return x == Floor(x)


def _as_real(value):
    return ToReal(value) if is_int(value) else value


def _expression_to_z3(node, variables):
    """Translate the arithmetic AST and its definedness without text rewrites."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return RealVal(str(node.value)), True
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id], True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, defined = _expression_to_z3(node.operand, variables)
        return (-value if isinstance(node.op, ast.USub) else value), defined
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
    ):
        value, defined = _expression_to_z3(node.args[0], variables)
        return If(value >= 0, Floor(value), -Floor(-value)), defined
    if isinstance(node, ast.BinOp):
        left, l_defined = _expression_to_z3(node.left, variables)
        right, r_defined = _expression_to_z3(node.right, variables)
        defined = And(l_defined, r_defined)
        if isinstance(node.op, ast.Add):
            return left + right, defined
        if isinstance(node.op, ast.Sub):
            return left - right, defined
        if isinstance(node.op, ast.Mult):
            return left * right, defined
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            defined = And(defined, right != 0)
            quotient = _as_real(left) / _as_real(right)
            if isinstance(node.op, ast.Div):
                return quotient, defined
            if isinstance(node.op, ast.FloorDiv):
                return Floor(quotient), defined
            return left - right * Floor(quotient), defined
        if isinstance(node.op, ast.Pow):
            # Constant integral powers are exact rational arithmetic. Variable
            # or fractional powers cannot be certified by this translator.
            exponent = ast.literal_eval(node.right)
            if type(exponent) is not int or abs(exponent) > 100:
                raise ValueError("Unsupported exponent")
            if exponent == 0:
                return RealVal(1), defined
            if exponent < 0:
                return 1 / (_as_real(left) ** (-exponent)), And(defined, left != 0)
            return _as_real(left) ** exponent, defined
    raise ValueError("Unsupported symbolic expression")


def validate_expression_equivalence(expr1, expr2, var_types):
    """Certify equality under the benchmark's positive-variable domains.

    Unsupported expressions and solver unknowns return False; randomized
    evaluation is not a proof. Division/floor/modulo follow Python precedence,
    and int() truncates towards zero. Constants use exact real arithmetic.
    """
    try:
        trees = [ast.parse(expr, mode="eval").body for expr in (expr1, expr2)]
        names = {
            n.id
            for tree in trees
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id != "int"
        }
        variables, domain = {}, []
        for name in names:
            kind = var_types.get(name)
            if kind not in ("int", "float", "float between 0 and 1"):
                return False
            var = Int(name) if kind == "int" else Real(name)
            variables[name] = var
            domain.append(var > 0)
            if kind == "float between 0 and 1":
                domain.append(var <= 1)
        (a, da), (b, db) = [_expression_to_z3(tree, variables) for tree in trees]
    except (SyntaxError, ValueError, TypeError, ArithmeticError):
        return False
    solver = Solver()
    solver.set(timeout=5000)
    solver.add(*domain)
    # Avoid certifying two expressions that are nowhere defined.
    solver.push()
    solver.add(da, db)
    has_domain = solver.check() == sat
    solver.pop()
    if not has_domain:
        return False
    solver.add(Or(And(da, Not(db)), And(Not(da), db), And(da, db, a != b)))
    return solver.check() == unsat
