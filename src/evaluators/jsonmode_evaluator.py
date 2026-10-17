import json

from jsonschema import SchemaError, ValidationError, validate

from .base_evaluator import BaseEvaluator


class JSONModeEvaluator(BaseEvaluator):
    def parse_completion_regex(self, response):
        return response

    def evaluate_answer(self, gen_output):
        response = gen_output["llm_response"]
        parses, correct = False, False
        error_type, error_message = "No Error", ""
        generated = None
        schema = gen_output["schema"]
        if isinstance(schema, str):
            schema = json.loads(schema)
        try:
            # Python's JSON parser otherwise accepts NaN/Infinity, which JSON
            # does not. Python literals and surrounding prose are not JSON.
            def reject_constant(value):
                raise ValueError(f"Invalid JSON constant: {value}")

            generated = json.loads(response, parse_constant=reject_constant)
            parses = True
        except (ValueError, TypeError) as exc:
            error_type, error_message = "Syntax Error", str(exc)
        if parses:
            try:
                validate(instance=generated, schema=schema)
                correct = True
            except (ValidationError, SchemaError) as exc:
                error_type, error_message = "Validation Error", exc.message
        prompt = gen_output["prompt"]
        question = prompt if isinstance(prompt, str) else "\n".join(m["content"] for m in prompt)
        result = {
            "question": question,
            "gold_answer": gen_output["completion"],
            "schema": gen_output["schema"],
            "correct": correct,
            "parses": parses,
            "error_type": error_type,
            "llm_response": response,
            "generated_json": str(generated),
            "error_message": error_message,
            "time": gen_output["response_info"]["time"],
        }
        if "idx" in gen_output:
            result["idx"] = gen_output["idx"]
        return result
