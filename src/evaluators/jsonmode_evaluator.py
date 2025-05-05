
from .base_evaluator import BaseEvaluator
import ast
import json
from jsonschema import validate, ValidationError




class JSONModeEvaluator(BaseEvaluator):
    def __init__(self, dataset, do_cot = False):
        super().__init__(dataset, do_cot)
    
    def parse_completion_regex(self, response):
        return response


    def evaluate_answer(self, gen_output):
        completion = gen_output['llm_response']
        json_schema = gen_output['schema']
        
        
        parses = False 
        error_type = "No Error"
        error_message = ""
        try:
            generated_json = json.loads(completion)
            parses = True
        except json.decoder.JSONDecodeError:
            try:
                generated_json = ast.literal_eval(completion)
                parses = True
            except (SyntaxError, ValueError) as e:
                generated_json = "" 
                error_message = e 
                error_type = "Syntax Error"
        
        correct = parses
        if parses:
            if isinstance(generated_json, list):
                for index, item in enumerate(generated_json):
                    try:
                        validate(instance=item, schema=json_schema)
                        
                    except ValidationError as e:
                        error_message = e
                        error_type = "Validation Error"
                        correct = False
                        break
            else: 
                try:
                    validate(instance=generated_json, schema=json_schema)
                except ValidationError as e:
                    error_message = e
                    error_type = "Validation Error"
                    correct = False
        
        results = {
            'question': gen_output['question'],
            'gold_answer': gen_output['completion'],
            'schema': json_schema,
            'correct': correct,
            'parses': parses,
            'error_type': error_type,
            'llm_response': completion,
            'generated_json': generated_json,
            'error_message': error_message,
            'time': gen_output['response_info']['time'], 
        }
        
        return results
