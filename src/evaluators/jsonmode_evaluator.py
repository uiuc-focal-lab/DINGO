
from .base_evaluator import BaseEvaluator
import ast
import json
from jsonschema import validate, ValidationError, SchemaError




class JSONModeEvaluator(BaseEvaluator):
    def __init__(self, dataset, do_cot = False):
        super().__init__(dataset, do_cot)
    
    def parse_completion_regex(self, response):
        return response


    def evaluate_answer(self, gen_output):
        completion = gen_output['llm_response'].replace('```json', '').replace('```', '')
        json_schema = json.loads(gen_output['schema'])
        
        
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
        
        question = gen_output['prompt'][0]['content'] + '\n' + gen_output['prompt'][1]['content']
        if not isinstance(error_message, str):
            if hasattr(error_message, 'message'):
                error_message = error_message.message
            else:
                error_message = str(error_message)
        results = {
            'question': question,
            'gold_answer': gen_output['completion'],
            'schema': gen_output['schema'],
            'correct': correct,
            'parses': parses,
            'error_type': error_type,
            'llm_response': completion,
            'generated_json': str(generated_json),
            'error_message': error_message,
            'time': gen_output['response_info']['time'], 
        }
        
        return results
