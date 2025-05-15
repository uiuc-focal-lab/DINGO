
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
        try:
            json.loads(gen_output['llm_response']) 
            completion = gen_output['llm_response']
        except Exception as e:
            try:
                ast.literal_eval(gen_output['llm_response'])
                completion = gen_output['llm_response']
                
            except Exception as e:
                if '```json' in gen_output['llm_response']: 
                    completion = gen_output['llm_response'][gen_output['llm_response'].find('```json') + len('```json'):]
                    if '```' in completion:
                        completion = completion[:completion.find('```')]
                elif '```' in gen_output['llm_response']:
                    completion = gen_output['llm_response'][gen_output['llm_response'].find('```') + len('```'):]
                    if '```' in completion:
                        completion = completion[:completion.find('```')]
                else:
                    if '{' in gen_output['llm_response'] and '}' in gen_output['llm_response']:
                        completion = gen_output['llm_response']
                        completion = completion[completion.find('{'):completion.rfind('}') + 1]
                    else:
                        completion = gen_output['llm_response']

        json_schema = json.loads(gen_output['schema'])
        
        
        parses = False 
        error_type = "No Error"
        error_message = ""
        try:
            generated_json = json.loads(completion)
            parses = True
        except Exception as e:
            try:
                generated_json = ast.literal_eval(completion)
                parses = True
            except Exception as e:
                generated_json = "" 
                error_message = e 
                error_type = "Syntax Error"
        
        correct = parses
        if parses:
            if isinstance(generated_json, list):
                for index, item in enumerate(generated_json):
                    try:
                        validate(instance=item, schema=json_schema)
                        
                    except Exception as e:
                        error_message = e
                        error_type = "Validation Error"
                        correct = False
                        break
            else: 
                try:
                    validate(instance=generated_json, schema=json_schema)
                except Exception as e:
                    error_message = e
                    error_type = "Validation Error"
                    correct = False
        
        if isinstance(gen_output['prompt'], str):
            question = gen_output['prompt']
        else:
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
