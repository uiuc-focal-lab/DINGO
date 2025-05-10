
from .base_evaluator import BaseEvaluator
import numpy as np
from uuid import UUID
from json import loads
from typing import List, Tuple
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError
from typing import Dict, Any
import json

Schema = Dict[str, Any]



def is_json_schema_valid(schema: Schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True, "", "No Error"
    except SchemaError as e:
        return False, e, "Schema Error"


format_checker = FormatChecker()


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def validate_json_schema(instance: Schema, schema: Schema) -> bool:
    valid, error, error_type = is_json_schema_valid(schema)
    if not valid:
        return False, error, error_type
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception as e:
        return False, e, "Validation Error"
    return True, "", "No Error"


class SchemaBenchEvaluator(BaseEvaluator):
    def __init__(self, dataset, do_cot = False):
        super().__init__(dataset, do_cot)
    
    def parse_completion_regex(self, response):
        return response


    def evaluate_answer(self, gen_output):
        completion = gen_output['llm_response']
        json_schema = json.loads(gen_output['json_schema'])
        
        
        parses = True 
        
        error_type = "No Error"
        error_message = ""
       
        
        try:
            json_object = loads(completion)
        except Exception as e:
            error_message = e
            error_type = "JSON Decoding Error"
            parses = False
            json_object = ''
            
        correct = parses
        if parses:
            correct, error_message, error_type = validate_json_schema(json_object, json_schema)
        
        if not isinstance(error_message, str):
            if hasattr(error_message, 'message'):
                error_message = error_message.message
            else:
                error_message = str(error_message)
        
        results = {
            'question': gen_output['question'],
            'schema': gen_output['json_schema'],
            'correct': correct,
            'parses': parses,
            'error_type': error_type,
            'llm_response': completion,
            'generated_json': str(json_object),
            'error_message': error_message,
            'time': gen_output['response_info']['time'], 
        }
        
        return results