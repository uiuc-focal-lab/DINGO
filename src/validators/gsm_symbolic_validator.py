from .base_validator import BaseValidator
import re

class GSMSymbolicValidator(BaseValidator):
    def extract_expressions(self, text):
        pattern = r'<<\s*(.*?)\s*>>'
        return re.findall(pattern, text, re.DOTALL)
    
    def is_valid_expression(self, expr):
        try:
            # Wrap the expression in << >> for the parser
            expr_to_parse = f"<< {expr} >>"
            self._parse(expr_to_parse)
            return True
        except Exception as e:
            print(f"Invalid expression: '{expr}'. Error: {e}")
            return False
    
    def validate(self, text):
        # 1) Make sure every '<<' has a matching '>>'
        if text.count('<<') != text.count('>>'):
            print("Mismatched '<<' and '>>' markers")
            return False

        # 2) Remove all properly paired markers and ensure no stray '<' or '>'
        cleaned = text.replace('<<', '').replace('>>', '')
        if '<' in cleaned or '>' in cleaned:
            print("Found a stray '<' or '>' that isn't part of a '<<'/'>>' pair")
            return False
        
        # 3) Extract & parse each expression as before
        expressions = self.extract_expressions(text)
        if not expressions:
            return True  # nothing to validate, and markers are balanced
        
        all_valid = True
        for expr in expressions:
            if not self.is_valid_expression(expr):
                all_valid = False
        
        return all_valid
    
    
        
        