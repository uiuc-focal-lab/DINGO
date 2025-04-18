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
        expressions = self.extract_expressions(text)
        
        if not expressions:
            return True  # No expressions found
        
        all_valid = True
        for expr in expressions:
            if not self.is_valid_expression(expr):
                all_valid = False
        
        return all_valid        
    
        
        