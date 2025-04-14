import re
import json 
import os
import unittest
from typing import List

class GSMSymbolicTest(unittest.TestCase):
    """Test class for GSM symbolic expressions validation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = r'<<( )?(\(){0,3}( )?([a-z]|[0-9]{1,3})( )?(\)){0,3}(( )?([+\-]|\/\/|\/|%|\*|\*\*)( )?((\(){0,3}( )?)?([a-z]|[0-9]{1,3})( )?(\)){0,3})*( )?>>'
        self.compiled_pattern = re.compile(self.pattern)
        self.folder_path = 'src/gsm_symbolic/'
    
    def process_json_files(self) -> List[str]:
        """
        Process all JSON files in the folder path and return answers in expected format.
        
        Returns:
            List of processed symbolic expressions
        """
        answers = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.folder_path, filename), 'r') as f:
                    json_data = json.load(f)
                    
                    current_vars = sorted(json_data['variable_types']) 
                    new_vars = [chr(ord('a') + i) for i in range(len(current_vars))]
                    var_map = {current_vars[i]: new_vars[i] for i in range(len(current_vars))}
                    
                    # This conversion isn't used but kept for compatibility
                    new_variable_types = {new_var: json_data['variable_types'][old_var] 
                                         for old_var, new_var in var_map.items()}
                    
                    answer = json_data['answer_parsed']
                    for old_var, new_var in var_map.items():
                        answer = re.sub(r'\b' + old_var + r'\b', new_var, answer)
                    
                    answer = answer.replace('int', '') 
                    answer = f'<<{answer}>>'
                    answers.append(answer)
        
        return answers
    
    def test_gsm_symbolic_expressions(self):
        """Test that all GSM symbolic expressions match the expected pattern."""
        print("\n==== TESTING GSM SYMBOLIC EXPRESSIONS ====")
        
        # Process all JSON files and get answers
        answers = self.process_json_files()
        
        # Check that we have answers to test
        self.assertTrue(len(answers) > 0, "No JSON files found or processed")
        
        # Test each answer against the pattern
        match_count = 0
        for i, answer in enumerate(answers):
            with self.subTest(answer=answer, index=i):
                match = self.compiled_pattern.match(answer)
                self.assertIsNotNone(match, f"Expression does not match pattern: {answer}")
                print(f"Match: {answer}")
                match_count += 1
        
        print(f"Successfully matched {match_count} expressions to the pattern")
        return match_count


if __name__ == "__main__":
    unittest.main()


