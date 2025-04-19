from abc import ABC, abstractmethod 
import re
from syncode.larkm import Lark

class BaseValidator(ABC):
    def __init__(self, task_name):
        self.task_name = task_name
        with open(f'grammar_folder/{task_name}.lark', 'r') as f:
            grammar_str = f.read()
        self.parser = Lark(grammar_str, start='start')
    
    @abstractmethod
    def validate(self, code):
        pass
    
    def _parse(self, code):
        return self.parser.parse(code)