from jinja2 import Environment, FileSystemLoader
import yaml
import json 
from abc import ABC, abstractmethod
from textwrap import dedent
import os
from copy import deepcopy

class BasePrompter(ABC):
    def __init__(self, dataset, num_shots=8, do_cot = False) -> None:
        self.dataset = dataset
        if self.dataset == 'json_mode':
            return
        self.num_shots = num_shots
        #get current file directory     
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f'{current_file_dir}/prompt_templates/{dataset}.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        if 'fewshots' in self.config:
            self.fewshots = self.config['fewshots']
            if do_cot:
                self.fewshots = self.fewshots.get('cot', []) 
            else:
                self.fewshots = self.fewshots.get('std', []) 
        else:
            self.fewshots = []

        if 'instruction' in self.config: 
            self.instruction_str = self.config['instruction'] 
            if do_cot:
                self.instruction_str = self.instruction_str.get('cot', "")
            else:
                self.instruction_str = self.instruction_str.get('std', "")
        else:
            self.instruction_str = ""
        
      
    def prompt(self, row, modify_system_prompt = True, chat_mode = True):
        if self.dataset == 'json_mode': 
            return row['prompt']
        messages = []
        if chat_mode:
            for example in self.fewshots[:self.num_shots]:
                messages.append(
                    {
                        "role": "user",
                        "content": self.instruction_str + '\n' + example['question']
                    }
                )
                messages.append(
                {
                    "role": "assistant",
                    "content": example['response']        
                })
            
            messages.append(
            {
                "role": "user",
                "content": self.instruction_str + '\n' + row['question']
            })
            
            messages.append(
            {
                "role": "assistant",
                "content": ""
            }) 
            return messages
        
        else:
            prompt = '\n'.join([example['question'] + '\n' + example['response'] for example in self.fewshots[:self.num_shots]] + [row['question']])
            return prompt