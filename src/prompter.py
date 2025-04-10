from jinja2 import Environment, FileSystemLoader
import yaml
import json 
from abc import ABC, abstractmethod
from textwrap import dedent
import os
from copy import deepcopy

class BasePrompter(ABC):
    def __init__(self, dataset, num_shots=8) -> None:
        self.num_shots = num_shots
        with open(f'prompt_templates/{dataset}.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.fewshots = self.config.get('fewshots', [])
        self.instruction_str = self.config.get('instruction', "")
        
      
    def prompt(self, row, modify_system_prompt = True, chat_mode = True):
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
            prompt = '\n'.join(messages + [self.instruction_str + '\n' + example['question'] + '\n' + example['response'] for example in self.fewshots[:self.num_shots]] + [self.instruction_str + '\n' + row['question']])
            return prompt