from datasets import load_dataset, Dataset
import json
import os
import re
from outlines_core.json_schema import build_regex_from_schema
from jsonschema import validate, ValidationError


def load_dataset_by_name(task_name):
    if task_name == 'gsm_symbolic':
        # data = []
        # i = 0
        # current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # folder_path = f'{current_file_dir}/gsm_symbolic/' 
        # for filename in os.listdir(folder_path):
        #     if filename.endswith('.json'):
        #         with open(os.path.join(folder_path, filename), 'r') as f:
        #             json_data = json.load(f)
                    
        #             current_vars = sorted(json_data['variable_types']) 
        #             i, j = 0, 1
        #             var_map = {}
        #             new_variable_types = {}
        #             for var in current_vars: 
        #                 if json_data['variable_types'][var] == 'str':
        #                     var_map[var] = f's{j}'
        #                     j += 1
        #                 else:
        #                     var_map[var] = chr(ord('a') + i)
        #                     i += 1
                        
        #                 new_variable_types[var_map[var]] = json_data['variable_types'][var]
        #             question, answer = json_data['question_parsed'], json_data['answer_parsed']
        #             for old_var, new_var in var_map.items():
        #                 question = question.replace("{" + old_var + "}", "{" + new_var + "}")
        #                 answer = re.sub(r'\b' + old_var + r'\b', new_var, answer)
        #                 answer = answer.replace('int', '')
        #             data.append(dict(question= question, answer=answer, idx=i, variable_types = str(new_variable_types)))
        #             i += 1
                    
        # ds = Dataset.from_list(data)
        ds = load_dataset('tarsur909/gsms', split = 'train')
        return ds, None
    elif task_name == 'json_mode':
        data = []
        i = 0
        for row in load_dataset('NousResearch/json-mode-eval', split = 'train'):
            schema = row['schema']
            try:
                pattern = build_regex_from_schema(schema)
                
            except Exception as e:
                pass

            row['prompt'][-1]['content'] = row['prompt'][-1]['content'] + '\nOnly output the JSON object, no other text or comments.\n'
            data.append({**row, 'idx': i})
            i += 1
        ds = Dataset.from_list(data)
        return ds, 'schema'
    elif task_name == 'schema_bench':
        data = []
        i = 0
        for row in load_dataset("epfl-dlab/JSONSchemaBench", "Github_trivial", split = 'train'):
            schema = row['json_schema']
            if i == 24:
                i += 1
                continue
            
            try:
                pattern = build_regex_from_schema(schema)
            except Exception as e:
                i += 1
                continue

            data.append({'question': row['json_schema'], 'json_schema': row['json_schema'], 'idx': i})
            i += 1
        ds = Dataset.from_list(data)
        return ds, 'json_schema'
        
    else:
        raise ValueError(f"Dataset {task_name} not found")
    

