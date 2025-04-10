from datasets import load_dataset, Dataset
import json
import os
import re



def load_dataset_by_name(task_name):
    if task_name == 'gsm_symbolic':
        data = []
        i = 0
        folder_path = '/home/tarun/struct_cot/src/gsm_symbolic/' 
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    json_data = json.load(f)
                    
                    current_vars = sorted(json_data['variable_types']) 
                    new_vars = [chr(ord('a') + i) for i in range(len(current_vars))]
                    var_map = {current_vars[i]: new_vars[i] for i in range(len(current_vars))}
                    new_variable_types = {new_var : json_data['variable_types'][old_var] for old_var, new_var in var_map.items()}
                    
                    question, answer = json_data['question_parsed'], json_data['answer_parsed']
                    for old_var, new_var in var_map.items():
                        question = question.replace("{" + old_var + "}", "{" + new_var + "}")
                        answer = re.sub(r'\b' + old_var + r'\b', new_var, answer)
                        answer = answer.replace('int', '')
                    data.append(dict(question= question, answer=answer, idx=i, variable_types = str(new_variable_types)))
                    i += 1
                    
        ds = Dataset.from_list(data)
        return ds
    else:
        raise ValueError(f"Dataset {task_name} not found")
    

