import os
import re
from pathlib import Path
import pandas as pd
import fire
from get_avgs import get_avgs

def extract_file_params(filename):
    """Extract parameters from the jsonl filename"""
    params = {}
    # Using regex to extract parameters from the filename
    step_match = re.search(r'step=(\d+)', filename)
    gen_length_match = re.search(r'gen_length=(\d+)', filename)
    block_length_match = re.search(r'block_length=(\d+)', filename)
    constrain_at_match = re.search(r'constrain_at=(\d+)', filename)
    
    if step_match:
        params['steps'] = int(step_match.group(1))
    if gen_length_match:
        params['gen_length'] = int(gen_length_match.group(1))
    if block_length_match:
        params['block_length'] = int(block_length_match.group(1))
    if constrain_at_match:
        params['constrain_at'] = int(constrain_at_match.group(1))
    
    return params

def analyze_task(task_name, output_file='task_results.txt'):
    """
    Analyze the logging results for a specific task across all models and configurations.
    
    Args:
        task_name: The name of the task to analyze
        output_file: The output file to write the results to
    """
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'logging' / task_name
    
    if not base_dir.exists():
        print(f"Task directory not found: {base_dir}")
        return
    
    results = []
    
    # Traverse the directory structure
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        for constraint_dir in model_dir.iterdir():
            if not constraint_dir.is_dir():
                continue
            
            constraint_type = constraint_dir.name
            
            for cot_dir in constraint_dir.iterdir():
                if not cot_dir.is_dir():
                    continue
                
                cot_value = cot_dir.name.split('=')[1]  # Extract True/False from cot=True
                
                for shot_dir in cot_dir.iterdir():
                    if not shot_dir.is_dir():
                        continue
                    
                    num_shots = shot_dir.name
                    
                    for result_file in shot_dir.glob('*.jsonl'):
                        # Extract parameters from filename
                        params = extract_file_params(result_file.name)
                        
                        # Get accuracy and time metrics
                        metrics = get_avgs(str(result_file))
                        
                        # Compile all information
                        result_entry = {
                            'model': model_name,
                            'constraint_type': constraint_type,
                            'cot': cot_value,
                            'num_shots': num_shots,
                            'accuracy': metrics['acc'],
                            'time': metrics['time'],
                            'parses': metrics['parses'],
                            **params
                        }
                        
                        results.append(result_entry)
    
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Sort by model, constraint_type, etc.
    if not df.empty:
        df = df.sort_values(['model', 'constraint_type', 'cot', 'num_shots', 'steps', 'gen_length', 'block_length', 'constrain_at'])
        
        # Create a table string
        table_str = df.to_string(index=False)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(f"Results for task: {task_name}\n\n")
            f.write(table_str)
        
        print(f"Results saved to {output_file}")
    else:
        print(f"No results found for task: {task_name}")

if __name__ == '__main__':
    fire.Fire(analyze_task) 