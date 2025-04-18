from .gsm_symbolic_validator import GSMSymbolicValidator 

def get_validator(task_name):
    if task_name == 'gsm_symbolic':
        return GSMSymbolicValidator(task_name)
    else:
        raise ValueError(f"No validator found for task {task_name}")
