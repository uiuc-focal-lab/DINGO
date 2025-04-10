from .gsm_evaluator import GSMSymbolicEvaluator 


def get_evaluator(task_name):
    if task_name == 'gsm_symbolic':
        return GSMSymbolicEvaluator
    else:
        raise ValueError(f"Evaluator for task {task_name} not found")