from .gsm_evaluator import GSMSymbolicEvaluator 
from .schemabench_evaluator import SchemaBenchEvaluator 
from .jsonmode_evaluator import JSONModeEvaluator

def get_evaluator(task_name):
    if task_name == 'gsm_symbolic':
        return GSMSymbolicEvaluator
    elif task_name == 'schema_bench':
        return SchemaBenchEvaluator
    elif task_name == 'json_mode':
        return JSONModeEvaluator
    else:
        raise ValueError(f"Evaluator for task {task_name} not found")