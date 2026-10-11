from datasets import load_dataset


def load_dataset_by_name(task_name):
    if task_name == "gsm_symbolic":
        ds = load_dataset("anon2525/gsms", split="train")
        return ds, None
    elif task_name == "json_mode":
        ds = load_dataset("anon2525/json-mode-eval-rgx", split="train")
        return ds, "pattern"

    else:
        raise ValueError(f"Dataset {task_name} not found")
