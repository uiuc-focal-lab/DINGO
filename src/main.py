import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import load_dataset_by_name
from dfa import build_for_dataset
from evaluators import get_evaluator
from models import BaseLM
from models.gen_utils import validate_generation_settings
from prompter import BasePrompter

if "HF_CACHE" in os.environ:
    os.environ.setdefault("HF_HOME", os.environ["HF_CACHE"])


# Setting up argument parsing
def parse_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Test LLMs on different datasets with various prompt styles."
    )

    # dataset args
    parser.add_argument(
        "--dataset", type=str, required=True, help='Dataset name, e.g., "gsm8k" or "spider"'
    )
    parser.add_argument("--start_idx", type=int, default=0, help="start index")
    parser.add_argument("--end_idx", type=int, default=-1, help="end index")
    parser.add_argument("--debug_ids", type=str, default=None, help="debug ids")

    # results args
    parser.add_argument("--log_dir", type=str, default="logging", help="Directory to save logs")
    parser.add_argument(
        "--overwrite_results", type=parse_bool, default=False, help="overwrite results file"
    )
    parser.add_argument("--write_file", type=parse_bool, default=False, help="save results in file")

    # CoT args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model name, e.g., "gpt-3.5-turbo" or "gemini-1.0-pro"',
    )
    parser.add_argument(
        "--constraint_mode",
        type=str,
        default="unconstrained",
        help="Generation mode",
        choices=["unconstrained", "diffusion_constrained", "ar_constrained"],
    )
    parser.add_argument("--num_shots", type=int, default=8, help="number of few shots")
    parser.add_argument("--do_cot", type=parse_bool, default=False, help="use COT")

    # general generation args
    parser.add_argument("--steps", type=int, default=128, help="number of steps for diffusion")
    parser.add_argument(
        "--gen_length", type=int, default=128, help="max number of new tokens to generate"
    )
    parser.add_argument("--block_length", type=int, default=128, help="block length for diffusion")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="cfg scale for diffusion")
    parser.add_argument(
        "--remasking", type=str, default="low_confidence", help="remasking strategy"
    )
    parser.add_argument("--constrain_at", type=int, default=0, help="constrain at step")
    parser.add_argument(
        "--enable_oppurtunistic",
        type=parse_bool,
        default=False,
        help="enable oppurtunistic constrained decoding",
    )

    # Distributed inference args
    parser.add_argument(
        "--enable_dist", type=parse_bool, default=False, help="enable distributed inference"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument(
        "--num_workers_per_gpu", type=int, default=1, help="number of workers per gpu"
    )

    parser.add_argument(
        "--require_complete",
        type=parse_bool,
        default=True,
        help="Require acceptance within the token budget; False uses the paper valid-prefix objective",
    )
    return parser.parse_args(argv)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def create_result_file(args):

    file_id = f"step={args.steps}_gen_length={args.gen_length}_block_length={args.block_length}_temperature={args.temperature}_cfg_scale={args.cfg_scale}_remasking={args.remasking}_constrain_at={args.constrain_at}_oppurtunistic={args.enable_oppurtunistic}_complete={args.require_complete}"

    result_file = f"{args.log_dir}/{args.dataset}/{args.model.split('/')[-1]}/{args.constraint_mode}/cot={args.do_cot}/{args.num_shots}-shot/{file_id}.jsonl"
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    return result_file


def prepare_run(args):
    if args.constrain_at is None:
        args.constrain_at = 0
    validate_generation_settings(
        args.steps, args.gen_length, args.block_length, args.constrain_at, args.constraint_mode
    )
    if args.num_gpus <= 0 or args.num_workers_per_gpu <= 0:
        raise ValueError("GPU and worker counts must be positive")
    dataset, schema_key = load_dataset_by_name(args.dataset)
    if "idx" not in dataset.column_names:
        dataset = dataset.add_column("idx", list(range(len(dataset))))
    if args.debug_ids is not None:
        dataset = dataset.select([int(i) for i in args.debug_ids.split(",")])
    else:
        end = len(dataset) if args.end_idx == -1 else args.end_idx
        dataset = dataset.select(range(args.start_idx, end))
    result_file = create_result_file(args)
    if args.overwrite_results and os.path.exists(result_file):
        os.remove(result_file)
    previous = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                result = json.loads(line)
                if "idx" not in result:
                    raise ValueError("Legacy result file has no idx; use --overwrite_results True")
                previous[result["idx"]] = result
    rows = list(dataset)
    selected_ids = {r["idx"] for r in rows}
    previous = {k: v for k, v in previous.items() if k in selected_ids}
    pending = [r for r in rows if r["idx"] not in previous]
    return pending, schema_key, result_file, list(previous.values())


def model_arguments(args, schema_key):
    return dict(
        constraint_mode=args.constraint_mode,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        constrain_at=args.constrain_at,
        enable_oppurtunistic=args.enable_oppurtunistic,
        schema_key=schema_key,
        require_complete=args.require_complete,
    )


def save_result(args, result_file, result):
    if args.write_file:
        with open(result_file, "a") as f:
            f.write(json.dumps(result, default=set_default) + "\n")


def process_dataset_interactive(args):
    rows, schema_key, result_file, results = prepare_run(args)
    if rows:
        llm = BaseLM(
            args.model,
            args.dataset,
            "cuda",
            args.do_cot,
            device_map="auto",
            **model_arguments(args, schema_key),
        )
        if args.constraint_mode in ("diffusion_constrained", "ar_constrained"):
            stores = build_for_dataset(
                rows,
                schema_key,
                args.dataset,
                args.do_cot,
                llm.tokenizer,
                llm.model.device,
                args.enable_oppurtunistic,
                True,
            )
        else:
            stores = [None] * len(rows)
        prompter = BasePrompter(args.dataset, args.num_shots, args.do_cot)
        evaluator = get_evaluator(args.dataset)(args.dataset, do_cot=args.do_cot)
        chat_mode = any(tag in args.model.lower() for tag in ("instruct", "chat"))
        for row, store in tqdm(zip(rows, stores, strict=True), total=len(rows)):
            batch = {**row, "prompt": prompter.prompt(row, chat_mode=chat_mode)}
            result = evaluator.evaluate_answer(llm(batch, store))
            result["idx"] = row["idx"]
            save_result(args, result_file, result)
            results.append(result)
    print(float(np.mean([r["correct"] for r in results])) if results else 0.0)


def process_dataset_parallel(args):
    import ray

    from dfa import DFAMatrixStore

    rows, schema_key, result_file, results = prepare_run(args)
    if not rows:
        print(float(np.mean([r["correct"] for r in results])) if results else 0.0)
        return
    if not ray.is_initialized():
        ray.init()
    chat_mode = any(tag in args.model.lower() for tag in ("instruct", "chat"))

    @ray.remote(num_gpus=1 / args.num_workers_per_gpu)
    class PromptActor:
        def __init__(self):
            self.prompter = BasePrompter(args.dataset, args.num_shots, args.do_cot)
            self.llm = BaseLM(
                args.model, args.dataset, "cuda", args.do_cot, **model_arguments(args, schema_key)
            )
            self.shared_store = None
            if schema_key is None and args.constraint_mode != "unconstrained":
                self.shared_store = DFAMatrixStore(
                    args.dataset, args.do_cot, args.enable_oppurtunistic
                ).build(None, None, self.llm.tokenizer)

        def process_item(self, row):
            store = self.shared_store
            if schema_key is not None and args.constraint_mode != "unconstrained":
                store = DFAMatrixStore(args.dataset, args.do_cot, args.enable_oppurtunistic).build(
                    row[schema_key], row["idx"], self.llm.tokenizer
                )
            batch = {**row, "prompt": self.prompter.prompt(row, chat_mode=chat_mode)}
            return self.llm(batch, store)

    @ray.remote(num_cpus=1)
    def evaluate(batch):
        result = get_evaluator(args.dataset)(args.dataset, do_cot=args.do_cot).evaluate_answer(
            batch
        )
        result["idx"] = batch["idx"]
        return result

    actors = [PromptActor.remote() for _ in range(args.num_gpus * args.num_workers_per_gpu)]
    try:
        futures = [
            evaluate.remote(actors[i % len(actors)].process_item.remote(row))
            for i, row in enumerate(rows)
        ]
        # Save each completed result immediately so an interrupted run resumes
        # by the actual dataset identity, including non-contiguous selections.
        with tqdm(total=len(futures)) as progress:
            while futures:
                ready, futures = ray.wait(futures, num_returns=1)
                result = ray.get(ready[0])
                save_result(args, result_file, result)
                results.append(result)
                progress.update(1)
    finally:
        for actor in actors:
            ray.kill(actor)
    print(float(np.mean([r["correct"] for r in results])) if results else 0.0)


def main():
    args = parse_arguments()
    if args.enable_dist:
        process_dataset_parallel(args)
    else:
        process_dataset_interactive(args)


if __name__ == "__main__":
    main()
