import os
import re
from pathlib import Path

import fire
import pandas as pd
import srsly

from get_avgs import get_avgs
from result_utils import align_results, better_result


def extract_file_params(filename):
    """Extract parameters from the jsonl filename"""
    params = {}
    # Using regex to extract parameters from the filename
    step_match = re.search(r"step=(\d+)", filename)
    gen_length_match = re.search(r"gen_length=(\d+)", filename)
    block_length_match = re.search(r"block_length=(\d+)", filename)
    constrain_at_match = re.search(r"constrain_at=(\d+)", filename)

    if step_match:
        params["steps"] = int(step_match.group(1))
    if gen_length_match:
        params["gen_length"] = int(gen_length_match.group(1))
    if block_length_match:
        params["block_length"] = int(block_length_match.group(1))
    if constrain_at_match:
        params["constrain_at"] = int(constrain_at_match.group(1))

    return params


def compute_metrics_from_results(results):
    """Compute accuracy, time, and parses metrics from a list of problem results"""
    times = []
    corrects = []
    parses = []

    for r in results:
        times.append(r["time"])
        corrects.append(r["correct"])
        parses.append(r["parses"])

    return {
        "accuracy": round(100 * sum(corrects) / len(corrects), 2),
        "time": round(sum(times) / len(times), 2),
        "parses": round(100 * sum(parses) / len(parses), 2),
    }


def analyze_task(task_name, output_file="task_results.txt"):
    """
    Analyze the logging results for a specific task across all models and configurations.

    Args:
        task_name: The name of the task to analyze
        output_file: The output file to write the results to
    """
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logging" / task_name

    if not base_dir.exists():
        print(f"Task directory not found: {base_dir}")
        return

    results = []
    # Store raw results for problem-wise comparison
    raw_results_map = {}

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

                cot_value = cot_dir.name.split("=")[1]  # Extract True/False from cot=True

                for shot_dir in cot_dir.iterdir():
                    if not shot_dir.is_dir():
                        continue

                    num_shots = shot_dir.name

                    for result_file in shot_dir.glob("*.jsonl"):
                        # Extract parameters from filename
                        params = extract_file_params(result_file.name)
                        file_path = str(result_file)

                        # Get accuracy and time metrics
                        metrics = get_avgs(file_path)

                        # Store raw results for problem-wise comparison if needed
                        config_key = (
                            model_name,
                            cot_value,
                            num_shots,
                            params.get("steps", None),
                            params.get("gen_length", None),
                            params.get("block_length", None),
                            params.get("constrain_at", None),
                            result_file.stem,
                        )

                        if constraint_type in ["ar_constrained", "unconstrained"]:
                            if config_key not in raw_results_map:
                                raw_results_map[config_key] = {}
                            raw_results_map[config_key][constraint_type] = file_path

                        # Compile all information
                        result_entry = {
                            "model": model_name,
                            "constraint_type": constraint_type,
                            "cot": cot_value,
                            "num_shots": num_shots,
                            "accuracy": metrics["acc"],
                            "time": metrics["time"],
                            "parses": metrics["parses"],
                            **params,
                        }

                        results.append(result_entry)

    # Now generate b_ar_unconstrained entries with problem-wise comparison
    for config_key, constraint_files in raw_results_map.items():
        if "ar_constrained" in constraint_files and "unconstrained" in constraint_files:
            model, cot, num_shots, steps, gen_length, block_length, constrain_at, _ = config_key

            # Load the raw results for both constraint types
            ar_results = list(srsly.read_jsonl(constraint_files["ar_constrained"]))
            unconstrained_results = list(srsly.read_jsonl(constraint_files["unconstrained"]))

            # Make sure both result lists have the same problems in the same order
            try:
                aligned = align_results(ar_results, unconstrained_results)
            except ValueError as error:
                print(f"Skipping incompatible runs for {model}: {error}")
                continue

            # Create a new combined result list taking the better result for each problem
            combined_results = []
            for ar_res, unc_res in aligned:
                # Choose the better result for this problem
                # Prefer correct over parsing, and if both are equal, prefer faster
                better_res = better_result(ar_res, unc_res)

                combined_results.append(better_res)

            # Compute metrics for the combined results
            metrics = compute_metrics_from_results(combined_results)

            # Create the b_ar_unconstrained entry
            result_entry = {
                "model": model,
                "constraint_type": "b_ar_unconstrained",
                "cot": cot,
                "num_shots": num_shots,
                "accuracy": metrics["accuracy"],
                "time": metrics["time"],
                "parses": metrics["parses"],
            }

            # Add the parameters if they exist
            if steps is not None:
                result_entry["steps"] = steps
            if gen_length is not None:
                result_entry["gen_length"] = gen_length
            if block_length is not None:
                result_entry["block_length"] = block_length
            if constrain_at is not None:
                result_entry["constrain_at"] = constrain_at

            results.append(result_entry)

    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Sort by model, constraint_type, etc.
    if not df.empty:
        df = df.sort_values(
            [
                "model",
                "constraint_type",
                "cot",
                "num_shots",
                "steps",
                "gen_length",
                "block_length",
                "constrain_at",
            ]
        )

        # Create a table string
        table_str = df.to_string(index=False)

        # Write to output file
        with open(output_file, "w") as f:
            f.write(f"Results for task: {task_name}\n\n")
            f.write(table_str)

        print(f"Results saved to {output_file}")
    else:
        print(f"No results found for task: {task_name}")


if __name__ == "__main__":
    fire.Fire(analyze_task)
