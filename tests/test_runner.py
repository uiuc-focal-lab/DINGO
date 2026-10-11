import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset


def test_runner_import_does_not_require_hf_cache():
    env = {k: v for k, v in os.environ.items() if k != "HF_CACHE"}
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", "import main"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def load_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_CACHE", str(tmp_path))
    return importlib.import_module("main")


def test_false_cli_flags_are_false(monkeypatch, tmp_path):
    main = load_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "json_mode",
            "--model",
            "fake",
            "--write_file",
            "False",
            "--enable_dist",
            "False",
        ],
    )
    args = main.parse_arguments()
    assert not args.write_file and not args.enable_dist


def test_resume_uses_dataset_identity_and_retains_results(monkeypatch, tmp_path, capsys):
    main = load_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "json_mode",
            "--model",
            "fake",
            "--write_file",
            "True",
            "--log_dir",
            str(tmp_path),
            "--steps",
            "1",
            "--gen_length",
            "1",
            "--block_length",
            "1",
        ],
    )
    args = main.parse_arguments()
    dataset = Dataset.from_list(
        [{"idx": 42, "prompt": "test", "schema": '{"type":"object"}', "completion": "{}"}]
    )
    monkeypatch.setattr(main, "load_dataset_by_name", lambda name: (dataset, None))
    calls = []

    class FakeLM:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(device="cpu")

        def __call__(self, batch, store):
            calls.append(batch["idx"])
            return {**batch, "llm_response": "{}", "response_info": {"time": 0.0}}

    monkeypatch.setattr(main, "BaseLM", FakeLM)
    main.process_dataset_interactive(args)
    main.process_dataset_interactive(args)
    assert calls == [42]
    assert capsys.readouterr().out.strip().endswith("1.0")


def test_parallel_orchestration_forwards_settings_and_builds_schema_stores(monkeypatch, tmp_path):
    main = load_runner(monkeypatch, tmp_path)
    args = main.parse_arguments(
        [
            "--dataset",
            "json_mode",
            "--model",
            "fake",
            "--steps",
            "6",
            "--gen_length",
            "2",
            "--block_length",
            "1",
            "--constraint_mode",
            "diffusion_constrained",
            "--num_workers_per_gpu",
            "2",
            "--write_file",
            "True",
            "--log_dir",
            str(tmp_path),
        ]
    )
    rows = [
        {"idx": i, "prompt": "{}", "schema": "{}", "pattern": "a", "completion": "{}"}
        for i in [42, 7]
    ]
    monkeypatch.setattr(
        main, "load_dataset_by_name", lambda _: (Dataset.from_list(rows), "pattern")
    )
    calls, configs, killed = [], [], []

    class FakeLM:
        def __init__(self, *a, **kw):
            configs.append(kw)
            self.tokenizer = object()

        def __call__(self, batch, store):
            calls.append((batch["idx"], store))
            return {**batch, "llm_response": "{}", "response_info": {"time": 0.0}}

    class Store:
        def __init__(self, *a):
            pass

        def build(self, pattern, idx, tokenizer):
            return (pattern, idx)

    class Handle:
        def __init__(self, instance):
            self.instance = instance

        def __getattr__(self, name):
            return SimpleNamespace(remote=getattr(self.instance, name))

    def remote(**resources):
        def wrap(target):
            if isinstance(target, type):
                return SimpleNamespace(remote=lambda: Handle(target()))
            return SimpleNamespace(remote=target)

        return wrap

    ray = SimpleNamespace(
        remote=remote,
        is_initialized=lambda: True,
        wait=lambda refs, **kw: ([refs[-1]], refs[:-1]),
        get=lambda ref: ref,
        kill=lambda actor: killed.append(actor),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(main, "BaseLM", FakeLM)
    import dfa

    monkeypatch.setattr(dfa, "DFAMatrixStore", Store)
    main.process_dataset_parallel(args)
    assert calls == [(42, ("a", 42)), (7, ("a", 7))]
    assert len(configs) == len(killed) == 2
    assert all(c["steps"] == 6 and c["require_complete"] for c in configs)
    import json

    saved = [
        json.loads(line) for line in Path(main.create_result_file(args)).read_text().splitlines()
    ]
    assert [r["idx"] for r in saved] == [7, 42]
