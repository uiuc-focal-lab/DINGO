import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from helpers import TinyTokenizer

import dfa as dfa_module
from dfa import DFAMatrixStore


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(dfa_module, "__file__", str(tmp_path / "dfa.py"))

    def build(pattern="ab", tokenizer=None, opportunistic=False):
        return DFAMatrixStore("test", enable_oppurtunistic=opportunistic).build(
            pattern, 0, tokenizer or TinyTokenizer()
        )

    return build
