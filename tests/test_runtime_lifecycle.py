from types import SimpleNamespace

import pytest
import torch

from models.base_model import BaseLM


def bare_lm():
    lm = BaseLM.__new__(BaseLM)
    lm.model = SimpleNamespace(device=torch.device("cpu"))
    lm.schema_key = "pattern"
    lm.generator = lambda prompt, store: "ab"
    return lm


def test_model_retains_only_one_schema_runtime_cache(store_factory):
    first, second = store_factory("ab"), store_factory("a")
    first._runtime_decoders = {("cpu", 2, 4, False): object()}
    first.completion_targets(0)
    lm = bare_lm()
    batch = {"prompt": "test", "pattern": "ab"}
    lm(batch, first)
    # Repeated use of this schema should amortize graph capture.
    lm(batch, first)
    assert first._runtime_decoders
    lm(batch, second)
    assert not getattr(first, "_runtime_decoders", {})
    assert not getattr(first, "_runtime_completion", {})


def test_model_exception_clears_pending_decode_validation(store_factory):
    store = store_factory("ab")
    lm = bare_lm()

    def interrupted(prompt, store):
        store._runtime_validity = {"cpu": torch.tensor(False)}
        raise RuntimeError("interrupted")

    lm.generator = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        lm({"prompt": "test", "pattern": "ab"}, store)
    assert not getattr(store, "_runtime_validity", {})
