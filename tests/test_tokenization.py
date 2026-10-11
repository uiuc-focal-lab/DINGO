import gc
import weakref
from types import SimpleNamespace

import pytest
from helpers import TinyTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from tokenization import get_mask_token_id, vocabulary_bytes


def test_byte_level_vocab_keeps_partial_utf8_and_added_token_spelling():
    mapping = bytes_to_unicode()
    tok = SimpleNamespace(
        byte_decoder={v: k for k, v in mapping.items()},
        get_vocab=lambda: {mapping[0xC3]: 0, mapping[0xA9]: 1, "Ġa": 2, "é": 3},
        get_added_vocab=lambda: {"é": 3},
    )
    assert vocabulary_bytes(tok) == {0: b"\xc3", 1: b"\xa9", 2: b" a", 3: b"\xc3\xa9"}


def test_llada_mask_fallback_when_tokenizer_property_is_none():
    tok = SimpleNamespace(mask_token_id=None, name_or_path="GSAI-ML/LLaDA-8B-Instruct")
    assert get_mask_token_id(tok) == 126336


class CountingTokenizer(TinyTokenizer):
    def __init__(self):
        super().__init__()
        self.vocab_reads = 0
        self.added_reads = 0

    def get_vocab(self):
        self.vocab_reads += 1
        return super().get_vocab()

    def get_added_vocab(self):
        self.added_reads += 1
        return super().get_added_vocab()


def test_vocabulary_extraction_reused_without_exposing_mutable_cache():
    tok = CountingTokenizer()
    first = vocabulary_bytes(tok)
    first[0] = b"wrong"
    assert vocabulary_bytes(tok)[0] == b"a"
    assert tok.vocab_reads == tok.added_reads == 1


def test_preprocessing_cache_does_not_keep_tokenizer_alive():
    tok = CountingTokenizer()
    reference = weakref.ref(tok)
    vocabulary_bytes(tok)
    del tok
    gc.collect()
    assert reference() is None


def test_schema_builds_reuse_tokenizer_preprocessing(store_factory):
    tok = CountingTokenizer()
    assert store_factory("a", tok).check_is_reachable([0], is_final_block=True)
    assert store_factory("b", tok).check_is_reachable([1], is_final_block=True)
    assert tok.vocab_reads == tok.added_reads == 1


def test_explicit_invalidation_rebuilds_mutated_vocabulary(store_factory):
    import tokenization

    tok = CountingTokenizer()
    before = store_factory("a", tok)
    tok.vocab = {"a": 1, "b": 0, "<eos>": 2, "<mask>": 3}
    tokenization.invalidate_tokenizer_cache(tok)
    after = store_factory("a", tok)
    assert before.fpath != after.fpath
    assert after.check_is_reachable([1], is_final_block=True)
    assert not after.check_is_reachable([0], is_final_block=True)
    assert tok.vocab_reads == tok.added_reads == 2


def test_special_token_invalidation_cannot_reuse_old_language(store_factory):
    from tokenization import invalidate_tokenizer_cache

    tok = TinyTokenizer()
    store_factory("a", tok)
    tok.all_special_ids = [0, 2, 3]
    invalidate_tokenizer_cache(tok)
    with pytest.raises(ValueError, match="tokenizable completion"):
        store_factory("a", tok)


def test_eos_invalidation_removes_old_padding_completions(store_factory):
    from tokenization import invalidate_tokenizer_cache

    tok = TinyTokenizer()
    before = store_factory("a", tok)
    assert before.completion_targets(2)[before.initial_state].item() is True
    tok.eos_token_id = None
    invalidate_tokenizer_cache(tok)
    after = store_factory("a", tok)
    assert after.completion_targets(2)[after.initial_state].item() is False
