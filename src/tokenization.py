"""Token bytes for the byte-level BPE tokenizers used by Dream and LLaDA."""

import hashlib
import json
import weakref
from dataclasses import dataclass

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


@dataclass
class _TokenizerPreprocessing:
    vocab_bytes: dict
    eot_id: int | None
    fingerprint: str | None = None


# Identity keys also support unhashable tokenizers. Entries never own a tokenizer.
_PREPROCESSING_CACHE = {}
_CACHE_ATTRIBUTE = "_diffusion_constrain_tokenizer_preprocessing"


def invalidate_tokenizer_cache(tokenizer):
    """Call after changing vocabulary, byte decoder, or special-token metadata.

    Tokenizers are treated as immutable between invalidations so schema builds
    can reuse byte extraction and fingerprinting without scanning the vocabulary.
    """
    _PREPROCESSING_CACHE.pop(id(tokenizer), None)
    if hasattr(tokenizer, _CACHE_ATTRIBUTE):
        delattr(tokenizer, _CACHE_ATTRIBUTE)


def _cached_preprocessing(tokenizer):
    identity = id(tokenizer)
    entry = _PREPROCESSING_CACHE.get(identity)
    if entry is not None and entry[0]() is tokenizer:
        return entry[1]
    owned = getattr(tokenizer, _CACHE_ATTRIBUTE, None)
    if owned is not None:
        return owned
    vocab = tokenizer.get_vocab()
    cached = _TokenizerPreprocessing(
        _extract_vocabulary_bytes(tokenizer, vocab), vocab.get("<|eot_id|>")
    )
    try:
        reference = weakref.ref(tokenizer, lambda _: _PREPROCESSING_CACHE.pop(identity, None))
    except TypeError:
        # Non-weak-referenceable objects can own their cache without a global
        # strong reference. Slot-only objects without this ability remain uncached.
        try:
            setattr(tokenizer, _CACHE_ATTRIBUTE, cached)
        except (AttributeError, TypeError):
            pass
    else:
        _PREPROCESSING_CACHE[identity] = (reference, cached)
    return cached


def tokenizer_preprocessing(tokenizer):
    """Return the shared, read-only-by-convention byte map and content fingerprint."""
    cached = _cached_preprocessing(tokenizer)
    if cached.fingerprint is None:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [
                    tokenizer.name_or_path,
                    sorted(tokenizer.all_special_ids),
                    tokenizer.eos_token_id,
                    get_mask_token_id(tokenizer),
                ]
            ).encode()
        )
        for token_id, token_bytes in sorted(cached.vocab_bytes.items()):
            digest.update(str(token_id).encode() + b":" + token_bytes + b"\0")
        cached.fingerprint = digest.hexdigest()
    return cached


def get_mask_token_id(tokenizer):
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None and "llada" in tokenizer.name_or_path.lower():
        mask_id = 126336
    if mask_id is None:
        raise ValueError("Tokenizer must provide a mask token ID")
    return mask_id


def vocabulary_bytes(tokenizer):
    """Preserve every ID and byte, including incomplete UTF-8 BPE fragments.

    Slow Dream tokenizers expose byte_decoder; fast LLaDA tokenizers expose
    a ByteLevel decoder. Added tokens are literal strings, not BPE byte codes.
    Unlike a bytes->ID dictionary, this map cannot collapse duplicate spellings.
    """
    return _cached_preprocessing(tokenizer).vocab_bytes.copy()


def _extract_vocabulary_bytes(tokenizer, vocab):
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if byte_decoder is None and backend is not None:
        decoder = json.loads(backend.decoder.__getstate__()) if backend.decoder else {}
        if decoder.get("type") != "ByteLevel":
            raise ValueError("Only byte-level BPE tokenizers are supported")
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    added = tokenizer.get_added_vocab()
    result = {}
    for token, token_id in vocab.items():
        if isinstance(token, bytes):
            result[token_id] = token
        elif byte_decoder is None or token in added:
            result[token_id] = token.encode("utf-8")
        else:
            result[token_id] = bytes(byte_decoder[c] for c in token)
    return result
