class TinyTokenizer:
    name_or_path = "tiny-dream"
    mask_token_id = 3
    eos_token_id = 2
    unk_token_id = 0
    all_special_ids = [2, 3]

    def __init__(self, unknown_eot=False, vocab=None):
        self.vocab = vocab or {"a": 0, "b": 1, "<eos>": 2, "<mask>": 3}
        self.unknown_eot = unknown_eot

    def get_vocab(self):
        return self.vocab.copy()

    def get_added_vocab(self):
        return {}

    def __len__(self):
        return max(self.vocab.values()) + 1

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id if self.unknown_eot else None)

    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        if isinstance(tokens, int):
            tokens = [tokens]
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(
            inv[t] for t in tokens if not (skip_special_tokens and t in self.all_special_ids)
        )
