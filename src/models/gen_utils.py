import torch


def find_target_after_phrase(token_ids, tokenizer, phrase="The final answer", target_word="."):
    """
    Find the token index corresponding to the first period after the phrase "The final answer".

    Args:
        token_ids: A 1D sequence of token IDs (list or tensor)
        tokenizer: The tokenizer used to decode the tokens

    Returns:
        The index of the first period token after "The final answer" phrase,
        or -1 if no such token is found.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    full_text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    start = full_text.lower().find(phrase.lower())
    if start < 0 or full_text.find(target_word, start + len(phrase)) < 0:
        return -1

    # Iterate through the token sequence
    for i in range(len(token_ids)):
        # Decode all tokens up to current position to get the current text
        current_text = tokenizer.decode(
            token_ids[: i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        start = current_text.lower().find(phrase.lower())
        if start >= 0 and current_text.find(target_word, start + len(phrase)) >= 0:
            return i

    # If no period found after "The final answer"
    return -1


def reveal_schedule(block_length, steps):
    """Fixed reveal counts for a block that starts fully masked."""
    base, remainder = divmod(block_length, steps)
    return [base + (i < remainder) for i in range(steps)]


def validate_generation_settings(
    steps, gen_length, block_length, constrain_at, constraint_mode, batch_size=1
):
    if constraint_mode not in ("unconstrained", "diffusion_constrained", "ar_constrained"):
        raise ValueError(f"Invalid constraint mode: {constraint_mode}")
    if steps <= 0 or gen_length <= 0 or block_length <= 0:
        raise ValueError("steps, gen_length and block_length must be positive")
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("gen_length must be divisible by block_length and steps by num_blocks")
    constrained = constraint_mode in ("diffusion_constrained", "ar_constrained")
    steps_per_block = steps // (gen_length // block_length)
    if constrained and (constrain_at is None or not 0 <= constrain_at < steps_per_block):
        raise ValueError("constrain_at must be within the diffusion steps of each block")
    if constrained and batch_size != 1:
        raise ValueError("Constrained generation currently supports batch_size=1")
    return steps_per_block


def completed_stop_prefix(
    x, prompt_length, end_gen_idx, tokenizer, mask_id, phrase, target_word, dfa_store=None
):
    """Stop only at a fully revealed, accepted prefix using original token IDs."""
    if phrase is None or target_word is None:
        return None
    generated = x[0, prompt_length:end_gen_idx]
    if (generated == mask_id).any():
        return None
    index = find_target_after_phrase(generated, tokenizer, phrase, target_word)
    if index < 0:
        return None
    prefix = generated[: index + 1]
    if dfa_store is not None and not dfa_store.check_is_reachable(
        prefix.tolist(), is_final_block=True
    ):
        return None
    return x[:, : prompt_length + index + 1]


def find_target_after_phrase_str(strng, phrase="The final answer", target_word=">>"):
    try:
        start_idx = strng.find(phrase)
        if start_idx == -1:
            return -1
        end_idx = strng.find(target_word, start_idx)
        if end_idx == -1:
            return -1
        return strng[: end_idx + len(target_word)]
    except (AttributeError, TypeError):
        return -1
