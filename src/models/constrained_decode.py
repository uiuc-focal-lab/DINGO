"""Shared per-step DINGO objective and terminal-state selection."""

import torch

from .viterbi import ViterbiGraph, constrained_path


def masked_log_scores(logits, masked, mask_token_id):
    """Current transformer scores with Appendix A's point-mass masking.

    Raw float32 logits suffice: each position's log-softmax normalizer is
    constant across every path, so omitting it leaves the argmax unchanged.
    """
    scores = logits.float().clone()
    scores.masked_fill_(masked[:, None], -torch.inf)
    scores[:, mask_token_id] = torch.where(masked, 0.0, -torch.inf)
    return scores


def decode_block(
    store,
    logits,
    masked,
    initial_state,
    remaining_tokens,
    require_complete=True,
    greedy=False,
    defer_validation=False,
):
    """Return (MAP token path, terminal state), or raise if no supported path.

    With require_complete=False, targets are exactly Q_live from Algorithm 1.
    With require_complete=True, additionally reserve enough real token steps
    to finish within the response budget. EOS/EOT self-loops permit padding.
    Greedy mode is the sequential baseline and has no MAP guarantee.
    """
    scores = masked_log_scores(logits, masked, store.mask_token_id)
    device = scores.device
    if require_complete:
        targets = store.completion_targets(remaining_tokens, device=device)
    else:
        targets = torch.zeros(store.num_states, dtype=torch.bool, device=device)
        targets[store.live_states] = True
    # Sound fast path: a feasible position-wise argmax reaches the upper bound
    # on the score of ANY sequence. Mere validity of an old output is insufficient.
    if store.enable_oppurtunistic and not greedy and device.type != "cuda" and not defer_validation:
        candidate = scores.argmax(-1)
        if torch.isfinite(scores.gather(1, candidate[:, None])).all():
            reachable = store.reachable_states(candidate.tolist(), initial_state)
            feasible = [s for s in sorted(reachable) if targets[s]]
            if feasible:
                return candidate, feasible[0]
    src = store.edge_src.to(device)
    dst = store.edge_dst.to(device)
    tok = store.edge_tok.to(device)
    # Device scalar start states avoid scalar reads between blocks.
    states = torch.arange(store.num_states, device=device)
    init = torch.where(states == initial_state, 0.0, -torch.inf)
    if device.type == "cuda":
        cache = getattr(store, "_runtime_decoders", None)
        if cache is None:
            cache = store._runtime_decoders = {}
        key = (str(device), len(scores), scores.shape[1], greedy)
        if key not in cache:
            cache[key] = ViterbiGraph(
                src,
                dst,
                tok,
                len(scores),
                scores.shape[1],
                store.num_states,
                device,
                decode=True,
                greedy=greedy,
            )
        path, end, valid = cache[key].run(scores, init, targets)
    else:
        path, end, valid = constrained_path(src, dst, tok, scores, init, targets, greedy)
    if defer_validation:
        pending = getattr(store, "_runtime_validity", None)
        if pending is None:
            pending = store._runtime_validity = {}
        key = str(device)
        # Preserve any earlier failure even when a later step is feasible.
        previous = pending.get(key)
        pending[key] = valid.clone() if previous is None else previous & valid
    elif not bool(valid):
        raise ValueError("No supported constrained path for this block and token budget")
    # Never publish invalid backpointer IDs, even before deferred validation.
    path = torch.where(valid, path, store.mask_token_id)
    return path, end if device.type == "cuda" else int(end.item())


def validate_decode_block(store, device):
    """One host validity check per block; failures cannot leak to another request."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    pending = getattr(store, "_runtime_validity", {})
    valid = pending.pop(str(device), None)
    if valid is not None and not bool(valid):
        raise ValueError("No supported constrained path for this block and token budget")


def reset_decode_validation(store):
    """Discard validity flags when starting/finishing a generation request."""
    getattr(store, "_runtime_validity", {}).clear()


def release_decode_cache(store):
    """Release graph and completion tensors when evicting a device automaton."""
    for name in ("_runtime_decoders", "_runtime_completion", "_runtime_validity"):
        store.__dict__.pop(name, None)
