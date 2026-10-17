import torch

from tokenization import get_mask_token_id

from .constrained_decode import decode_block, reset_decode_validation, validate_decode_block
from .gen_utils import completed_stop_prefix, reveal_schedule, validate_generation_settings


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    # Stable log-domain Gumbel max, equivalent in ranking to exp(logits)
    # / (-log(U))**temperature, without overflow at large logits.
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits).clamp_min(torch.finfo(torch.float64).tiny)
    return logits - temperature * torch.log(-torch.log(noise))


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


class LLADA:
    def __init__(
        self,
        model,
        tokenizer,
        constraint_mode="unconstrained",
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        constrain_at=0,
        stop_word=None,
        stop_phrase=None,
        require_complete=True,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.constraint_mode = constraint_mode
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.mask_id = get_mask_token_id(tokenizer)
        self.require_complete = require_complete
        self.constrain_at = constrain_at

        self.target_word = stop_word
        self.phrase = stop_phrase

    def __call__(self, prompt, dfa_store):
        if dfa_store is not None:
            self.dfa_store = dfa_store

        input_ids = self.tokenizer(prompt)["input_ids"]
        input_ids = torch.tensor(input_ids).to(self.model.device).unsqueeze(0)

        if self.constraint_mode in ["diffusion_constrained", "ar_constrained", "unconstrained"]:
            output_ids = self.generate(input_ids, constraint_mode=self.constraint_mode)

        else:
            raise NotImplementedError(self.constraint_mode)

        return self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @torch.no_grad()
    def generate(self, prompt, constraint_mode="unconstrained"):
        steps = validate_generation_settings(
            self.steps,
            self.gen_length,
            self.block_length,
            self.constrain_at,
            constraint_mode,
            prompt.shape[0],
        )
        if prompt.shape[0] != 1:
            raise ValueError("LLaDA generation currently supports batch_size=1")
        constrained = constraint_mode in ("diffusion_constrained", "ar_constrained")
        if constrained and not hasattr(self, "dfa_store"):
            raise ValueError("Constrained generation requires a DFA store")
        if constrained:
            reset_decode_validation(self.dfa_store)
        x = torch.full(
            (1, prompt.shape[1] + self.gen_length),
            self.mask_id,
            dtype=torch.long,
            device=prompt.device,
        )
        x[:, : prompt.shape[1]] = prompt
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, : prompt.shape[1]] = True
        state = self.dfa_store.initial_state if constrained else None
        for block in range(self.gen_length // self.block_length):
            start = prompt.shape[1] + block * self.block_length
            end = start + self.block_length
            schedule = reveal_schedule(self.block_length, steps)
            for i in range(steps):
                model_x = x[:, :end]
                if self.cfg_scale > 0.0:
                    un_x = model_x.clone()
                    un_x[prompt_index[:, :end]] = self.mask_id
                    logits, un_logits = self.model(torch.cat([model_x, un_x], dim=0)).logits.chunk(
                        2
                    )
                    logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(model_x).logits
                # Only the active block participates in sampling/remasking.
                block_logits = logits[:, start:end].float().clone()
                block_logits[:, :, self.mask_id] = -torch.inf
                proposed = add_gumbel_noise(block_logits, self.temperature).argmax(-1)
                masked = x[:, start:end] == self.mask_id
                if self.remasking == "low_confidence":
                    confidence = (
                        block_logits.softmax(-1).gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
                    )
                elif self.remasking == "random":
                    confidence = torch.rand_like(proposed, dtype=torch.float)
                else:
                    raise ValueError(f"Unsupported remasking strategy: {self.remasking}")
                confidence = confidence.masked_fill(~masked, -torch.inf)
                k = schedule[i]
                if k:
                    indices = confidence[0].topk(k).indices
                    x[0, start + indices] = proposed[0, indices]
                if constrained and i >= self.constrain_at:
                    path, end_state = decode_block(
                        self.dfa_store,
                        block_logits[0],
                        x[0, start:end] == self.mask_id,
                        state,
                        x.shape[1] - end,
                        require_complete=self.require_complete,
                        greedy=constraint_mode == "ar_constrained",
                        defer_validation=True,
                    )
                    x[0, start:end] = path
                if i == steps - 1:
                    if constrained:
                        validate_decode_block(self.dfa_store, x.device)
                    stopped = completed_stop_prefix(
                        x,
                        prompt.shape[1],
                        end,
                        self.tokenizer,
                        self.mask_id,
                        self.phrase,
                        self.target_word,
                        self.dfa_store if constrained else None,
                    )
                    if stopped is not None:
                        return stopped
            if constrained:
                state = end_state
        return x
