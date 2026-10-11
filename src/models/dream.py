import warnings
from typing import Optional, Union

import torch
import torch.distributions as dists
import torch.nn.functional as F
from transformers import GenerationConfig, __version__
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
)

from .constrained_decode import decode_block, reset_decode_validation, validate_decode_block
from .gen_utils import completed_stop_prefix, reveal_schedule, validate_generation_settings


def get_num_transfer_tokens(mask_index: torch.BoolTensor, steps: int) -> torch.LongTensor:
    """
    Pre-computes, for every batch element, how many tokens should be revealed at
    each reverse step so that the **expected** number per step is constant.
    Returned shape: [B, steps]
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B,1]
    base = mask_num // steps
    remainder = mask_num % steps

    n = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    )  # [B,steps]
    for b in range(mask_num.size(0)):
        n[b, : remainder[b]] += 1
    return n


def _active_block_mask(seq_len: int, block_slice: slice, device, batch):
    """
    Returns a [B,L] boolean tensor that is True exactly on the positions
    belonging to block_slice.
    """
    m = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    m[:, block_slice] = True
    return m


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                setattr(self, key, value)

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < self.eps < 1:
            raise ValueError("eps must be in (0, 1)")
        if self.alg not in ("origin", "topk", "topk_margin", "entropy"):
            raise ValueError(f"Unsupported Dream algorithm: {self.alg}")


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, -torch.inf)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, -torch.inf)
    return logits


def sampling_logits(logits, temperature=0.0, top_p=None, top_k=None):
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    return logits


def sample_tokens(
    logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False
):
    logits = sampling_logits(logits, temperature, top_p, top_k)
    probs = torch.softmax(logits.float(), dim=-1)

    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


class DREAM:
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
        self.constrain_at = constrain_at
        self.require_complete = require_complete
        self.target_word = stop_word
        self.phrase = stop_phrase

    def __call__(self, prompt, dfa_store=None):
        if dfa_store is not None:
            self.dfa_store = dfa_store

        if self.cfg_scale != 0.0:
            raise ValueError("Dream classifier-free guidance is not supported")
        algorithms = {"low_confidence": "topk", "entropy": "entropy", "random": "origin"}
        if self.remasking not in algorithms:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if self.constraint_mode in ["unconstrained", "ar_constrained", "diffusion_constrained"]:
            output_ids = self.generate(
                inputs.input_ids,
                inputs.attention_mask,
                constraint_mode=self.constraint_mode,
                max_new_tokens=self.gen_length,
                output_history=False,
                return_dict_in_generate=False,
                steps=self.steps,
                temperature=self.temperature,
                alg=algorithms[self.remasking],
                alg_temp=0.0,
                top_k=None,
                top_p=None,
            )
        else:
            raise ValueError(f"Invalid constraint mode: {self.constraint_mode}")
        return self.tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        constraint_mode: str = "unconstrained",
        **kwargs,
    ) -> Union[torch.LongTensor]:
        kwargs["mask_token_id"] = self.tokenizer.mask_token_id
        generation_config = self.model.generation_config

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self.model._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop(
            "generation_tokens_hook_func", lambda step, x, logits: x
        )
        generation_logits_hook_func = kwargs.pop(
            "generation_logits_hook_func", lambda step, x, logits: logits
        )

        # 2. Define model inputs
        device = input_ids.device
        self.model._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None and generation_config.max_length is not None
        )
        generation_config = self.model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self.model._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
                stacklevel=2,
            )
        if (
            hasattr(generation_config, "pad_token_id")
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
                stacklevel=2,
            )

        input_ids, attention_mask = self.model._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        result = self.constrained_sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            constraint_mode=constraint_mode,
        )

        return result

    def constrained_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        constraint_mode: str = None,
    ) -> Union[torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps_total = generation_config.steps
        DreamGenerationConfig(
            **{
                name: getattr(generation_config, name)
                for name in ("temperature", "top_k", "top_p", "eps", "alg")
            }
        )

        constrained = constraint_mode in ("diffusion_constrained", "ar_constrained")
        steps_per_block = validate_generation_settings(
            steps_total,
            self.gen_length,
            self.block_length,
            self.constrain_at,
            constraint_mode,
            input_ids.shape[0],
        )
        if max_length != input_ids.shape[1] + self.gen_length:
            raise ValueError("max_new_tokens must equal the configured gen_length")
        if constrained and not hasattr(self, "dfa_store"):
            raise ValueError("Constrained generation requires a DFA store")
        if constrained:
            reset_decode_validation(self.dfa_store)
        histories = [] if (return_dict_in_generate and output_history) else None
        num_blocks = self.gen_length // self.block_length
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        state = self.dfa_store.initial_state if constrained else None

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        for num_block in range(num_blocks):
            start_gen_idx = input_ids.shape[1] + num_block * self.block_length
            end_gen_idx = input_ids.shape[1] + (num_block + 1) * self.block_length

            # pre-compute constant-expected-reveal schedule
            k_per_step = reveal_schedule(self.block_length, steps_per_block)

            timesteps = torch.linspace(
                1, generation_config.eps, steps_per_block + 1, device=x.device
            )

            for i in range(steps_per_block):
                active_x = x[:, :end_gen_idx]
                active_attention = (
                    attention_mask[:, :, :end_gen_idx, :end_gen_idx]
                    if isinstance(attention_mask, torch.Tensor)
                    else attention_mask
                )
                active_positions = tok_idx[:, :end_gen_idx] if tok_idx is not None else None
                logits = self.model(active_x, active_attention, active_positions).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = generation_logits_hook_func(i, active_x, logits)
                logits = logits.float().clone()
                logits[:, :, mask_token_id] = -torch.inf
                block_logits = logits[:, start_gen_idx:end_gen_idx]
                masked = x[:, start_gen_idx:end_gen_idx] == mask_token_id
                kwargs = dict(
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    margin_confidence=(generation_config.alg == "topk_margin"),
                    neg_entropy=(generation_config.alg == "entropy"),
                )
                conf, new_ids = sample_tokens(block_logits, **kwargs)
                if generation_config.alg == "origin":
                    p_t = 1 - timesteps[i + 1] / timesteps[i] if i < steps_per_block - 1 else 1
                    transfer = (torch.rand_like(conf) < p_t) & masked
                else:
                    k = k_per_step[i]
                    # Keep the original prompt+block topk shape: its tie
                    # choices affect later model conditioning.
                    confidence = torch.full_like(active_x, -torch.inf, dtype=conf.dtype)
                    confidence[:, start_gen_idx:end_gen_idx] = conf.masked_fill(~masked, -torch.inf)
                    transfer = torch.zeros_like(masked)
                    if k:
                        indices = confidence.topk(k, dim=-1).indices - start_gen_idx
                        transfer.scatter_(1, indices.clamp(0, self.block_length - 1), True)
                    transfer &= masked
                x[:, start_gen_idx:end_gen_idx] = torch.where(
                    transfer, new_ids, x[:, start_gen_idx:end_gen_idx]
                )

                # this allows user-defined token control of the intermediate steps
                x[:, :end_gen_idx] = generation_tokens_hook_func(i, active_x, logits)

                if constrained and i >= self.constrain_at:
                    path, end_state = decode_block(
                        self.dfa_store,
                        sampling_logits(
                            logits[0, start_gen_idx:end_gen_idx],
                            generation_config.temperature,
                            generation_config.top_p,
                            generation_config.top_k,
                        ),
                        x[0, start_gen_idx:end_gen_idx] == mask_token_id,
                        state,
                        x.shape[1] - end_gen_idx,
                        require_complete=self.require_complete,
                        greedy=constraint_mode == "ar_constrained",
                        defer_validation=True,
                    )
                    x[0, start_gen_idx:end_gen_idx] = path
                if i == steps_per_block - 1:
                    if constrained:
                        validate_decode_block(self.dfa_store, x.device)
                    stopped = completed_stop_prefix(
                        x,
                        input_ids.shape[1],
                        end_gen_idx,
                        self.tokenizer,
                        mask_token_id,
                        self.phrase,
                        self.target_word,
                        self.dfa_store if constrained else None,
                    )
                    if stopped is not None:
                        if return_dict_in_generate:
                            return ModelOutput(sequences=stopped, history=histories)
                        return stopped

                if histories is not None:
                    histories.append(x.clone())

            if constrained:
                state = end_state

        if return_dict_in_generate:
            return ModelOutput(sequences=x, history=histories)
        return x
