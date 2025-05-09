import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig, __version__
from typing import Optional, Union
import torch.distributions as dists
import torch.nn.functional as F
import warnings
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from .viterbi import viterbi_block
from .gen_utils import find_target_after_phrase, find_target_after_phrase_str


def get_num_transfer_tokens(mask_index: torch.BoolTensor, steps: int) -> torch.LongTensor:
    """
    Pre-computes, for every batch element, how many tokens should be revealed at
    each reverse step so that the **expected** number per step is constant.
    Returned shape: [B, steps]
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)                 # [B,1]
    base      = mask_num // steps
    remainder = mask_num %  steps

    n = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                    dtype=torch.long) + base                       # [B,steps]
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
        self.alg: str = kwargs.pop("alg", 'origin')
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
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass
    

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

class DREAM: 
    def __init__(self, model, tokenizer, constraint_mode = 'original', steps = 128, gen_length = 128,
                 block_length = 128, temperature = 0.0, cfg_scale = 0.0, remasking = 'low_confidence', constrain_at = 30, stop_word = None, stop_phrase = None, opp_end_char = None):
        
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
        self.target_word = stop_word
        self.phrase = stop_phrase
        self.opp_end_char = opp_end_char
        
    def __call__(self, prompt, dfa_store = None):
        if dfa_store is not None:
            self.dfa_store = dfa_store
            self.edge_src = dfa_store.edge_src
            self.edge_dst = dfa_store.edge_dst
            self.edge_tok = dfa_store.edge_tok
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if self.constraint_mode in ['unconstrained', 'ar_constrained', 'diffusion_constrained']:
            output_ids = self.generate(inputs.input_ids, inputs.attention_mask, constraint_mode=self.constraint_mode, max_new_tokens=self.gen_length,
                                                output_history=False,
                                                return_dict_in_generate=False,
                                                steps=self.steps,
                                                temperature=self.temperature,
                                                alg="entropy",
                                                alg_temp=0., 
                                                top_k = None, 
                                                top_p = None)
        else:
            raise ValueError(f"Invalid constraint mode: {self.constraint_mode}")
        return self.tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        constraint_mode: str = 'original',
        **kwargs,
    ) -> Union[torch.LongTensor]:
        kwargs['mask_token_id'] = self.tokenizer.mask_token_id
        generation_config = self.model.generation_config
    
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self.model._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        device = input_ids.device
        self.model._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self.model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self.model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
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
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self.model._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        
        result = self.constrained_sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func, 
            constraint_mode=constraint_mode
        )

        
        return result
    
    def constrained_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        constraint_mode: str = None
    ) -> Union[torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps_total = generation_config.steps
        
        if constraint_mode in ['diffusion_constrained', 'ar_constrained']:
            assert self.constrain_at is not None


        histories = [] if (return_dict_in_generate and output_history) else None

        assert self.gen_length % self.block_length == 0, "max_length must be divisible by block_length"
        num_blocks = self.gen_length // self.block_length
        assert steps_total % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block  = steps_total // num_blocks
        
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if constraint_mode == 'diffusion_constrained':
            num_states = self.dfa_store.num_states
            cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
            cost_vector[self.dfa_store.initial_state] = 0.0

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
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
            end_gen_idx   = input_ids.shape[1] + (num_block + 1) * self.block_length

            # pre-compute constant-expected-reveal schedule
            block_mask_init = (x[:, start_gen_idx:end_gen_idx] == mask_token_id)
            k_per_step = get_num_transfer_tokens(block_mask_init, steps_per_block)
            
            timesteps = torch.linspace(1, generation_config.eps,
                                   steps_per_block + 1, device=x.device)
            
            active_block_mask = _active_block_mask(x.size(1), slice(start_gen_idx, end_gen_idx),
                                               x.device, x.size(0))
            
            opp_stop = False
            for i in range(steps_per_block):
                mask_all   = (x == mask_token_id)
                mask_block = mask_all & active_block_mask

                logits = self.model(x, attention_mask, None).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = generation_logits_hook_func(i, x, logits)

                mask_logits = logits[mask_block]
                
                if generation_config.alg == "origin":
                    p_t = 1 - timesteps[i+1] / timesteps[i] if i < steps_per_block-1 else 1
                    rand_take = torch.rand(mask_logits.shape[0], device=x.device) < p_t
                    _, new_ids = sample_tokens(mask_logits[rand_take],
                                            temperature=generation_config.temperature,
                                            top_p=generation_config.top_p,
                                            top_k=generation_config.top_k)
                    x_tmp = torch.full_like(mask_logits, mask_token_id, dtype=torch.long)
                    x_tmp[rand_take] = new_ids
                    x[mask_block] = x_tmp                              ### FIX restrict
                else:
                    kwargs = dict(
                        temperature=generation_config.temperature,
                        top_p=generation_config.top_p,
                        top_k=generation_config.top_k,
                        margin_confidence=(generation_config.alg == "topk_margin"),
                        neg_entropy       =(generation_config.alg == "entropy"),
                    )
                    conf, new_ids = sample_tokens(mask_logits, **kwargs)

                    # block-wide confidence tensor
                    conf_full = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                    conf_full[mask_block] = conf
                    # exclude *future* blocks
                    conf_full[:, end_gen_idx:] = -torch.inf              ### FIX future leak

                    # how many to reveal this step / batch
                    transfer = torch.zeros_like(mask_block, dtype=torch.bool)
                    for b_id in range(x.size(0)):
                        k = int(k_per_step[b_id, i])
                        if k == 0:
                            continue
                        k = min(k, mask_block[b_id].sum().item())       ### FIX corner
                        if k == 0:
                            continue
                        _, top_idx = torch.topk(conf_full[b_id], k)
                        transfer[b_id, top_idx] = True

                    # write new ids
                    x_write = torch.full_like(x, mask_token_id)
                    x_write[mask_block] = new_ids
                    x[transfer] = x_write[transfer]


                # this allows user-defined token control of the intermediate steps
                x = generation_tokens_hook_func(i, x, logits)

                if self.constrain_at is not None and i >= self.constrain_at: 
                    target_idx = -1
                    if self.target_word is not None and self.phrase is not None:
                        decoded_gen = self.tokenizer.decode(x[0, input_ids.shape[1]:end_gen_idx], skip_special_tokens=True)
                        target_idx = find_target_after_phrase_str(decoded_gen, self.phrase, self.target_word)
                        if target_idx != -1:
                            decoded_gen = target_idx

                        if self.opp_end_char is not None and not decoded_gen.endswith(self.opp_end_char) and (num_block == num_blocks - 1 or target_idx != -1):
                            decoded_gen = decoded_gen + self.opp_end_char
                        
                        tokd_gen = self.tokenizer(decoded_gen).input_ids
                    else:
                        tokd_gen = x[0, input_ids.shape[1]:end_gen_idx].tolist()
                    
                    
                    if constraint_mode == 'ar_constrained':
                        if not self.dfa_store.enable_oppurtunistic or not self.dfa_store.check_is_reachable(tokd_gen, is_final_block = num_block == num_blocks - 1):
                            gen_ids = []                           # tokens we’ve accepted so far
                            ar_prefix = x[:, input_ids.shape[1]:start_gen_idx]
                            for j in range(start_gen_idx, end_gen_idx):
                                # 1. context ids that the processor will see
                                ar_x = torch.cat(
                                    [ar_prefix,  # frozen prefix
                                    torch.tensor(gen_ids, device=x.device).unsqueeze(0)],  # generated so far
                                    dim=1,
                                )
                                
                                current_logits = logits[:, j:j+1, :]
                                if not self.dfa_store.is_valid_ar(ar_x, current_logits.argmax(-1).squeeze(1)):
                                    current_logits = self.dfa_store.ar_logits_process(ar_x, current_logits)
                                
                                next_token = current_logits.argmax(-1).squeeze(1)   # greedy; replace with sampler if desired
                                gen_ids.append(next_token)

                                logits[:, j:j+1, :] = current_logits
                            
                            if gen_ids: 
                                x[:, start_gen_idx:end_gen_idx] = torch.stack(gen_ids, dim=1)
                            
                    elif constraint_mode == 'diffusion_constrained':
                        if self.dfa_store.enable_oppurtunistic and self.dfa_store.check_is_reachable(tokd_gen, is_final_block = num_block == num_blocks - 1):
                            if target_idx != -1: 
                                opp_stop = True 
                            else:
                                if i == steps_per_block - 1:
                                    best_intermediate_state = self.dfa_store.traverse_token_path(x[0, start_gen_idx:end_gen_idx].tolist(), cost_vector.argmax().item())
                        else:
                            gen_log_probs = F.log_softmax(logits[:, start_gen_idx:end_gen_idx, :], dim=-1)[0]
                            
                            current_cost, selected_tokens, selected_sources = viterbi_block(
                                self.edge_src,
                                self.edge_dst,
                                self.edge_tok,
                                gen_log_probs,
                                cost_vector
                            )
                            
                            if num_block == num_blocks - 1:
                                best_intermediate_state = self.dfa_store.final_states[torch.argmax(current_cost[self.dfa_store.final_states])]
                            else:
                                best_intermediate_state = torch.argmax(current_cost)
                            
                            token_path = []
                            current_state = best_intermediate_state
                            for t in range(self.block_length - 1, - 1, -1):
                                tok = selected_tokens[t][current_state].item()  
                                token_path.append(tok)
                                current_state = selected_sources[t][current_state].item() 
                            token_path.reverse()
                            x[:, start_gen_idx:end_gen_idx] = torch.tensor(token_path, dtype=torch.long, device=x.device)

                if histories is not None:
                    histories.append(x.clone())
        
            if opp_stop: 
                return x
            
            if constraint_mode == 'diffusion_constrained': 
                cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
                cost_vector[best_intermediate_state] = 0.0

        return x
    
    