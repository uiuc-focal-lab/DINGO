import torch
import torch.distributions as dists
import torch.nn.functional as F
from .attention_patch import replace_attention_mask 
replace_attention_mask()
import transformers

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
import torch.nn as nn


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def top_p_logits(logits, p=0.9):
    if p == 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


class DiffuLlama: 
    def __init__(self, model, tokenizer, constraint_mode = 'original', dfa_store = None, ar_logits_processor = None, steps = 128, gen_length = 128,
                 block_length = 128, temperature = 0.0, cfg_scale = 0.0, remasking = 'low_confidence', constrain_at = 30):
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.constraint_mode = constraint_mode 
        self.steps = steps 
        self.gen_length = gen_length 
        self.block_length = block_length 
        self.temperature = temperature 
        self.cfg_scale = cfg_scale 
        self.remasking = remasking 
        self.dfa_store = dfa_store
        self.ar_logits_processor = ar_logits_processor
        self.constrain_at = constrain_at
        self.shift = True
        if self.dfa_store is not None:
            self.edge_src = self.dfa_store.edge_src.clone().to(self.model.device)
            self.edge_dst = self.dfa_store.edge_dst.clone().to(self.model.device)
            self.edge_tok = self.dfa_store.edge_tok.clone().to(self.model.device)
    
    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt") 
        
        if self.constraint_mode in ['unconstrained', 'diffusion_constrained', 'ar_constrained']:
            output_ids = self.generate(inputs, constraint_mode = self.constraint_mode)
        else:
            raise ValueError(f"Invalid constraint mode: {self.constraint_mode}")
        
        return self.tokenizer.batch_decode(output_ids[:, len(inputs.input_ids):], skip_special_tokens=True)[0]
    
    @torch.no_grad()
    def generate(self, inputs, constraint_mode = 'unconstrained'):
        """
            select 1/T% tokens to denoise at each step
        """
        logits_temp = self.temperature
        topp_temp = 1.0

        x = inputs["input_ids"].to(self.model.device)
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).to(self.model.device)
        else:
            src_mask = inputs["src_mask"].bool().to(self.model.device)

        x_embed = self.model.get_embeds(x)
        seq_len = x.size(1)
        batch_size = x.size(0)
        attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0) # all 0

        init_maskable_mask = maskable_mask = ~src_mask
        
        # first forward, all position except src is [M]
        xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)

        logits = self.model(xt, attention_mask=attention_mask)
        temp_scaled_logits = logits/logits_temp if logits_temp > 0 else logits
        filter_logits = top_p_logits(temp_scaled_logits, p=topp_temp)
        scores = torch.log_softmax(filter_logits, dim=-1)
        # x0_scores, x0 = scores.max(-1)
        # scores = scores.to(torch.float16)
        x0 = dists.Categorical(logits=scores).sample()
        x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

        if self.shift:
            #### deal with shift, left most token will be replaced anyway
            x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
            x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
        
        #### replace output of non-[MASK] positions with xt
        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        for t in range(self.steps-1, 0, -1): # t from T-1 to 1
            with torch.no_grad():
                #### select rate% tokens to be still [MASK]
                p_to_x0 = 1/(t+1)
                
                masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
                xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
                maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)

                logits = model(xt, attention_mask=attention_mask)
                filter_logits = top_p_logits(logits/logits_temp, p=topp_temp)
                scores = torch.log_softmax(filter_logits, dim=-1)
                x0 = dists.Categorical(logits=scores).sample()
                x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

                if self.shift:
                    #### deal with shift, left most token will be replaced anyway
                    x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                    x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
                
                # replace output of non-[MASK] positions with xt
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
        if self.shift:
            x0 = x0[:,1:]

        return x0