import torch
import numpy as np
import torch.nn.functional as F
from .viterbi import viterbi_block, ViterbiGraph
from transformers import AutoTokenizer, AutoModel 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dfa import traverse_token_path


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


class LLADA: 
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
        self.mask_id = 126336
        self.dfa_store = dfa_store
        self.ar_logits_processor = ar_logits_processor
        self.constrain_at = constrain_at
        if self.dfa_store is not None:
            self.edge_src = self.dfa_store.edge_src.clone().to(self.model.device)
            self.edge_dst = self.dfa_store.edge_dst.clone().to(self.model.device)
            self.edge_tok = self.dfa_store.edge_tok.clone().to(self.model.device)


    def __call__(self, prompt):
        input_ids = self.tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.model.device).unsqueeze(0)
        
        if self.constraint_mode in ['diffusion_constrained', 'ar_constrained', 'unconstrained']:
            output_ids = self.generate(input_ids, constraint_mode = self.constraint_mode)
        
        else:
            raise NotImplementedError(self.constraint_mode)
        
        return self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        

    @torch.no_grad()
    def generate(self, prompt, constraint_mode = 'unconstrained'):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length.
                        If < gen_length, we do semi-autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The token id of [MASK] is 126336.
            initial_state: The ID of the start state in transitions_by_state.
            final_states: A list of valid final states.
            transition_matrix: transition_matrix[q, q'] = [list of valid tokens
                            that take state q -> q'] or -1 if invalid.
        '''

        x = torch.full((1, prompt.shape[1] + self.gen_length),
                    self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert self.gen_length % self.block_length == 0
        num_blocks = self.gen_length // self.block_length

        assert self.steps % num_blocks == 0
        steps = self.steps // num_blocks

        if constraint_mode == 'diffusion_constrained': 
            num_states = self.dfa_store.num_states
            cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
            cost_vector[self.dfa_store.initial_state] = 0.0


        for num_block in range(num_blocks):
            start_gen_idx = prompt.shape[1] + num_block * self.block_length
            end_gen_idx   = prompt.shape[1] + (num_block + 1) * self.block_length
            block_mask_index = (x[:, start_gen_idx:end_gen_idx] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == self.mask_id)
                if self.cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits
                logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)


                x0 = torch.argmax(logits_with_noise, dim=-1)

                if self.remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float32), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif self.remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(self.remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * self.block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                if i >= self.constrain_at:
                    decoded_gen = self.tokenizer.decode(x[0, prompt.shape[1]:end_gen_idx], skip_special_tokens=True)
                    if constraint_mode == 'ar_constrained': 
                        if not self.ar_logits_processor.enable_oppurtunistic or not self.ar_logits_processor.validate(decoded_gen, is_final_block = num_block == num_blocks - 1):
                            gen_ids = []                           # tokens we’ve accepted so far
                            ar_prefix = x[:, prompt.shape[1]:start_gen_idx]
                            
                            for j in range(start_gen_idx, end_gen_idx):
                                # 1. context ids that the processor will see
                                ar_x = torch.cat(
                                    [ar_prefix,  # frozen prefix
                                    torch.tensor(gen_ids, device=x.device).unsqueeze(0)],  # generated so far
                                    dim=1,
                                )

                                # 2. logits for current position
                                current_logits = logits_with_noise[:, j:j+1, :]

                                # 3. optionally change them
                                if not self.ar_logits_processor.is_valid(ar_x, current_logits.argmax(-1).squeeze(1)):
                                    current_logits = self.ar_logits_processor(ar_x, current_logits)

                                # 4. sample / pick the token and store it
                                next_token = current_logits.argmax(-1).squeeze(1)   # greedy; replace with sampler if desired
                                gen_ids.append(next_token)

                                # 5. (optional) put the processed logits back if other code later relies on them
                                logits_with_noise[:, j:j+1, :] = current_logits
                            
                            if gen_ids: 
                                x[:, start_gen_idx:end_gen_idx] = torch.stack(gen_ids, dim=1)
                    
                    elif constraint_mode == 'diffusion_constrained': 
                        if self.dfa_store.enable_oppurtunistic and self.dfa_store.validate(decoded_gen, is_final_block = num_block == num_blocks - 1):
                            best_intermediate_state = self.dfa_store.traverse_token_path(x[0, start_gen_idx:end_gen_idx].tolist(), cost_vector.argmax().item())
                        else:
                            gen_log_probs = F.log_softmax(logits_with_noise[:, start_gen_idx:end_gen_idx, :], dim=-1)[0]
                            

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
                            x[:, start_gen_idx:end_gen_idx] = torch.tensor(token_path, dtype=torch.long, device=x0.device)

            if constraint_mode == 'diffusion_constrained': 
                cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
                cost_vector[best_intermediate_state] = 0.0
        
        return x