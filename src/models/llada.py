import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
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
    def __init__(self, model, tokenizer, constraint_mode = 'original', dfa_store = None, steps = 128, gen_length = 128,
                 block_length = 128, temperature = 0.0, cfg_scale = 0.0, remasking = 'low_confidence'):
        
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
        
    def __call__(self, prompt):
        input_ids = self.tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.model.device).unsqueeze(0)
        
        if self.constraint_mode == 'unconstrained':
            output_ids = self.generate_unconstrained(input_ids)
        
        elif self.constraint_mode == 'diffusion_constrained':
            output_ids = self.generate_diffusion_constrained(input_ids)
        
        else:
            raise NotImplementedError(self.constraint_mode)
        
        return self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
    
    @torch.no_grad()
    def generate_unconstrained(self, prompt):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        x = torch.full((1, prompt.shape[1] + self.gen_length), self.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert self.gen_length % self.block_length == 0
        num_blocks = self.gen_length // self.block_length

        assert self.steps % num_blocks == 0
        self.steps = self.steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * self.block_length: prompt.shape[1] + (num_block + 1) * self.block_length:] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps)
            for i in range(self.steps):
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
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if self.remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
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

        return x


    @torch.no_grad()
    def generate_diffusion_constrained(self, prompt):
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
        self.steps = self.steps // num_blocks

        num_states = self.dfa_store.transition_matrix.shape[0]

        cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
        cost_vector[self.dfa_store.initial_state] = 0.0

        selected_tokens = torch.full((num_states, self.gen_length + 1),
                                    -1, device=x.device, dtype=torch.long)

        selected_sources = torch.full((num_states, self.gen_length + 1),
                                    -1, device=x.device, dtype=torch.long)

        global_vec_iter = 0

        for num_block in range(num_blocks):
            start_gen_idx = prompt.shape[1] + num_block * self.block_length
            end_gen_idx   = prompt.shape[1] + (num_block + 1) * self.block_length
            block_mask_index = (x[:, start_gen_idx:end_gen_idx] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps)
            vec_iter = global_vec_iter
            current_cost = cost_vector.clone()
            for i in range(self.steps):
                current_cost = cost_vector.clone()
                vec_iter = global_vec_iter
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

                gen_logits = logits_with_noise[:, start_gen_idx:end_gen_idx, :] 
                gen_log_probs = F.log_softmax(gen_logits, dim=-1)[0]

                valid_mask = (self.dfa_store.transition_matrix >= 0)

                gen_log_probs = gen_log_probs.unsqueeze(1).unsqueeze(2)  # => [block_length, 1, 1, vocab_size]
                gen_log_probs = gen_log_probs.expand(-1, num_states, num_states, -1)

                # transition_matrix shape: [num_states, num_states, max_tokens_per_transition]
                # valid_mask shape: [num_states, num_states, max_tokens_per_transition]

                # 1) clamp negative indices to 0
                clamped_indices = self.dfa_store.transition_matrix.clamp(min=0)

                # 2) expand to 4D
                clamped_indices_4d = clamped_indices.unsqueeze(0).expand(
                    gen_log_probs.shape[0], -1, -1, -1
                ).to(gen_log_probs.device)
                # shape: [block_length, num_states, num_states, max_tokens_per_transition]

                # 3) gather
                gathered_log_probs = torch.gather(
                    gen_log_probs,  # [block_length, num_states, num_states, vocab_size]
                    dim=3,
                    index=clamped_indices_4d
                )
                # => shape: [block_length, num_states, num_states, max_tokens_per_transition]

                # 4) mask out the invalid transitions
                valid_mask_4d = (self.dfa_store.transition_matrix >= 0).unsqueeze(0).expand(
                    gen_log_probs.shape[0], -1, -1, -1,
                ).to(x.device)
                gathered_log_probs = gathered_log_probs.masked_fill(~valid_mask_4d, float('-inf'))

                # 5) max over the last dimension => best among tokens
                max_log_probs, best_tokens_idx = gathered_log_probs.max(dim=3)
                
                best_tokens = torch.gather(
                    clamped_indices_4d, dim=3, index=best_tokens_idx.unsqueeze(3)
                ).squeeze(3)

                for pos in range(self.block_length):
                    # max_log_probs[pos] => shape [num_states, num_states]
                    # best_tokens[pos]   => shape [num_states, num_states]

                    # Weighted log-probs = cost(q) + max_log_probs[pos,q,q']
                    weighted_log_probs = max_log_probs[pos] + current_cost.unsqueeze(1)
                    # shape => [num_states, num_states]

                    # next_cost[q'] = max over q
                    max_values, max_indices = torch.max(weighted_log_probs, dim=0)
                    next_cost = max_values  # shape [num_states]

                    # record back-pointer
                    for q_prime in range(num_states):
                        if next_cost[q_prime] != float('-inf'):
                            best_source = max_indices[q_prime]
                            selected_tokens[q_prime, vec_iter+1] = best_tokens[pos, best_source, q_prime]
                            selected_sources[q_prime, vec_iter+1] = best_source

                    current_cost = next_cost
                    vec_iter += 1

                x0 = torch.argmax(logits_with_noise, dim=-1)  

                if self.remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
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

                if i == self.steps - 1:
                    if num_block == num_blocks - 1:
                        best_intermediate_state = self.dfa_store.final_states[torch.argmax(current_cost[self.dfa_store.final_states])]
                    else:
                        best_intermediate_state = torch.argmax(current_cost)

                    token_path = []
                    for t in range(vec_iter, global_vec_iter, -1):
                        tok = selected_tokens[best_intermediate_state, t].item()  
                        # if tok != 126081:
                        #     import pdb; pdb.set_trace()
                        token_path.append(tok)
                        best_intermediate_state = selected_sources[best_intermediate_state, t] 
                    token_path.reverse()
                    x[:, start_gen_idx:end_gen_idx] = torch.tensor(token_path, dtype=torch.long, device=x0.device)

            global_vec_iter = vec_iter
            cost_vector = current_cost
        # final_time = global_vec_iter
        # final_costs = cost_vector[self.dfa_store.final_states] 
        # best_final_index = torch.argmax(final_costs)
        # best_state = self.dfa_store.final_states[best_final_index]
        # best_cost  = final_costs[best_final_index]

        # token_path = []
        # curr_state = best_state

        # for t in range(final_time, 0, -1):
        #     tok = selected_tokens[curr_state, t].item()  
        #     token_path.append(tok)
        #     curr_state = selected_sources[curr_state, t] 

        # token_path.reverse()

        # x[:, prompt.shape[1]:] = torch.tensor(token_path, dtype=torch.long, device=x.device)

        return x