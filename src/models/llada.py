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
    def __init__(self, model, tokenizer, constraint_mode = 'original', dfa_store = None, steps = 128, gen_length = 128,
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
        self.constrain_at = constrain_at
        if self.dfa_store is not None:
            self._build_sparse_edges()
    
    def _build_sparse_edges(self):
        """Convert self.dfa_store.transition_matrix (shape [S,S,max_toks], -1 means invalid)
        into three 1‑D tensors edge_src, edge_dst, edge_tok that enumerate **only**
        valid transitions. Saves both memory and compute later.
        """
        tm = self.dfa_store.transition_matrix  # [S, S, max_tok]
        src, dst, tok = [], [], []

        # find all valid (q,q',token) triples
        valid = (tm >= 0).nonzero(as_tuple=False)   # [E, 3] (q, q', k)
        edge_src = valid[:, 0]
        edge_dst = valid[:, 1]
        tok_idx  = valid[:, 2]
        edge_tok = tm[edge_src, edge_dst, tok_idx]

        self.edge_src = edge_src.to(self.model.device)
        self.edge_dst = edge_dst.to(self.model.device)
        self.edge_tok = edge_tok.to(self.model.device)
        self.num_states = tm.shape[0]
        self.num_edges  = self.edge_src.numel()
        

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
        steps = self.steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * self.block_length: prompt.shape[1] + (num_block + 1) * self.block_length:] == self.mask_id)
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
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

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
        steps = self.steps // num_blocks

        num_states = self.dfa_store.transition_matrix.shape[0]

        cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
        cost_vector[self.dfa_store.initial_state] = 0.0

        global_vec_iter = 0

        for num_block in range(num_blocks):
            start_gen_idx = prompt.shape[1] + num_block * self.block_length
            end_gen_idx   = prompt.shape[1] + (num_block + 1) * self.block_length
            block_mask_index = (x[:, start_gen_idx:end_gen_idx] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            vec_iter = global_vec_iter
            current_cost = cost_vector.clone()
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

                gen_logits = logits_with_noise[:, start_gen_idx:end_gen_idx, :] 
                
                selected_tokens = []
                selected_sources = []
                
                if i >= self.constrain_at:
                    current_cost = cost_vector.clone()
                    vec_iter = global_vec_iter + self.block_length
                
                    gen_log_probs = F.log_softmax(gen_logits, dim=-1)[0]
                    

                    for pos in range(self.block_length):
                        tok_log = gen_log_probs[pos]
                        edge_score = current_cost[self.edge_src] + tok_log[self.edge_tok]
                        
                        # scatter‑max: next_cost[q'] = max_{edges src->q'} edge_score
                        next_cost = torch.full_like(current_cost, -float('inf'))
                        next_cost.scatter_reduce_(0, self.edge_dst, edge_score, reduce='amax')

                        # grab the winning edge per dst state (for back‑pointer)
                        mask = next_cost[self.edge_dst] == edge_score          # [E]
                        chosen_edges = mask.nonzero(as_tuple=False).squeeze(1)

                        next_tok  = torch.full_like(current_cost, -1, dtype=torch.long)
                        next_src  = torch.full_like(current_cost, -1, dtype=torch.long)
                        if chosen_edges.numel():
                            next_tok[self.edge_dst[chosen_edges]] = self.edge_tok[chosen_edges]
                            next_src[self.edge_dst[chosen_edges]] = self.edge_src[chosen_edges]

                        selected_tokens.append(next_tok)
                        selected_sources.append(next_src)

                        current_cost = next_cost

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
                    if num_block == num_blocks - 1:
                        best_intermediate_state = self.dfa_store.final_states[torch.argmax(current_cost[self.dfa_store.final_states])]
                    else:
                        best_intermediate_state = torch.argmax(current_cost)

                    token_path = []
                    for t in range(vec_iter - 1, global_vec_iter - 1, -1):
                        tok = selected_tokens[t][best_intermediate_state].item()  
                        token_path.append(tok)
                        best_intermediate_state = selected_sources[t][best_intermediate_state].item() 
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