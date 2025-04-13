import time, sys, os
import interegular
from syncode.mask_store.byte_fsm import ByteFSM
from transformers import AutoTokenizer
from typing import Dict, Tuple, Set, Optional, Any
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dfa import compute_token_transitions


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

def naive_matrix_build(transitions_by_state, max_tokens_transition): 
    max_key = -1
    for state in transitions_by_state: 
        for next_state in transitions_by_state[state]:  
            max_key = max(max_key, state, next_state) 
    
    assert max_key >= 0
    
    transition_matrix = torch.zeros(max_key + 1, max_key + 1, max_tokens_transition, dtype=torch.long) - 1
    for state, transitions in transitions_by_state.items():
        for next_state, token_ids in transitions.items(): 
            transition_matrix[state, next_state, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    
    return transition_matrix
            



@torch.no_grad()
def generate_constrained(model, prompt, steps=128, gen_length=128, block_length=128,
             temperature=0., cfg_scale=0., remasking='low_confidence',
             mask_id=126336, transitions_by_state=None,
             initial_state=None, final_states=None, transition_matrix=None):
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
        transitions_by_state: A representation of the states in your FSA.
        initial_state: The ID of the start state in transitions_by_state.
        final_states: A list of valid final states.
        transition_matrix: transition_matrix[q, q'] = [list of valid tokens
                         that take state q -> q'] or -1 if invalid.
    '''

    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    num_states = transition_matrix.shape[0]

    cost_vector = torch.full((num_states, ), fill_value=-float('inf'), dtype=torch.float, device=x.device)
    cost_vector[initial_state] = 0.0

    selected_tokens = torch.full((num_states, gen_length + 1),
                                 -1, device=x.device, dtype=torch.long)

    selected_sources = torch.full((num_states, gen_length + 1),
                                  -1, device=x.device, dtype=torch.long)

    global_vec_iter = 0

    for num_block in range(num_blocks):
        start_gen_idx = prompt.shape[1] + num_block * block_length
        end_gen_idx   = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, start_gen_idx:end_gen_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        vec_iter = global_vec_iter
        current_cost = cost_vector.clone()
        step_start = time.time()
        for i in range(steps):
            current_cost = cost_vector.clone()
            vec_iter = global_vec_iter
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                forward_pass_start = time.time()
                logits = model(x).logits
                print(f"Forward pass took {time.time() - forward_pass_start:.4f} seconds")
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            gen_logits = logits_with_noise[:, start_gen_idx:end_gen_idx, :] 
            gen_log_probs = F.log_softmax(gen_logits, dim=-1)[0]

            valid_mask = (transition_matrix >= 0)
            block_start = time.time()
            
            

            gen_log_probs = gen_log_probs.unsqueeze(1).unsqueeze(2)  # => [block_length, 1, 1, vocab_size]
            gen_log_probs = gen_log_probs.expand(-1, num_states, num_states, -1)

            # transition_matrix shape: [num_states, num_states, max_tokens_per_transition]
            # valid_mask shape: [num_states, num_states, max_tokens_per_transition]

            # 1) clamp negative indices to 0
            clamped_indices = transition_matrix.clamp(min=0)

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
            valid_mask_4d = (transition_matrix >= 0).unsqueeze(0).expand(
                gen_log_probs.shape[0], -1, -1, -1,
            ).to(x.device)
            gathered_log_probs = gathered_log_probs.masked_fill(~valid_mask_4d, float('-inf'))

            # 5) max over the last dimension => best among tokens
            max_log_probs, best_tokens_idx = gathered_log_probs.max(dim=3)
            
            best_tokens = torch.gather(
                clamped_indices_4d, dim=3, index=best_tokens_idx.unsqueeze(3)
            ).squeeze(3)

            for pos in range(block_length):
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
            
            print(f"block {i} took {time.time() - block_start:.4f} seconds")
            x0 = torch.argmax(logits_with_noise, dim=-1)  

            back_pointer_start = time.time()
            best_intermediate_state = torch.argmax(current_cost)
            token_path = []
            for t in range(vec_iter, global_vec_iter, -1):
                tok = selected_tokens[best_intermediate_state, t].item()  
                token_path.append(tok)
                best_intermediate_state = selected_sources[best_intermediate_state, t] 
            token_path.reverse()
            x0[:, start_gen_idx:end_gen_idx] = torch.tensor(token_path, dtype=torch.long, device=x0.device)
            print(f"Back translate took {time.time() - back_pointer_start:.4f} seconds")

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        step_end = time.time()
        print(f"Step {i} took {step_end - step_start:.4f} seconds")
        global_vec_iter = vec_iter
        cost_vector = current_cost
    final_time = global_vec_iter
    final_costs = cost_vector[final_states] 
    best_final_index = torch.argmax(final_costs)
    best_state = final_states[best_final_index]
    best_cost  = final_costs[best_final_index]

    token_path = []
    curr_state = best_state

    for t in range(final_time, 0, -1):
        tok = selected_tokens[curr_state, t].item()  
        token_path.append(tok)
        curr_state = selected_sources[curr_state, t] 

    token_path.reverse()

    x[:, prompt.shape[1]:] = torch.tensor(token_path, dtype=torch.long, device=x.device)

    return x


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                forward_pass_start = time.time()
                logits = model(x).logits
                print(f"Forward pass took {time.time() - forward_pass_start:.4f} seconds")

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

def run_pattern_test(pattern, tokenizer):
    """
    Run a test with the given pattern and tokenizer.
    
    Args:
        pattern: The regex pattern to test
        tokenizer: The tokenizer to use
        
    Returns:
        Tuple of (num_states, vocab_size, total_time, num_finals)
    """
    
    print(f"Creating ByteFSM for pattern: {pattern}")
    dfa = ByteFSM(pattern)
    # Get vocabulary size
    vocab_size = len(tokenizer.get_vocab())
    
    # Compute token transitions with timing
    print("Computing token transitions...")
    start_time = time.time()
    token_transitions, states = compute_token_transitions(tokenizer, dfa)
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get number of states
    num_states = len(states)
    
    # Print the required information
    print(f"\nResults:")
    print(f"Number of states in the DFA: {num_states}")
    print(f"Number of vocabulary tokens: {vocab_size}")
    print(f"Total time taken: {total_time:.4f} seconds")
    
    # Group transitions by starting state and next state
    transitions_by_state = {}
    for (state, token_id), next_state in token_transitions.items():
        if next_state is not None:  # Only consider valid transitions
            if state not in transitions_by_state:
                transitions_by_state[state] = {}
            if next_state not in transitions_by_state[state]:
                transitions_by_state[state][next_state] = []
            transitions_by_state[state][next_state].append(token_id)
    
    for final_state in dfa.finals:
        if final_state not in transitions_by_state:
            transitions_by_state[final_state] = {}
        
        if final_state not in transitions_by_state[final_state]:
            transitions_by_state[final_state][final_state] = []
        
        transitions_by_state[final_state][final_state].append(tokenizer.eos_token_id)
    
    
    max_tokens_transition = 0
    for state in transitions_by_state: 
        for next_state in transitions_by_state[state]: 
            max_tokens_transition = max(max_tokens_transition, len(transitions_by_state[state][next_state]))
       
    # Display only core DFA information
    print(f"\nDFA Information:")
    print(f"  - States: {num_states}")
    print(f"  - Final states: {len(dfa.finals)}")
    
    # Show just a few transitions from the first 5 states
    print("\nSample transitions (first 5 states only):")
    states_shown = 0
    
    
    for state in sorted(transitions_by_state.keys()):
        if states_shown >= 5:
            break
            
        print(f"\nFrom state {state}:")
        transitions_shown = 0
        
        # Show just 3 destination states and 1 token for each
        for next_state, token_ids in sorted(transitions_by_state[state].items()):
            if transitions_shown >= 3 or not token_ids:
                continue
                
            token_id = token_ids[0]  # Just show the first token
            token_text = tokenizer.decode([token_id])
            is_final = " (FINAL)" if next_state in dfa.finals else ""
            print(f"  → State {next_state}{is_final} via token: '{token_text}'")
            transitions_shown += 1
            
        states_shown += 1
    
    transition_matrix = naive_matrix_build(transitions_by_state, max_tokens_transition)

    return transitions_by_state, dfa.initial,dfa.finals, transition_matrix

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir = os.environ['HF_CACHE']).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, cache_dir = os.environ['HF_CACHE'])

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    pattern = r'4[0-9]*'
    #r'<<( )?(\(){0,3}( )?([a-z]|[0-9]{1,3})( )?(\)){0,3}(( )?([+\-]|\/\/|\/|%|\*|\*\*)( )?((\(){0,3}( )?)?([a-z]|[0-9]{1,3})( )?(\)){0,3})*( )?>>'
    
    transitions_by_state, initial_state, final_states, transition_matrix = run_pattern_test(pattern, tokenizer)

    start_time = time.time()
    out = generate(model, input_ids, steps=100, gen_length=8, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence')
    end_time = time.time()
    print(f"Unconstrained Time taken: {end_time - start_time:.4f} seconds")
    print("Unconstrained output:")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    
    
    
    start_time = time.time()
    out = generate_constrained(model, input_ids, steps=100, gen_length=8, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence', transitions_by_state=transitions_by_state, initial_state=initial_state, final_states=sorted(list(final_states)), transition_matrix=transition_matrix)
    end_time = time.time()
    print(f"Constrained Time taken: {end_time - start_time:.4f} seconds")
    print("Constrained output:")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    


if __name__ == '__main__':
    main()