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
    transition_matrix = torch.zeros((len(transitions_by_state), len(transitions_by_state), max_tokens_transition), dtype=torch.long) - 1
    for state, transitions in transitions_by_state.items():
        for next_state, token_ids in transitions.items(): 
            transition_matrix[state, next_state, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    
    return transition_matrix
            



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, transitions_by_state=None, initial_state=None, final_states=None, transition_matrix=None):
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
    
    cost_vector = torch.zeros((len(transitions_by_state), gen_length + 1), device=x.device) - torch.inf
    cost_vector[initial_state, 0] = 0
    
    selected_tokens = torch.zeros((len(transitions_by_state), gen_length + 1), device=x.device, dtype=torch.long) - 1
    
    vec_iter = 0

    for num_block in range(num_blocks):
        start_gen_idx = prompt.shape[1] + num_block * block_length
        end_gen_idx = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, start_gen_idx: end_gen_idx] == mask_id)
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
                logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            
            gen_logits = logits_with_noise[:, start_gen_idx: end_gen_idx, :]

            
            gen_log_probs = F.log_softmax(gen_logits, dim=-1)[0]  # [block_length, vocab_size]
            
            current_cost = cost_vector[:, vec_iter]  
            num_states = len(current_cost)
            
            
            valid_mask = transition_matrix >= 0 

            for pos in range(block_length):
                token_log_probs = gen_log_probs[pos]
                

                max_transition_log_probs = torch.zeros((num_states, num_states), device=gen_log_probs.device) - torch.inf
                max_transition_tokens = torch.zeros((num_states, num_states), device=gen_log_probs.device, dtype=torch.long) - 1

                for q in range(num_states):
                    if current_cost[q] == -torch.inf:
                        continue 
                        
                    for q_prime in range(num_states):
                        valid_indices = valid_mask[q, q_prime]
                        if not valid_indices.any():
                            continue 
                        
                        valid_tokens = transition_matrix[q, q_prime][valid_indices]
                        if len(valid_tokens) > 0:
                            token_probs = token_log_probs[valid_tokens]
                            max_idx = torch.argmax(token_probs)
                            max_transition_log_probs[q, q_prime] = token_probs[max_idx]
                            max_transition_tokens[q, q_prime] = valid_tokens[max_idx]

                
                weighted_log_probs = max_transition_log_probs + current_cost.unsqueeze(1)
                
        
                max_values, max_indices = torch.max(weighted_log_probs, dim=0)
                next_cost = max_values
                

                for q_prime in range(num_states):
                    if torch.any(weighted_log_probs[:, q_prime] > -torch.inf):
                        best_source = max_indices[q_prime]
                        selected_tokens[q_prime, vec_iter+1] = max_transition_tokens[best_source, q_prime]

                current_cost = next_cost
                vec_iter += 1
                cost_vector[:, vec_iter] = current_cost
            

            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            mask_index = (x == mask_id)
            
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

    # Find the final state with the highest cost
    final_costs = torch.full((1,), -torch.inf, device=x.device)
    for state in final_states:
        if state < cost_vector.shape[0]:
            final_costs = torch.max(final_costs, cost_vector[state, vec_iter])
    
    best_final_state = None
    for state in final_states:
        if state < cost_vector.shape[0] and cost_vector[state, vec_iter] == final_costs:
            best_final_state = state
            break

    if best_final_state is not None:
        best_tokens = []
        current_state = best_final_state
        for pos in range(vec_iter, 0, -1):
            token = selected_tokens[current_state, pos]
            if token >= 0: 
                best_tokens.insert(0, token)
                for prev_state in range(len(transitions_by_state)):
                    if (prev_state in transitions_by_state and 
                        current_state in transitions_by_state[prev_state] and 
                        int(token.item()) in transitions_by_state[prev_state][current_state]):
                        current_state = prev_state
                        break
        
        x[0, prompt.shape[1]:] = torch.tensor(best_tokens, dtype=torch.long, device=x.device)
    
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
    
    pattern = r'(ab)+'
    
    transitions_by_state, initial_state, final_states, transition_matrix = run_pattern_test(pattern, tokenizer)

    out = generate(model, input_ids, steps=1, gen_length=4, block_length=4, temperature=0., cfg_scale=0., remasking='low_confidence', transitions_by_state=transitions_by_state, initial_state=initial_state, final_states=final_states, transition_matrix=transition_matrix)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()