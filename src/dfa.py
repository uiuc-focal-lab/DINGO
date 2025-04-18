import unittest
import time
import interegular
from syncode.mask_store.byte_fsm import ByteFSM
from transformers import AutoTokenizer
from typing import Dict, Tuple, Set, Optional, Any
import torch
import os
import pickle
from pathlib import Path

def compute_token_transitions(tokenizer, fsm):
    """
    Compute the state transitions for each token in the tokenizer vocabulary and each state in the FSM.
    
    Args:
        tokenizer: An LLM tokenizer from Hugging Face Transformers
        fsm: A ByteFSM instance
        
    Returns:
        A dictionary mapping (state, token_id) -> next_state, indicating the state reached 
        when starting from 'state' and processing a token with ID 'token_id'.
    """
    # Initialize the transition map
    token_transitions = {}
    
    # Get all states in the FSM
    states = set()
    
    # Add states from transition dictionary keys
    states.update(fsm.transitions.keys())
    
    # Add states from transition dictionary values
    for transitions_dict in fsm.transitions.values():
        for byte, next_state in transitions_dict.items():
            if next_state is not None:
                states.add(next_state)
    
    # Add initial and final states
    states.add(fsm.initial)
    states.update(fsm.finals)
    
    # For each token ID and each state, compute the next state
    # vocab = tokenizer.get_vocab()
    from syncode.mask_store.byte_tokenizer import ByteTokenizer
    byte_tokenizer = ByteTokenizer(tokenizer)
    vocab = byte_tokenizer.byte_vocab
        
    for token_bytes, token_id in vocab.items():
        for state in states:
            # Start from the given state
            current = state
            
            # Process each byte of the token
            for byte in token_bytes:
                current = fsm.get_next_state(current, byte)
                if current is None:
                    break
            
            # Store the resulting state
            token_transitions[(state, token_id)] = current
    
    return token_transitions, states


def build_transition_matrix(transitions_by_state, max_tokens_transition): 
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


def traverse_token_path(tokens, token_transitions, initial_state):
    current_state = initial_state
    for token in tokens:
        current_state = token_transitions[(current_state, token)]
        try:
            assert current_state is not None, f'Invalid transition from {current_state} via {token}'
        except:
            import pdb; pdb.set_trace()
    return current_state



class DFAMatrixStore:
    def __init__(self, task_name, do_cot=False):
        self.task_name = task_name
        self.suffix = 'cot' if do_cot else 'std'
        
    def load(self):
        success = False
        Path('diffusion_cache').mkdir(parents=True, exist_ok=True)
        if os.path.exists(f'diffusion_cache/{self.task_name}_{self.suffix}.pkl'): 
            new_cls = pickle.load(open(f'diffusion_cache/{self.task_name}_{self.suffix}.pkl', 'rb'))
            self.initial_state = new_cls.initial_state
            self.final_states = new_cls.final_states
            self.transition_matrix = new_cls.transition_matrix
            success = True
        
        return success
    
    def build(self, tokenizer):
        regex_path = f'regex_folder/{self.task_name}_{self.suffix}.text'
        if os.path.exists(regex_path):
            with open(regex_path, 'r') as f:
                pattern = f.read()
        else:
            raise FileNotFoundError(f"Regex not found at {regex_path}")
        
        print(f"Creating ByteFSM for pattern: {pattern}")
        dfa = ByteFSM(pattern)
        
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
        
        #tocan we transition to different final states from a final state via eos/eot
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        for final_state in dfa.finals:
            for final_state_2 in dfa.finals:
                if eot_id is not None:
                    token_transitions[(final_state, eot_id)] = final_state_2
                if tokenizer.eos_token_id is not None:
                    token_transitions[(final_state, tokenizer.eos_token_id)] = final_state_2

        # Group transitions by starting state and next state
        transitions_by_state = {}
        for (state, token_id), next_state in token_transitions.items():
            if next_state is not None:  # Only consider valid transitions
                if state not in transitions_by_state:
                    transitions_by_state[state] = {}
                if next_state not in transitions_by_state[state]:
                    transitions_by_state[state][next_state] = []
                transitions_by_state[state][next_state].append(token_id)
        

        edge_src, edge_dst, edge_tok = [], [], []
        for src, dst_dict in transitions_by_state.items():
            for dst, tok_ids in dst_dict.items():
                edge_src.extend([src] * len(tok_ids))
                edge_dst.extend([dst] * len(tok_ids))
                edge_tok.extend(tok_ids)

        edge_src = torch.tensor(edge_src, dtype=torch.long)  # [E]
        edge_dst = torch.tensor(edge_dst, dtype=torch.long)  # [E]
        edge_tok = torch.tensor(edge_tok, dtype=torch.long)  # [E]

        # Optionally: sort by destination to improve scatter‑reduce locality
        dst_order = torch.argsort(edge_dst)
        edge_src = edge_src[dst_order]
        edge_dst = edge_dst[dst_order]
        edge_tok = edge_tok[dst_order]

        # ---------- save to the DFA store ---------------------------------
        self.initial_state = dfa.initial
        self.final_states  = sorted(list(dfa.finals))
        self.edge_src      = edge_src           # tensors live on CPU for now
        self.edge_dst      = edge_dst
        self.edge_tok      = edge_tok
        self.num_states    = len(states)
        self.token_transitions = token_transitions

    
    def save(self):
        with open(f'diffusion_cache/{self.task_name}_{self.suffix}.pkl', 'wb') as f:
            pickle.dump(self, f)
