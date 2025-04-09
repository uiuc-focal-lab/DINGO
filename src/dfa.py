import unittest
import time
import interegular
from syncode.mask_store.byte_fsm import ByteFSM
from transformers import AutoTokenizer
from typing import Dict, Tuple, Set, Optional, Any

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
    vocab = tokenizer.get_vocab()
    for token_text, token_id in vocab.items():
        # Convert token to bytes
        token_bytes = token_text.encode('utf-8')
        
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
