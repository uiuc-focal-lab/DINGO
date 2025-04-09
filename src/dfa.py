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



class TestDatePatternTransitions(unittest.TestCase):
    def test_token_transitions(self):
        """Test compute_token_transitions with the Qwen2.5-1.5B tokenizer and a date pattern."""
        # Set up pattern for mm-dd-yyyy date format
        date_pattern = r"(0[1-9]|1[012])[-/\.](0[1-9]|[12][0-9]|3[01])[-/\.](19|20)\d\d"
        
        print("Loading tokenizer from 'Qwen/Qwen2.5-1.5B'...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        
        print(f"Creating ByteFSM for date pattern: {date_pattern}")
        dfa = ByteFSM(date_pattern)
        
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

if __name__ == "__main__":
    # Run the test directly
    test = TestDatePatternTransitions()
    test.test_token_transitions()
    print("\nTest completed successfully!")
