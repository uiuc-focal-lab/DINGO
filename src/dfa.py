import unittest
import time
import interegular
from syncode.mask_store.byte_fsm import ByteFSM
from syncode.mask_store.byte_tokenizer import ByteTokenizer
from transformers import AutoTokenizer
from typing import Dict, Tuple, Set, Optional, Any
import torch
import os
import pickle
from pathlib import Path
from grammar_folder.grammar_to_regex import lark_to_regex
import rust_dfa
from collections import defaultdict, deque
from typing import Any, Set, Dict
from syncode import SyncodeLogitsProcessor, Grammar

def get_live_states(self) -> Set[Any]:
    """
    Return the set of all states reachable from the initial state that can
    eventually reach a final state.
    """
    # 1) Build reverse-transition graph so we can walk backwards from finals
    rev: Dict[Any, Set[Any]] = defaultdict(set)
    for src, trans in self.transitions.items():
        for symbol, tgt in trans.items():
            rev[tgt].add(src)

    # 2) Find all states that can reach a final (backward BFS from finals)
    live_states: Set[Any] = set(self.finals)
    queue = deque(self.finals)
    while queue:
        s = queue.popleft()
        for prev in rev.get(s, ()):
            if prev not in live_states:
                live_states.add(prev)
                queue.append(prev)

    # 3) From initial, collect all reachable states (forward BFS)
    reachable: Set[Any] = set()
    queue = deque([self.initial])
    while queue:
        s = queue.popleft()
        if s in reachable:
            continue
        reachable.add(s)
        for tgt in self.transitions.get(s, {}).values():
            queue.append(tgt)

    # 4) Intersection: states both reachable from initial and live
    result = reachable & live_states

    # (Optional) update cache so future islive() calls are faster
    self._live_states_cache.update(result)

    return result

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



def build_augment_token_transitions(dfa, tokenizer, initial_state, final_states, token_transitions_path):
    if os.path.exists(token_transitions_path):
        with open(token_transitions_path, 'rb') as f:
            token_transitions = pickle.load(f)
        
        all_states = set()
        for (src_state, _), dst_state in token_transitions.items():
            all_states.add(src_state)
            if dst_state is not None:
                all_states.add(dst_state)
    else:
        byte_tokenizer = ByteTokenizer(tokenizer)
        vocab = byte_tokenizer.byte_vocab
        dfa.compute_token_transitions(vocab)
        token_transitions = dfa.get_all_token_transitions()
        

        # Add transitions for EOS and EOT tokens between final states
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        for final_state in final_states:
            if eot_id is not None:
                token_transitions[(final_state, eot_id)] = final_state
            if tokenizer.eos_token_id is not None:
                token_transitions[(final_state, tokenizer.eos_token_id)] = final_state
        
        # Collect all unique states from transitions
        all_states = set()
        for (src_state, _), dst_state in token_transitions.items():
            all_states.add(src_state)
            if dst_state is not None:
                all_states.add(dst_state)
        
    
        
        for state in all_states: 
            token_transitions[(state, 126336)] = state  
        
        with open(token_transitions_path, 'wb') as f:
            pickle.dump(token_transitions, f)
        
    return token_transitions, all_states


class ARLogitsProcessor:
    def __init__(self, task_name, do_cot, tokenizer, enable_oppurtunistic = False):
        self.task_name = task_name
        self.do_cot = do_cot
        self.tokenizer = tokenizer 
        self.suffix = 'cot' if do_cot else 'std'
        self.enable_oppurtunistic = enable_oppurtunistic

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = f'{current_file_dir}/grammar_folder/{self.task_name}_{self.suffix}.lark'
        if os.path.exists(grammar_path):
            with open(grammar_path, 'r') as f:
                grammar_string = f.read()
            pattern_dict = lark_to_regex(grammar_string)
            pattern = pattern_dict['start']
        
        else:
            raise FileNotFoundError(f"Grammar not found at {grammar_path}")
        
        self.dfa = rust_dfa.RegexDFA()
        self.dfa.initialize(pattern)
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        token_transitions_path = f'{current_file_dir}/token_transitions/{self.task_name}_{self.suffix}.pkl'
        Path(token_transitions_path).parent.mkdir(parents=True, exist_ok=True)
        initial_state = self.dfa.get_initial_state()
        final_states = self.dfa.get_final_states()
        self.token_transitions, all_states = build_augment_token_transitions(self.dfa, self.tokenizer, initial_state, final_states, token_transitions_path)
    
    def is_valid(self, input_ids, next_token_id):
        ids_to_traverse = input_ids[0].tolist() + next_token_id.tolist() if not isinstance(next_token_id, int) else input_ids[0].tolist() + [next_token_id]
        current_state = self.dfa.get_initial_state()
        for token in ids_to_traverse:
            current_state = self.token_transitions.get((current_state, token), None)
            if current_state is None:
                return False
        return True
    
    def __call__(self, input_ids, scores):
        #traverse each token in the vocabulary and check if the next token is valid and if is not then set the score to -inf
        for token_id in range(scores.shape[2]):
            if not self.is_valid(input_ids, token_id):
                scores[:, :, token_id] = -float('inf')
        return scores

    def validate(self, code, is_final_block = False):
        if self.enable_oppurtunistic:
            if is_final_block:
                return self.dfa.matches(code)
            else:
                return self.dfa.prefix_matches(code)
        else:
            return False
        

def load_ar_logits_processor(task_name, do_cot, tokenizer, enable_oppurtunistic = False):
    # suffix = 'cot' if do_cot else 'std'
    # #get current file directory
    # current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # grammar_path = f'{current_file_dir}/grammar_folder/{task_name}_{suffix}.lark'
    
    # if os.path.exists(grammar_path):
    #     with open(grammar_path, 'r') as f:
    #         grammar_string = f.read()
    # else:
    #     raise FileNotFoundError(f"Grammar not found at {grammar_path}")
    
    # pattern_dict = lark_to_regex(grammar_string)
    # pattern = pattern_dict['start']
    
    # grammar_string = f'start: /{pattern}/'
    
    # ar_logits_processor = SyncodeLogitsProcessor(Grammar(grammar_string), tokenizer, dev_mode = True)
    
    ar_logits_processor = ARLogitsProcessor(task_name, do_cot, tokenizer, enable_oppurtunistic)
    
    return ar_logits_processor
        

class DFAMatrixStore:
    def __init__(self, task_name, do_cot=False, enable_oppurtunistic=False):
        self.task_name = task_name
        self.suffix = 'cot' if do_cot else 'std'
        self.enable_oppurtunistic = enable_oppurtunistic
        
    
    def build(self, tokenizer):
        #get current file directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = f'{current_file_dir}/grammar_folder/{self.task_name}_{self.suffix}.lark'
        if os.path.exists(grammar_path):
            with open(grammar_path, 'r') as f:
                grammar_string = f.read()
            pattern_dict = lark_to_regex(grammar_string)
            pattern = pattern_dict['start']
        
        else:
            raise FileNotFoundError(f"Grammar not found at {grammar_path}")
        
        dfa = rust_dfa.RegexDFA()
        dfa.initialize(pattern)
        
        # Get initial state and final states using the new methods
        initial_state = dfa.get_initial_state()
        final_states = dfa.get_final_states()

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        token_transitions_path = f'{current_file_dir}/token_transitions/{self.task_name}_{self.suffix}.pkl'
        Path(token_transitions_path).parent.mkdir(parents=True, exist_ok=True) 
        token_transitions, all_states = build_augment_token_transitions(dfa, tokenizer, initial_state, final_states, token_transitions_path)
        
        # Save the original DFA for validation if enabled
        if self.enable_oppurtunistic:
            self.validator = dfa
        
        
        # Create a mapping from original state IDs to consecutive integers
        self.state_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_states))}
        self.debug_state_mapping = {new_id: old_id for new_id, old_id in enumerate(sorted(all_states))}
        
        # Remap all state IDs in the transitions dictionary
        remapped_token_transitions = {}
        for (state, token_id), next_state in token_transitions.items():
            new_state = self.state_mapping[state]
            new_next_state = self.state_mapping[next_state] if next_state is not None else None
            remapped_token_transitions[(new_state, token_id)] = new_next_state
        
        # Update token_transitions to use the remapped states
        token_transitions = remapped_token_transitions
        
        # Remap initial state and final states
        initial_state = self.state_mapping[initial_state]
        final_states = [self.state_mapping[fs] for fs in final_states]
        
        # Print mapping information
        print(f"Remappestate_mappingd {len(all_states)} states to consecutive IDs from 0 to {len(all_states) - 1}")
        num_states = len(all_states)

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
        self.initial_state = initial_state
        self.final_states  = sorted(final_states)
        self.edge_src      = edge_src           # tensors live on CPU for now
        self.edge_dst      = edge_dst
        self.edge_tok      = edge_tok
        self.num_states    = num_states
        self.token_transitions = token_transitions

    def validate(self, code, is_final_block = False):
        if self.enable_oppurtunistic:
            if is_final_block:
                return self.validator.matches(code)
            else:
                return self.validator.prefix_matches(code)
        else:
            return False

    def traverse_token_path(self, tokens, initial_state):
        current_state = initial_state
        for token in tokens:
            try:
                current_state = self.token_transitions[(current_state, token)]
            except:
                import pdb; pdb.set_trace()
        return current_state