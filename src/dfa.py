from syncode.mask_store.byte_tokenizer import ByteTokenizer
import torch
import os
from pathlib import Path
from grammar_folder.grammar_to_regex import lark_to_regex
import rust_dfa
from outlines_core.fsm.json_schema import build_regex_from_schema
import hashlib
import sys
import dill


def sha256_digest(s: str) -> str:
    """Get the sha256 digest of a string

    Supports the `usedforsecurity` argument for Python 3.9+ to allow running on
    a FIPS-enabled system.
    """
    if sys.version_info >= (3, 9):
        return hashlib.sha256(s.encode('utf8'), usedforsecurity=False).hexdigest()
    else:
        return hashlib.sha256(s.encode('utf8')).hexdigest()
        

class DFAMatrixStore:
    def __init__(self, task_name, do_cot=False, enable_oppurtunistic=False):
        self.task_name = task_name
        self.suffix = 'cot' if do_cot else 'std'
        self.enable_oppurtunistic = enable_oppurtunistic
    
    def get_tok_name(self, tokenizer):
        if 'llada' in tokenizer.name_or_path.lower():
            return 'llada'
        elif 'dream' in tokenizer.name_or_path.lower():
            return 'dream'
        else:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")
    
    @classmethod
    def load_obj(cls, filename: str):
        """
        Load an instance of this class from a dill file.
        Returns:
            An object of type cls, reconstructed from the file.
        """
        with open(filename, 'rb') as f:
            obj = dill.load(f)
        # Optional: check that the loaded object is indeed an instance of cls
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
    
    def load_regex_dfa(self, schema, grammar_string):
        if schema is not None:
            pattern = build_regex_from_schema(schema)
        else:
            pattern_dict = lark_to_regex(grammar_string)
            pattern = pattern_dict['start']

            self.grammar_hash = sha256_digest(grammar_string)

        dfa = rust_dfa.RegexDFA()
        dfa.initialize(pattern)
        return dfa
    
    def load(self, schema, idx, tokenizer):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        if schema is not None:
            self.fpath = f'{current_file_dir}/dfa_store/{self.task_name}_schema{idx}_{self.get_tok_name(tokenizer)}.dill'
            if os.path.exists(self.fpath):
                old_obj = self.load_obj(self.fpath)
                old_obj.dfa = self.load_regex_dfa(schema, None)
                return old_obj, None
            else:
                return None, None

        else:
            self.fpath = f'{current_file_dir}/dfa_store/{self.task_name}_{self.suffix}_{self.get_tok_name(tokenizer)}.dill'
            grammar_path = f'{current_file_dir}/grammar_folder/{self.task_name}_{self.suffix}.lark'
            if os.path.exists(grammar_path):
                with open(grammar_path, 'r') as f:
                    grammar_string = f.read()
                
            else:
                raise FileNotFoundError(f"Grammar not found at {grammar_path}")

            if os.path.exists(self.fpath):
                old_obj = self.load_obj(self.fpath) 
                if old_obj.grammar_hash == sha256_digest(grammar_string): 
                    old_obj.dfa = self.load_regex_dfa(None, grammar_string)
                    return old_obj, grammar_string
                else:
                    os.remove(self.fpath)
                    return None, grammar_string
            else:
                return None, grammar_string
                
    def build_augment_token_transitions(self, dfa, tokenizer, final_states):
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
        
        if 'llada' in tokenizer.name_or_path.lower():
            for state in all_states: 
                token_transitions[(state, 126336)] = state  
        elif 'dream' in tokenizer.name_or_path.lower():
            for state in all_states: 
                token_transitions[(state, tokenizer.mask_token_id)] = state  

        return token_transitions, all_states
    
    def build(self, schema, idx, tokenizer):
        new_obj, grammar_string = self.load(schema, idx, tokenizer)
        if new_obj is not None:
            return new_obj
        
        dfa = self.load_regex_dfa(schema, grammar_string)
        
        
        # Get initial state and final states using the new methods
        initial_state = dfa.get_initial_state()
        final_states = dfa.get_final_states()

        token_transitions, all_states = self.build_augment_token_transitions(dfa, tokenizer, final_states)

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
        print(f"Remapped {len(all_states)} states to consecutive IDs from 0 to {len(all_states) - 1}")
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
        
        Path(self.fpath).parent.mkdir(parents=True, exist_ok=True) 
        with open(self.fpath, 'wb') as f:
            dill.dump(self, f)
        
        self.dfa = dfa
        return self

    def validate(self, code, is_final_block = False):
        if self.enable_oppurtunistic:
            if is_final_block:
                return self.dfa.matches(code)
            else:
                return self.dfa.prefix_matches(code)
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

    def is_valid_ar(self, input_ids, next_token_id):
        ids_to_traverse = input_ids[0].tolist() + next_token_id.tolist() if not isinstance(next_token_id, int) else input_ids[0].tolist() + [next_token_id]
        current_state = self.initial_state
        for token in ids_to_traverse:
            current_state = self.token_transitions.get((current_state, token), None)
            if current_state is None:
                return False
        return True

    def ar_logits_process(self, input_ids, scores):
        #traverse each token in the vocabulary and check if the next token is valid and if is not then set the score to -inf
        for token_id in range(scores.shape[2]):
            if not self.is_valid_ar(input_ids, token_id):
                scores[:, :, token_id] = -float('inf')
        return scores


def build_for_dataset(dataset, schema_key, task_name, do_cot, tokenizer, device, enable_oppurtunistic = False):
    if schema_key is None:
        dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(None, None, tokenizer)
        dfa_store.edge_src = dfa_store.edge_src.to(device)
        dfa_store.edge_dst = dfa_store.edge_dst.to(device)
        dfa_store.edge_tok = dfa_store.edge_tok.to(device)
        return [dfa_store] * len(dataset)
    else:
        dfa_stores = []
        for row in dataset:
            schema = row[schema_key]
            idx = row['idx']
            dfa_stores.append(DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(schema, idx, tokenizer))
        return dfa_stores