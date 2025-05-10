from syncode.mask_store.byte_tokenizer import ByteTokenizer
import torch
import os
from pathlib import Path
from grammar_folder.grammar_to_regex import lark_to_regex
import rust_dfa
import hashlib
import sys
import dill
from collections import defaultdict
from tqdm import tqdm
from outlines_core.json_schema import build_regex_from_schema
import re

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
    
    def load_regex_dfa(self, schema, idx, grammar_string):
        if schema is not None:
            if idx == 15:
                pattern = re.compile(
                    r'^(?:'
                    r'\{"deviceType": "smartphone", "brand": "[^"{}\n]+", "model": "[^"{}\n]+", "screenSize": "[^"{}\n]+"\}'
                    r'|'
                    r'\{"deviceType": "laptop", "brand": "[^"{}\n]+", "model": "[^"{}\n]+", "processor": "[^"{}\n]+", "RAMSize": "[^"{}\n]+"\}'
                    r')$'
                ).pattern
            elif idx == 19:
                pattern = re.compile(r'^\{\s*"totalShipments":\s*\d+,\s*"onTimeDeliveryRate":\s*\d+(?:\.\d+)?,\s*"averageDeliveryTime":\s*\d+(?:\.\d+)?,\s*"pendingShipments":\s*\d+\s*\}$').pattern
            elif idx == 27:
                pattern = re.compile(
                    r'^\{\s*'
                    r'"customerID":\s*"[^"\n]+",\s*'
                    r'"vehicleID":\s*"[^"\n]+",\s*'
                    r'"serviceRecords":\s*\[\s*'
                    r'\{\s*'
                        r'"serviceDate":\s*"\d{4}-\d{2}-\d{2}",\s*'
                        r'"description":\s*"[^"\n]+",\s*'
                        r'"cost":\s*\d+(?:\.\d+)?\s*'
                    r'\}'
                    r'(?:\s*,\s*'
                        r'\{\s*'
                        r'"serviceDate":\s*"\d{4}-\d{2}-\d{2}",\s*'
                        r'"description":\s*"[^"\n]+",\s*'
                        r'"cost":\s*\d+(?:\.\d+)?\s*'
                        r'\}'
                    r')*'
                    r'\s*\],\s*'
                                    r'"totalSpent":\s*\d+(?:\.\d+)?\s*'
                                    r'\}$'
                                ).pattern
            elif idx == 33:
                pattern =re.compile(r'^\{\s*"LanguageLearning":\s*\{\s*"target_language":\s*"[^"{}\n]+",\s*"current_proficiency":\s*"[^"{}\n]+",\s*"learning_goals":\s*\[\s*"[^"{}\n]+"\s*(?:,\s*"[^"{}\n]+"\s*)*\],\s*"recommended_resources":\s*\[\s*"[^"{}\n]+"\s*(?:,\s*"[^"{}\n]+"\s*)*\]\s*\}\s*\}$').pattern
            elif idx == 45:
                pattern = re.compile(r'^\{\s*"cropType":\s*"[^"]+",\s*"harvestedArea":\s*\d+(?:\.\d+)?,\s*"yield":\s*\d+(?:\.\d+)?,\s*"productionDate":\s*"[^"]+"\s*\}$').pattern
            elif idx == 72: 
                pattern = re.compile(
    r'^\{\s*'
    r'"current_location":\s*"(?:\\.|[^"\\\n])+",\s*'
    r'"preferred_units":\s*"(?:\\.|[^"\\\n])+",\s*'
    r'"weather_forecast":\s*"(?:\\.|[^"\\\n])+",\s*'
                    r'"recommended_activities":\s*\[\s*"(?:\\.|[^"\\\n])+"(?:\s*,\s*"(?:\\.|[^"\\\n])+")*\s*\]\s*'
                    r'\}$'
                ).pattern
            elif idx == 97:
                pattern = re.compile(
    r'^\{"[^"{}\n]+": \{"fuelType": "[^"{}\n]+", "quantityInStock": \d+(?:\.\d+)?, "reorderLevel": \d+(?:\.\d+)?\}'
    r'(?:, "[^"{}\n]+": \{"fuelType": "[^"{}\n]+", "quantityInStock": \d+(?:\.\d+)?, "reorderLevel": \d+(?:\.\d+)?\}){2}\}$'
).pattern
            else:
                pattern =  build_regex_from_schema(schema)
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
                old_obj.dfa = self.load_regex_dfa(schema, idx, None)
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
                    old_obj.dfa = self.load_regex_dfa(None, None, grammar_string)
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

        if 'llada' in tokenizer.name_or_path.lower():
            for state in all_states: 
                token_transitions[(state, 126336)] = state  
        elif 'dream' in tokenizer.name_or_path.lower():
            for state in all_states: 
                token_transitions[(state, tokenizer.mask_token_id)] = state  

        return token_transitions, all_states

    def build_sparse_edges(self, transitions_by_state: dict[int, dict[int, torch.Tensor]],
                        mask_token_id: int):
        """
        Parameters
        ----------
        transitions_by_state
            dict[src_state][dst_state] = 1-D LongTensor of *token IDs*
            (no mask-token edges yet, negative numbers never appear).
        mask_token_id
            The single *[MASK]* token ID in your vocabulary.
        Returns
        -------
        edge_src, edge_dst, edge_tok  (each 1-D LongTensor )
        """

        src, dst, tok = [], [], []

        for q_src, dests in transitions_by_state.items():
            for q_dst, tokens in dests.items():
                # add all *real* token edges
                n = len(tokens)
                src.extend([q_src] * n)
                dst.extend([q_dst] * n)
                tok.extend(tokens)

                # add exactly *one* mask-edge for this (src,dst) pair
                src.append(q_src)
                dst.append(q_dst)
                tok.append(mask_token_id)

        edge_triplets = torch.tensor(
            list(zip(src, dst, tok)),          # shape [E,3] on CPU
            dtype=torch.long
        )

        # ---------- deduplicate triplets (cost O(E log E)) ----------
        edge_triplets = torch.unique(edge_triplets, dim=0)

        # unpack
        edge_src, edge_dst, edge_tok = edge_triplets.t().contiguous()
        dst_order = torch.argsort(edge_dst)
        edge_src = edge_src[dst_order]
        edge_dst = edge_dst[dst_order]
        edge_tok = edge_tok[dst_order]

        return edge_src, edge_dst, edge_tok
    
    def build(self, schema, idx, tokenizer):
        new_obj, grammar_string = self.load(schema, idx, tokenizer)
        if new_obj is not None:
            return new_obj
        
        dfa = self.load_regex_dfa(schema, idx, grammar_string)
        
        
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
        
        if 'llada' in tokenizer.name_or_path.lower():
            mask_id = 126336
        elif 'dream' in tokenizer.name_or_path.lower():
            mask_id = tokenizer.mask_token_id
        else:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")

        edge_src, edge_dst, edge_tok = self.build_sparse_edges(transitions_by_state, mask_id)
        edge_is_mdm = (edge_tok == mask_id) 

        # convert token transitions to NFA due to mask_id  
        new_token_transitions = defaultdict(list)
        for (src_state, token_id), dst_state in token_transitions.items():
            new_token_transitions[(src_state, token_id)].append(dst_state)
            new_token_transitions[(src_state, mask_id)].append(dst_state)
            
        token_transitions = dict(new_token_transitions)
        # ---------- save to the DFA store ---------------------------------
        self.initial_state = initial_state
        self.final_states  = sorted(final_states)
        self.edge_src      = edge_src           # tensors live on CPU for now
        self.edge_dst      = edge_dst
        self.edge_tok      = edge_tok
        self.edge_src_nomdm = self.edge_src[~edge_is_mdm]
        self.edge_dst_nomdm = self.edge_dst[~edge_is_mdm]
        self.edge_tok_nomdm = self.edge_tok[~edge_is_mdm]
        self.num_states    = num_states
        self.token_transitions = token_transitions
        self.decoded_mask_id = tokenizer.decode(mask_id)
        self.eos_token_id = tokenizer.eos_token_id
        
        Path(self.fpath).parent.mkdir(parents=True, exist_ok=True) 
        with open(self.fpath, 'wb') as f:
            dill.dump(self, f)
        
        self.dfa = dfa
        return self

    def validate(self, code, is_final_block = False, last_step = False):
        if self.enable_oppurtunistic:
            if last_step:
                if is_final_block:
                    return self.dfa.matches(code)
                else:
                    return self.dfa.prefix_matches(code)
            else:
                prefixes = code.split(self.decoded_mask_id)
        else:
            return False

    def traverse_token_path(self, tokens, initial_state):
        current_state = initial_state
        for token in tokens:
            try:
                assert len(self.token_transitions[(current_state, token)]) == 1
                current_state = self.token_transitions[(current_state, token)][0]
            except:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
                import pdb; pdb.set_trace()
        return current_state

    def check_is_reachable(self, tokens, is_final_block = False):
        """
        Return True iff *at least one* path produced by `tokens` can take us
        from `initial_state` to any state in `final_states`.

        self.token_transitions is expected to map (state, token) -> list[int]
        (an empty list or a missing key = no outgoing edges for that pair).
        """
        final_states = self.final_states if is_final_block else None
        # start with the single initial state
        current = {self.initial_state}

        for token in tokens:
            next_states = set()

            for state in current:
                for nxt in self.token_transitions.get((state, token), []):
                    next_states.add(nxt)

            if not next_states:
                return False

            current = next_states  
            
        if final_states is not None:
            return not current.isdisjoint(final_states)
        else:
            return current is not None

    def is_valid_ar(self, input_ids, next_token_id):
        ids_to_traverse = input_ids[0].tolist() + next_token_id.tolist() if not isinstance(next_token_id, int) else input_ids[0].tolist() + [next_token_id]
        return self.check_is_reachable(ids_to_traverse)

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
        for row in tqdm(dataset, total=len(dataset), desc="Building DFA stores"):
            schema = row[schema_key]
            idx = row['idx']
            try:
                dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(schema, idx, tokenizer)
                dfa_stores.append(dfa_store)
            except Exception as e:
                print(f"Error building DFA store for row {idx}: {e}")
                dfa_stores.append(None)
        return dfa_stores