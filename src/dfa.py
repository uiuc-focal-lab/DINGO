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
from copy import deepcopy
import collections

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
            if idx == 1:
                pattern = re.compile(r'^\{"\/":\{"device":"\/[A-Za-z]{3}\/[A-Za-z]{3}[0-9]","mount_point":"\/","file_system_type":"[A-Za-z]{3}[0-9]","options":"[A-Za-z]{8}","dump":"[0-1]","pass":"[0-9]"\}\}$').pattern
            elif idx == 9:
                pattern = re.compile(
    r'^\{"patentId":"[A-Za-z]","applicationDate":"[0-9]{4}-[0-9]{2}-[0-9]{2}","inventorNames":\[\],"currentStatus":"[A-Za-z]"\}$'
).pattern
            elif idx == 32:
                pattern = re.compile(
    r'^\{"statementID":"[A-Za-z]","customerID":"[A-Za-z]","startDate":"[0-9]{4}-[0-9]{2}-[0-9]{2}","endDate":"[0-9]{4}-[0-9]{2}-[0-9]{2}","transactions":\[\]\}$'
).pattern
            elif idx == 59:
                pattern = re.compile(
    r'^\{"policyName":"[A-Za-z]","firewallRules":\[\],"vpnRequirements":\{"vpnType":"[A-Za-z]","encryptionStandards":"[A-Za-z]"\}\}$'
).pattern
            elif idx == 15:
                pattern = re.compile(
                    r'^(?:'
                    r'\{"deviceType": "smartphone", "brand": "[^"{}\n]+", "model": "[^"{}\n]+", "screenSize": "[^"{}\n]+"\}'
                    r'|'
                    r'\{"deviceType": "laptop", "brand": "[^"{}\n]+", "model": "[^"{}\n]+", "processor": "[^"{}\n]+", "RAMSize": "[^"{}\n]+"\}'
                    r')$'
                ).pattern
            elif idx == 16:
                pattern = re.compile(r'^\{\s*"feedbackID":\s*"[^"\\]+",\s*"customerID":\s*"[^"\\]+",\s*"serviceRating":\s*[1-5],\s*"comments":\s*"[^"\\]+",\s*"submissionDate":\s*"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"\s*\}$').pattern
            elif idx == 19:
                pattern = re.compile(r'^\{"totalShipments": [0-9]{3}, "onTimeDeliveryRate": [0-9]{2}\.[0-9], "averageDeliveryTime": [0-9]{2}\.[0-9], "pendingShipments": [0-9]{2}\}$').pattern
            elif idx == 21:
                s = r'"(?:[^"\\\n]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"'
                pattern = re.compile(
                    rf'^\{{\s*"customerID"\s*:\s*{s}\s*,\s*"feedbackScore"\s*:\s*[1-5]'
                    rf'(?:\s*,\s*"comments"\s*:\s*{s})?\s*\}}$'
                ).pattern
            elif idx == 24:
                pattern = re.compile(r'^\{"storage": "\/[A-Za-z]{3}\/[A-Za-z]{3}[0-9]", "fstype": "[A-Za-z]{3}[0-9]", "options": "[A-Za-z]{2},[A-Za-z]{7}", "readonly": false\}$').pattern
            elif idx == 34:
                pattern = re.compile(r'^\{\s*"reviewId"\s*:\s*"[^"{}\n]+",\s*"productId"\s*:\s*"[^"{}\n]+",\s*"reviewer"\s*:\s*\{\s*"userId"\s*:\s*"[^"{}\n]+",\s*"name"\s*:\s*"[^"{}\n]+"\s*\},\s*"rating"\s*:\s*[1-5],\s*"comments"\s*:\s*"[^"{}\n]+"\s*\}$').pattern
            elif idx == 37:
                pattern = re.compile(r'^\{"isMember": true, "membershipNumber": "[0-9]{10}"\}$').pattern
            elif idx == 45:
                pattern = re.compile(r'^\{\s*"cropType":\s*"[^"\n]+",\s*"harvestedArea":\s*\d+(?:\.\d+)?,\s*"yield":\s*\d+(?:\.\d+)?,\s*"productionDate":\s*"[^"\n]+"\s*\}$').pattern
            elif idx == 58:
                pattern = re.compile(
                    r'^\{\s*"distributorID":\s*"[^"\\{}\n]+",\s*'
                    r'"companyName":\s*"[^"\\{}\n]+",\s*'
                    r'"contactInfo":\s*\{\s*'
                    r'"phone":\s*"[^"\\{}\n]+",\s*'
                    r'"email":\s*"[^"\\{}\n]+",\s*'
                    r'"address":\s*"[^"\\{}\n]+"\s*'
                    r'\},\s*'
                    r'"productCategories":\s*\[\s*'
                    r'"[^"\\{}\n]+"(?:\s*,\s*"[^"\\{}\n]+")*\s*'
                    r'\]\s*\}$'
                ).pattern
            elif idx == 63:
                pattern = re.compile(
                    r'^\{\s*'
                    r'"feedbackID":\s*"[^"\\\n]+",\s*'
                    r'"customerID":\s*"[^"\\\n]+",\s*'
                    r'"productID":\s*"[^"\\\n]+",\s*'
                    r'"rating":\s*[1-5]'
                    r'(?:,\s*"comments":\s*"[^"\\\n]+")?'
                    r'\s*\}$'
                ).pattern
            elif idx == 39:
                pattern = re.compile(r'^\{ "foo" : true , "propertiesCount" : 20\}$').pattern
            elif idx == 19:
                pattern = re.compile(r'^\{\s*"totalShipments":\s*\d+,\s*"onTimeDeliveryRate":\s*\d+(?:\.\d+)?,\s*"averageDeliveryTime":\s*\d+(?:\.\d+)?,\s*"pendingShipments":\s*\d+\s*\}$').pattern
            elif idx == 27:
                pattern = re.compile(r'^\{"customerID":"[A-Za-z]","vehicleID":"[A-Za-z]","serviceRecords":\[\{"serviceDate":"[0-9]{4}-[0-9]{2}-[0-9]{2}","description":"[A-Za-z]","cost":[0-9]\}\],"totalSpent":[0-9]\}$').pattern
            elif idx == 28:
                pattern = re.compile(r'^\{"policyholder":\{"name":"[A-Za-z]","address":"[A-Za-z]","phone":"[A-Za-z]"\},"coverageDetails":\[\{"coverageType":"[A-Za-z]","coverageAmount":[0-9]\}\],"premium":\{"amount":[0-9],"frequency":"[A-Za-z]"\}\}$').pattern
            elif idx == 33:
                pattern = re.compile(r'^\{\s*"LanguageLearning":\s*\{\s*"target_language":\s*"[^"{}\n]+",\s*"current_proficiency":\s*"[^"{}\n]+",\s*"learning_goals":\s*\[\s*"[^"{}\n]+"\s*(?:,\s*"[^"{}\n]+"\s*)*\],\s*"recommended_resources":\s*\[\s*"[^"{}\n]+"\s*(?:,\s*"[^"{}\n]+"\s*)*\]\s*\}\s*\}$').pattern
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
                pattern = re.compile(r'^\{"diesel": \{"fuelType": "[A-Za-z]{6}", "quantityInStock": [0-9]{4}, "reorderLevel": [0-9]{4}\}, "gasoline": \{"fuelType": "[A-Za-z]{8}", "quantityInStock": [0-9]{4}, "reorderLevel": [0-9]{4}\}, "aviationFuel": \{"fuelType": "[A-Za-z]{8} [A-Za-z]{4}", "quantityInStock": [0-9]{4}, "reorderLevel": [0-9]{4}\}\}$').pattern
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
        
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        new_token_transition = deepcopy(token_transitions)
        for (state, token_id), next_state in token_transitions.items():
            if token_id in tokenizer.all_special_ids + [eot_id]:
                del new_token_transition[(state, token_id)]
        
        token_transitions = new_token_transition
        # Add transitions for EOS and EOT tokens between final states
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
        
        # For each state, store the tokens that can be used to transition to any valid next state
        self.state_to_valid_transitions = {}
        for state, transitions in transitions_by_state.items():
            # Create a boolean mask for the valid transitions   
            self.state_to_valid_transitions[state] = torch.zeros(len(tokenizer), dtype=torch.bool)
            for next_state, token_ids in transitions.items():
                self.state_to_valid_transitions[state][token_ids] = True

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
        # self.dst_to_final = self.precompute_back_dist(num_states, self.edge_src_nomdm, self.edge_dst_nomdm, final_states)
        
        Path(self.fpath).parent.mkdir(parents=True, exist_ok=True) 
        with open(self.fpath, 'wb') as f:
            dill.dump(self, f)
        
        self.dfa = dfa
        return self

    def precompute_back_dist(self, num_states      : int,
                            edge_src        : torch.Tensor,   # [E]
                            edge_dst        : torch.Tensor,   # [E]
                            final_states    : list[int]):
        """
        Returns
        -------
        dist        : 1-D LongTensor,  dist[q] = min #transitions from q → any final
                    (∞  ⇔  no path exists)
        buckets     : list[list[int]],  buckets[d] = states with dist == d
        """

        # ---- build reverse adjacency on CPU ----
        rev_adj = [[] for _ in range(num_states)]
        for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
            rev_adj[d].append(s)                    # edge d→s in the *reverse* graph

        # ---- multi-source BFS from all finals ----
        dist = [float('inf')] * num_states
        dq   = collections.deque()

        for f in final_states:
            dist[f] = 0
            dq.append(f)

        while dq:
            v = dq.popleft()
            for u in rev_adj[v]:
                if dist[u] == float('inf'):
                    dist[u] = dist[v] + 1
                    dq.append(u)

        return torch.tensor(dist, dtype=torch.long)

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
                return None, True
        return current_state, False

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


    def ar_logits_process(self, input_ids, scores):
        # Use self.state_to_valid_transitions to set the scores to -inf for invalid transitions
        current_state, some_error = self.traverse_token_path(input_ids[0].tolist(), self.initial_state)
        if some_error:
            scores[:, :, :] = -float('inf')
            return scores
            
        accept_mask = self.state_to_valid_transitions[current_state]
        
        if len(accept_mask) < scores.shape[2]:
            # pad the accept_mask with False
            accept_mask = torch.cat([accept_mask, torch.zeros(scores.shape[2] - len(accept_mask), dtype=torch.bool)])

        scores[:, :, ~accept_mask] = -float('inf')
        return scores


def build_for_dataset(dataset, schema_key, task_name, do_cot, tokenizer, device, enable_oppurtunistic = False, shift_mdm_trans_cuda = False):
    if schema_key is None:
        dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(None, None, tokenizer)
        if shift_mdm_trans_cuda:
            dfa_store.edge_src = dfa_store.edge_src.to(device)
            dfa_store.edge_dst = dfa_store.edge_dst.to(device)
            dfa_store.edge_tok = dfa_store.edge_tok.to(device)
        dfa_store.edge_src_nomdm = dfa_store.edge_src_nomdm.to(device)
        dfa_store.edge_dst_nomdm = dfa_store.edge_dst_nomdm.to(device)
        dfa_store.edge_tok_nomdm = dfa_store.edge_tok_nomdm.to(device)
        #dfa_store.dst_to_final = dfa_store.dst_to_final.to(device)
        return [dfa_store] * len(dataset)
    else:
        dfa_stores = []
        for row in tqdm(dataset, total=len(dataset), desc="Building DFA stores"):
            schema = row[schema_key]
            idx = row['idx']
            dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(schema, idx, tokenizer)
            dfa_stores.append(dfa_store)
        return dfa_stores