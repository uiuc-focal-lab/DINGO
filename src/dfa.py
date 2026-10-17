import collections
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import dill
import numpy as np
import torch
from tqdm import tqdm

import rust_dfa
from grammar_folder.grammar_to_regex import lark_to_regex
from tokenization import get_mask_token_id, tokenizer_preprocessing


def sha256_digest(s: str) -> str:
    """Get the sha256 digest of a string

    Supports the `usedforsecurity` argument for Python 3.9+ to allow running on
    a FIPS-enabled system.
    """
    if sys.version_info >= (3, 9):
        return hashlib.sha256(s.encode("utf8"), usedforsecurity=False).hexdigest()
    else:
        return hashlib.sha256(s.encode("utf8")).hexdigest()


class DFAMatrixStore:
    def __init__(self, task_name, do_cot=False, enable_oppurtunistic=False):
        self.task_name = task_name
        self.suffix = "cot" if do_cot else "std"
        self.enable_oppurtunistic = enable_oppurtunistic

    def get_tok_name(self, tokenizer):
        if "llada" in tokenizer.name_or_path.lower():
            return "llada"
        elif "dream" in tokenizer.name_or_path.lower():
            return "dream"
        else:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")

    @classmethod
    def load_obj(cls, filename: str):
        """
        Load an instance of this class from a dill file.
        Returns:
            An object of type cls, reconstructed from the file.
        """
        with open(filename, "rb") as f:
            obj = dill.load(f)
        # Optional: check that the loaded object is indeed an instance of cls
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def load_regex_dfa(self, schema, idx, grammar_string):
        if schema is not None:
            pattern = schema
        else:
            pattern_dict = lark_to_regex(grammar_string)
            pattern = pattern_dict["start"]

            self.grammar_hash = sha256_digest(grammar_string)

        dfa = rust_dfa.RegexDFA()
        dfa.initialize(pattern)
        return dfa

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if not k.startswith("_runtime_")}
        state.pop("dfa", None)  # PyO3 extension objects are not pickleable.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dfa = rust_dfa.RegexDFA()
        self.dfa.initialize(self.pattern)

    def load(self, schema, idx, tokenizer):
        for key in list(self.__dict__):
            if key.startswith("_runtime_"):
                del self.__dict__[key]
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_string = None
        if schema is None:
            grammar_path = f"{current_file_dir}/grammar_folder/{self.task_name}_{self.suffix}.lark"
            with open(grammar_path) as f:
                grammar_string = f.read()
            pattern = lark_to_regex(grammar_string)["start"]
        else:
            pattern = schema
        self.pattern = pattern
        self.mask_token_id = get_mask_token_id(tokenizer)
        preprocessing = tokenizer_preprocessing(tokenizer)
        self.vocab_bytes = preprocessing.vocab_bytes
        self._runtime_tokenizer_preprocessing = preprocessing
        # Content-addressed caches invalidate old mask/regex semantics as well
        # as schema edits and tokenizer revisions. Runtime options are not cached.
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [
                    6,
                    pattern,
                    preprocessing.fingerprint,
                ]
            ).encode()
        )
        self.fpath = f"{current_file_dir}/dfa_store/{self.task_name}_{digest.hexdigest()}.dill"
        if os.path.exists(self.fpath):
            obj = self.load_obj(self.fpath)
            obj.enable_oppurtunistic = self.enable_oppurtunistic
            return obj, grammar_string
        return None, grammar_string

    def build_augment_token_transitions(self, dfa, tokenizer, final_states):
        by_bytes = defaultdict(list)
        special_ids = set(tokenizer.all_special_ids)
        special_ids.add(self.mask_token_id)
        # convert_tokens_to_ids may return the ordinary UNK ID for an absent
        # EOT spelling; membership in the vocabulary is the authoritative test.
        eot_id = self._runtime_tokenizer_preprocessing.eot_id
        if eot_id is not None:
            special_ids.add(eot_id)
        if tokenizer.eos_token_id is not None:
            special_ids.add(tokenizer.eos_token_id)
        for token_id, token_bytes in self.vocab_bytes.items():
            if token_id not in special_ids:
                by_bytes[token_bytes].append(token_id)
        dfa.compute_token_transitions(dict(by_bytes))
        token_transitions = dfa.get_all_token_transitions()
        # Invisible termination/padding tokens are allowed only at acceptance.
        # No artificial mask self-loop: every mask edge must have a real-token
        # witness, as required by Section 4.1.
        for state in final_states:
            for token_id in (eot_id, tokenizer.eos_token_id):
                if token_id is not None and token_id != self.mask_token_id:
                    token_transitions[(state, token_id)] = state
        all_states = {dfa.get_initial_state(), *final_states}
        for (src, _), dst in token_transitions.items():
            all_states.update((src, dst))
        return token_transitions, all_states

    def build_sparse_edges(
        self, transitions_by_state: dict[int, dict[int, torch.Tensor]], mask_token_id: int
    ):
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
            list(zip(src, dst, tok, strict=True)),  # shape [E,3] on CPU
            dtype=torch.long,
        ).reshape(-1, 3)

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
        cached, grammar_string = self.load(schema, idx, tokenizer)
        if cached is not None:
            return cached
        dfa = self.load_regex_dfa(schema, idx, grammar_string)
        finals = dfa.get_final_states()
        transitions, states = self.build_augment_token_transitions(dfa, tokenizer, finals)
        self.state_mapping = {old: new for new, old in enumerate(sorted(states))}
        self.debug_state_mapping = {new: old for old, new in self.state_mapping.items()}
        self.initial_state = self.state_mapping[dfa.get_initial_state()]
        self.final_states = sorted(self.state_mapping[s] for s in finals)
        self.num_states = len(states)
        transitions = {
            (self.state_mapping[s], t): self.state_mapping[d] for (s, t), d in transitions.items()
        }
        # Character liveness is insufficient for a restricted vocabulary.
        # Compute co-accessibility over real token transitions (Section 4.1).
        src = torch.tensor([s for s, _ in transitions], dtype=torch.long)
        dst = torch.tensor(list(transitions.values()), dtype=torch.long)
        self.dst_to_final = self.precompute_back_dist(self.num_states, src, dst, self.final_states)
        self.live_states = (
            (self.dst_to_final < torch.iinfo(torch.long).max).nonzero().flatten().tolist()
        )
        live = set(self.live_states)
        if self.initial_state not in live:
            raise ValueError("Constraint has no tokenizable completion in this vocabulary")
        transitions = {(s, t): d for (s, t), d in transitions.items() if s in live and d in live}
        grouped = defaultdict(lambda: defaultdict(list))
        for (s, token), d in transitions.items():
            grouped[s][d].append(token)
        self.state_to_valid_transitions = {}
        for s, destinations in grouped.items():
            valid = {token for tokens in destinations.values() for token in tokens}
            valid.add(self.mask_token_id)
            self.state_to_valid_transitions[s] = torch.tensor(sorted(valid), dtype=torch.long)
        self.edge_src, self.edge_dst, self.edge_tok = self.build_sparse_edges(
            grouped, self.mask_token_id
        )
        real = self.edge_tok != self.mask_token_id
        self.edge_src_nomdm = self.edge_src[real]
        self.edge_dst_nomdm = self.edge_dst[real]
        self.edge_tok_nomdm = self.edge_tok[real]
        nfa = defaultdict(set)
        for (s, token), d in transitions.items():
            nfa[(s, token)].add(d)
            nfa[(s, self.mask_token_id)].add(d)
        self.token_transitions = {k: sorted(v) for k, v in nfa.items()}
        self.acceptance_can_pad = all(
            any(s == d for (q, _), d in transitions.items() if q == s) for s in self.final_states
        )
        self.decoded_mask_id = tokenizer.decode(self.mask_token_id)
        self.eos_token_id = tokenizer.eos_token_id
        del self.vocab_bytes  # only needed for preprocessing and the fingerprint
        del self._runtime_tokenizer_preprocessing
        self.dfa = dfa
        Path(self.fpath).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=Path(self.fpath).parent, delete=False) as f:
            temp_path = f.name
            dill.dump(self, f)
        os.replace(temp_path, self.fpath)
        return self

    def precompute_back_dist(
        self,
        num_states: int,
        edge_src: torch.Tensor,  # [E]
        edge_dst: torch.Tensor,  # [E]
        final_states: list[int],
    ):
        """
        Returns
        -------
        dist        : 1-D LongTensor,  dist[q] = min #transitions from q → any final
                    (∞  ⇔  no path exists)
        buckets     : list[list[int]],  buckets[d] = states with dist == d
        """

        # ---- build reverse adjacency on CPU ----
        rev_adj = [[] for _ in range(num_states)]
        for s, d in zip(edge_src.tolist(), edge_dst.tolist(), strict=True):
            rev_adj[d].append(s)  # edge d→s in the *reverse* graph

        # ---- multi-source BFS from all finals ----
        unreachable = torch.iinfo(torch.long).max
        dist = [unreachable] * num_states
        dq = collections.deque()

        for f in final_states:
            dist[f] = 0
            dq.append(f)

        while dq:
            v = dq.popleft()
            for u in rev_adj[v]:
                if dist[u] == unreachable:
                    dist[u] = dist[v] + 1
                    dq.append(u)

        return torch.tensor(dist, dtype=torch.long)

    def validate(self, code, is_final_block=False, last_step=False):
        # Text splitting cannot validate one-token substitutions. Intermediate
        # masked responses must use check_is_reachable on the actual token IDs.
        if not self.enable_oppurtunistic or not last_step:
            return False
        return self.dfa.matches(code) if is_final_block else self.dfa.prefix_matches(code)

    def traverse_token_path(self, tokens, initial_state):
        state = int(initial_state)
        for token in tokens:
            destinations = self.token_transitions.get((state, int(token)), [])
            if len(destinations) != 1:
                return None, True
            state = destinations[0]
        return state, False

    def reachable_states(self, tokens, initial_state=None):
        current = {self.initial_state if initial_state is None else int(initial_state)}
        for token in tokens:
            current = {d for s in current for d in self.token_transitions.get((s, int(token)), [])}
            if not current:
                break
        return current

    def check_is_reachable(self, tokens, is_final_block=False):
        targets = self.final_states if is_final_block else self.live_states
        return not self.reachable_states(tokens).isdisjoint(targets)

    def completion_targets(self, remaining_tokens, device=None):
        """Cached states with an accepting path of exactly this many real tokens.

        Returned masks are shared runtime tensors; callers must not mutate them.
        The default device follows dst_to_final, including stores moved to CUDA.
        """
        if remaining_tokens < 0:
            raise ValueError("remaining_tokens must be nonnegative")
        device = torch.device(self.dst_to_final.device if device is None else device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        runtime = self.__dict__.setdefault("_runtime_completion", {})
        cached = runtime.setdefault(device, {"masks": {}})
        masks = cached["masks"]
        if remaining_tokens in masks:
            return masks[remaining_tokens]
        # A real self-loop on each final state makes minimum distance enough.
        if self.acceptance_can_pad:
            if "distances" not in cached:
                cached["distances"] = self.dst_to_final.to(device)
            masks[remaining_tokens] = cached["distances"] <= remaining_tokens
        else:
            if "pairs" not in cached:
                pairs = {
                    (s, d)
                    for (s, token), destinations in self.token_transitions.items()
                    if token != self.mask_token_id
                    for d in destinations
                }
                cached["pairs"] = (
                    torch.tensor(sorted(pairs), dtype=torch.long, device=device).reshape(-1, 2).t()
                )
                masks[0] = torch.zeros(self.num_states, dtype=torch.bool, device=device)
                masks[0][self.final_states] = True
            src, dst = cached["pairs"]
            for length in range(max(masks) + 1, remaining_tokens + 1):
                targets = torch.zeros(self.num_states, dtype=torch.bool, device=device)
                targets[src[masks[length - 1][dst]]] = True
                masks[length] = targets
        return masks[remaining_tokens]

    def ar_logits_process(self, input_ids, scores):
        current = self.reachable_states(input_ids[0].tolist())
        accept = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        for state in current:
            valid = self.state_to_valid_transitions.get(state)
            if valid is not None:
                accept[valid.to(scores.device)] = True
        return scores.masked_fill(~accept, -float("inf"))


def build_for_dataset(
    dataset,
    schema_key,
    task_name,
    do_cot,
    tokenizer,
    device,
    enable_oppurtunistic=False,
    shift_mdm_trans_cuda=False,
):
    if schema_key is None:
        start_time = time.time()
        dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(
            None, None, tokenizer
        )
        end_time = time.time()
        print(f"Time taken to build DFA store: {end_time - start_time} seconds")
        if shift_mdm_trans_cuda:
            dfa_store.edge_src = dfa_store.edge_src.to(device)
            dfa_store.edge_dst = dfa_store.edge_dst.to(device)
            dfa_store.edge_tok = dfa_store.edge_tok.to(device)
        dfa_store.edge_src_nomdm = dfa_store.edge_src_nomdm.to(device)
        dfa_store.edge_dst_nomdm = dfa_store.edge_dst_nomdm.to(device)
        dfa_store.edge_tok_nomdm = dfa_store.edge_tok_nomdm.to(device)
        # dfa_store.dst_to_final = dfa_store.dst_to_final.to(device)
        return [dfa_store] * len(dataset)
    else:
        times = []
        dfa_stores = []
        for row in tqdm(dataset, total=len(dataset), desc="Building DFA stores"):
            schema = row[schema_key]
            idx = row["idx"]
            start_time = time.time()
            dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic).build(
                schema, idx, tokenizer
            )
            end_time = time.time()
            times.append(end_time - start_time)
            dfa_stores.append(dfa_store)
        print(f"Time taken to build DFA stores: {np.mean(times)} seconds")
        return dfa_stores
