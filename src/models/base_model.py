import os
import time

import torch
from transformers import AutoModel, AutoTokenizer

from .constrained_decode import release_decode_cache, reset_decode_validation


def get_stop_word_phrase(task_name, model_name):
    if task_name == "gsm_symbolic" and "base" in model_name.lower():
        return ">>", "The final answer"
    else:
        return None, None


class BaseLM:
    def __init__(
        self,
        model_name,
        task_name,
        device,
        do_cot,
        device_map=None,
        constraint_mode="unconstrained",
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        constrain_at=0,
        enable_oppurtunistic=False,
        schema_key=None,
        require_complete=True,
    ):

        if device_map is None:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_CACHE", os.environ.get("HF_HOME")),
            ).to(device)

        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_CACHE", os.environ.get("HF_HOME")),
            )

        self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.environ.get("HF_CACHE", os.environ.get("HF_HOME")),
        )

        if "dream" in model_name.lower():
            from .dream import DREAM

            self.backend = "DREAM"
            generator_class = DREAM
        elif "llada" in model_name.lower():
            from .llada import LLADA

            self.backend = "LLADA"
            generator_class = LLADA
        else:
            raise ValueError(f"Model {model_name} not supported")

        stop_word, stop_phrase = get_stop_word_phrase(task_name, model_name)
        self.generator = generator_class(
            self.model,
            self.tokenizer,
            constraint_mode,
            steps,
            gen_length,
            block_length,
            temperature,
            cfg_scale,
            remasking,
            constrain_at,
            stop_word,
            stop_phrase,
            require_complete=require_complete,
        )

        self.task_name = task_name
        self.do_cot = do_cot
        self.enable_oppurtunistic = enable_oppurtunistic
        self.constraint_mode = constraint_mode
        self.schema_key = schema_key

    def __call__(self, batch, dfa_store=None):
        prompt = batch["prompt"]
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        previous = getattr(self, "_active_decode_store", None)
        if previous is not None and previous is not dfa_store:
            release_decode_cache(previous)
            for name in ("edge_src", "edge_dst", "edge_tok", "dst_to_final"):
                setattr(previous, name, getattr(previous, name).cpu())
        self._active_decode_store = dfa_store
        if dfa_store is not None:
            for name in ("edge_src", "edge_dst", "edge_tok", "dst_to_final"):
                setattr(dfa_store, name, getattr(dfa_store, name).to(self.model.device))
        try:
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            start = time.perf_counter()
            out = self.generator(prompt, dfa_store)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            elapsed = time.perf_counter() - start
        finally:
            # Retain only the current automaton to amortize capture. Switching
            # schemas evicts its device cache before loading the next one.
            if dfa_store is not None:
                reset_decode_validation(dfa_store)
        return {**batch, "llm_response": out, "response_info": {"time": elapsed}}
