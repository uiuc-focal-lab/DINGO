import torch
from transformers import AutoModel, AutoTokenizer 
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def get_stop_word_phrase(task_name, model_name):
    if task_name == 'gsm_symbolic' and 'base' in model_name.lower():
        return '>>', 'The final answer', None
    else:
        return None, None, None

class BaseLM:
    def __init__(self, model_name, task_name, device, do_cot, device_map = None, constraint_mode = 'original', steps = 128, gen_length = 128,
                 block_length = 128, temperature = 0.0, cfg_scale = 0.0, remasking = 'low_confidence', constrain_at = 30, enable_oppurtunistic = False, schema_key = None):
        
        if device_map is None: 
            self.model = AutoModel.from_pretrained(model_name, torch_dtype = torch.bfloat16, trust_remote_code = True, cache_dir = os.environ['HF_CACHE']).to(device)
        
        else:
            self.model = AutoModel.from_pretrained(model_name, device_map = device_map, torch_dtype = torch.bfloat16, trust_remote_code = True, cache_dir = os.environ['HF_CACHE'])
        
        self.model = self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, cache_dir = os.environ['HF_CACHE'])
        
        if 'dream' in model_name.lower():
            from .dream import DREAM
            self.backend = 'DREAM' 
        elif 'llada' in model_name.lower():
            from .llada import LLADA
            self.backend = 'LLADA'
        elif 'diffu_llama' in model_name.lower():
            from .diffu_llama import DiffuLlama
            self.backend = 'DiffuLlama'
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        stop_word, stop_phrase, opp_end_char = get_stop_word_phrase(task_name, model_name)
        self.generator = eval(f'{self.backend}')(self.model, self.tokenizer, constraint_mode, steps, gen_length, block_length, temperature, cfg_scale, remasking, constrain_at, stop_word, stop_phrase, opp_end_char)

        self.task_name = task_name
        self.do_cot = do_cot
        self.enable_oppurtunistic = enable_oppurtunistic
        self.constraint_mode = constraint_mode
        self.schema_key = schema_key
    
    def __call__(self, batch, dfa_store):
        prompt = batch['prompt']
        
        schema = batch[self.schema_key] if self.schema_key is not None else None
        if schema is not None and dfa_store is not None:
            dfa_store.edge_src = dfa_store.edge_src.to(self.model.device)
            dfa_store.edge_dst = dfa_store.edge_dst.to(self.model.device)
            dfa_store.edge_tok = dfa_store.edge_tok.to(self.model.device)
        
        if isinstance(prompt, list):
            prompt =  self.tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True)
        
        new_batch = batch
        start_time = time.time()
        out = self.generator(prompt, dfa_store)
        end_time = time.time()
        
        if schema is not None and dfa_store is not None: 
            dfa_store.edge_src = dfa_store.edge_src.to('cpu')
            dfa_store.edge_dst = dfa_store.edge_dst.to('cpu')
            dfa_store.edge_tok = dfa_store.edge_tok.to('cpu')
        
        new_batch['llm_response'] = out
        new_batch['response_info'] = {'time': end_time - start_time}
        return new_batch