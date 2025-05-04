import torch
from transformers import AutoModel, AutoTokenizer 
from .llada import LLADA
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dfa import DFAMatrixStore, load_ar_logits_processor

class BaseLM:
    def __init__(self, model_name, task_name, device, do_cot, device_map = None, constraint_mode = 'original', steps = 128, gen_length = 128,
                 block_length = 128, temperature = 0.0, cfg_scale = 0.0, remasking = 'low_confidence', constrain_at = 30, enable_oppurtunistic = False):
        
        if device_map is None: 
            self.model = AutoModel.from_pretrained(model_name, torch_dtype = torch.bfloat16, trust_remote_code = True, cache_dir = os.environ['HF_CACHE']).to(device)
        
        else:
            self.model = AutoModel.from_pretrained(model_name, device_map = device_map, torch_dtype = torch.bfloat16, trust_remote_code = True, cache_dir = os.environ['HF_CACHE'])
        
        self.model = self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, cache_dir = os.environ['HF_CACHE'])
        
        if 'dream' in model_name.lower():
            self.backend = 'DREAM' 
        elif 'llada' in model_name.lower():
            self.backend = 'LLADA'
        elif 'diffu_llama' in model_name.lower():
            self.backend = 'DiffuLlama'
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        dfa_store = None
        ar_logits_processor = None
        if constraint_mode == 'diffusion_constrained':
            dfa_store = DFAMatrixStore(task_name, do_cot, enable_oppurtunistic)
            dfa_store.build(self.tokenizer)
        elif constraint_mode == 'ar_constrained':
            ar_logits_processor = load_ar_logits_processor(task_name, do_cot, self.tokenizer, enable_oppurtunistic)
    
        self.generator = eval(f'{self.backend}')(self.model, self.tokenizer, constraint_mode, dfa_store, ar_logits_processor, steps, gen_length, block_length, temperature, cfg_scale, remasking, constrain_at)
    
    
    def __call__(self, batch):
        prompt = batch['prompt']
        
        if isinstance(prompt, list):
            prompt =  self.tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True)
        
        new_batch = batch
        start_time = time.time()
        out = self.generator(prompt)
        end_time = time.time()
        new_batch['llm_response'] = out
        new_batch['response_info'] = {'time': end_time - start_time}
        return new_batch