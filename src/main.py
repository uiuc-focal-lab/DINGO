
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from models import *
from evaluators import get_evaluator
from pathlib import Path
import atexit
import torch 
from datasets import Dataset
from prompter import BasePrompter
from dataset import load_dataset_by_name
from dfa import build_for_dataset
from transformers import AutoTokenizer

os.environ['HF_HOME'] = os.environ['HF_CACHE']



# Setting up argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Test LLMs on different datasets with various prompt styles.')
    
    #dataset args
    parser.add_argument('--dataset', type=str, help='Dataset name, e.g., "gsm8k" or "spider"')
    parser.add_argument('--start_idx', type=int, default=0, help='start index')
    parser.add_argument('--end_idx', type=int, default=-1, help='end index')
    parser.add_argument('--debug_ids', type=str, default=None, help='debug ids')
    
    #results args
    parser.add_argument("--log_dir", type=str, default="logging", help="Directory to save logs")
    parser.add_argument('--overwrite_results', type=bool, default=False, help='overwrite results file')
    parser.add_argument('--write_file', type=bool, default=False, help='save results in file')
    

    #CoT args
    parser.add_argument('--model', type=str, help='Model name, e.g., "gpt-3.5-turbo" or "gemini-1.0-pro"')
    parser.add_argument('--constraint_mode', type = str, default= 'unconstrained', help = 'Generation mode', choices = ['unconstrained', 'diffusion_constrained', 'ar_constrained'])
    parser.add_argument('--num_shots', type=int, default=8, help='number of few shots')
    parser.add_argument('--do_cot', type=bool, default=False, help='use COT')
    

    # general generation args
    parser.add_argument('--steps', type=int, default=128, help='number of steps for diffusion')
    parser.add_argument('--gen_length', type=int, default=128, help='max number of new tokens to generate')
    parser.add_argument('--block_length', type=int, default=128, help='block length for diffusion')
    parser.add_argument('--temperature', type=float, default=0.0, help='sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='cfg scale for diffusion')
    parser.add_argument('--remasking', type=str, default='low_confidence', help='remasking strategy')
    parser.add_argument('--constrain_at', type=int, default=None, help='constrain at step')
    parser.add_argument('--enable_oppurtunistic', type=bool, default=False, help='enable oppurtunistic constrained decoding')

    #Distributed inference args
    parser.add_argument('--enable_dist', type=bool, default=False, help='enable distributed inference')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_workers_per_gpu', type=int, default=1, help='number of workers per gpu')
    
    return parser.parse_args()

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def create_result_file(args):

    file_id = f"step={args.steps}_gen_length={args.gen_length}_block_length={args.block_length}_temperature={args.temperature}_cfg_scale={args.cfg_scale}_remasking={args.remasking}_constrain_at={args.constrain_at}_oppurtunistic={args.enable_oppurtunistic}"
    
    result_file = f"{args.log_dir}/{args.dataset}/{args.model.split('/')[-1]}/{args.constraint_mode}/cot={args.do_cot}/{args.num_shots}-shot/{file_id}.jsonl"
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    return result_file



def process_dataset_interactive(args):
    if args.constrain_at is None:
        args.constrain_at = (args.steps // (args.gen_length // args.block_length)) - 1
    
    dataset, schema_key = load_dataset_by_name(args.dataset)
    
    llm = BaseLM(args.model,args.dataset, 'cuda', args.do_cot, device_map = 'auto', constraint_mode = args.constraint_mode, steps = args.steps, gen_length = args.gen_length,
                 block_length = args.block_length, temperature = args.temperature, cfg_scale = args.cfg_scale, remasking = args.remasking, constrain_at = args.constrain_at, enable_oppurtunistic = args.enable_oppurtunistic, schema_key = schema_key)
    
    shift_mdm_trans_cuda = (args.constrain_at < (args.steps // (args.gen_length // args.block_length)) - 1)
    
    if args.end_idx == -1:
        args.end_idx = len(dataset)
    
    if args.debug_ids is not None: 
        debug_ids = [int(i) for i in args.debug_ids.split(',')]
        dataset = dataset.select(debug_ids)
    else:
        dataset = dataset.select(range(args.start_idx, args.end_idx))
    
    if args.constraint_mode in ['diffusion_constrained', 'ar_constrained']:
        dfa_stores = build_for_dataset(dataset, schema_key, args.dataset, args.do_cot, llm.tokenizer, llm.model.device, args.enable_oppurtunistic, shift_mdm_trans_cuda)
    else:
        dfa_stores = [None] * len(dataset)
    
    results = []
    processed_ids = set()
    result_file = create_result_file(args)
    print(result_file)
    if os.path.exists(result_file) and args.overwrite_results:
        os.remove(result_file)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])


    prompt_fn = BasePrompter(dataset = args.dataset, num_shots= args.num_shots, do_cot = args.do_cot)

    parse_fn = get_evaluator(args.dataset)(args.dataset, do_cot = args.do_cot)
    
    chat_mode = 'instruct' in args.model or 'Instruct' in args.model or 'it' in args.model or 'chat' in args.model
    
    results.clear()
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in processed_ids and not args.overwrite_results:
                pbar.update(1)
                continue

            prompt = prompt_fn.prompt(row, chat_mode=chat_mode)
            batch = {**row, 'prompt': prompt}

            new_batch = llm(batch, dfa_stores[idx])
            
            res = parse_fn.evaluate_answer(new_batch)
            
            if args.write_file:
                with open(result_file, 'a') as fout:
                    fout.write(json.dumps(res, default=set_default) + '\n')
            
            results.append(res['correct'])
            pbar.update(1)
            pbar.set_description(f"acc={np.mean(results):.4f}")
            

    accuracy = np.mean(results) if results else 0
    print(accuracy)

def process_dataset_parallel(args):
    import ray
    if not ray.is_initialized():
        ray.init()
    
    if args.constrain_at is None:
        args.constrain_at = (args.steps // (args.gen_length // args.block_length)) - 1

    
    shift_mdm_trans_cuda = (args.constrain_at < (args.steps // (args.gen_length // args.block_length)) - 1)
    dataset, schema_key =  load_dataset_by_name(args.dataset)
    if args.end_idx == -1:
        args.end_idx = len(dataset)
    
    if args.debug_ids is not None: 
        debug_ids = [int(i) for i in args.debug_ids.split(',')]
        dataset = dataset.select(debug_ids)
    else:
        dataset = dataset.select(range(args.start_idx, args.end_idx))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True, cache_dir = os.environ['HF_CACHE'])
    dfa_stores = build_for_dataset(dataset, schema_key, args.dataset, args.do_cot, tokenizer, llm.model.device, args.enable_oppurtunistic, shift_mdm_trans_cuda)
    
    results = []
    processed_ids = set()
    result_file = create_result_file(args)
    print(result_file)
    if os.path.exists(result_file) and args.overwrite_results:
        os.remove(result_file)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])


    chat_mode = 'instruct' in args.model or 'Instruct' in args.model or 'it' in args.model or 'chat' in args.model 
    
    @ray.remote(num_gpus=1/args.num_workers_per_gpu)
    class PromptActor:
        def __init__(self):
            self.prompt_fn = BasePrompter(dataset = args.dataset, num_shots= args.num_shots, do_cot = args.do_cot)
            self.llm = BaseLM(args.model, args.dataset, 'cuda', args.do_cot, device_map = None, constraint_mode = args.constraint_mode, gen_length = args.gen_length,
                 block_length = args.block_length, temperature = args.temperature, cfg_scale = args.cfg_scale, remasking = args.remasking, constrain_at = args.constrain_at, enable_oppurtunistic = args.enable_oppurtunistic, schema_key = schema_key)
        
        def process_item(self, batch):
            batch = {key: value[0] for key, value in batch.items()}
            batch['prompt'] = self.prompt_fn.prompt(batch, chat_mode=chat_mode)
    
            return [self.llm(batch)]
          
    ray_dataset = ray.data.from_huggingface(dataset)    
    
    actors = [PromptActor.remote() for _ in range(args.num_gpus * args.num_workers_per_gpu)]

    futures = []
    d_idx = 0
    for batch in ray_dataset.iter_batches(batch_size= 1):
        actor_id = len(futures) % len(actors)
        futures.append(actors[actor_id].process_item.remote(batch, dfa_stores[d_idx]))    
        d_idx += 1
    
    initial_responses = ray.get(futures)
    
    del actors
    del dfa_stores
    torch.cuda.empty_cache()

    new_dataset = []
    for initial_batched_response in initial_responses:
        new_dataset.extend(initial_batched_response)
    
    new_dataset = Dataset.from_list(new_dataset)
    
    ray_dataset = ray.data.from_huggingface(new_dataset) 

    
    @ray.remote(num_gpus=1/args.num_workers_per_gpu)
    class ParseActor:
        def __init__(self):
            self.parse_fn = get_evaluator(args.dataset)(args.dataset, do_cot = args.do_cot)
            
        
        def process_item(self, batch):
            batch = {key: value[0] for key, value in batch.items()}
            return [self.parse_fn.evaluate_answer(batch)]     
    
    actors = [ParseActor.remote() for _ in range(args.num_gpus * args.num_workers_per_gpu)]
    
    futures = []
    for batch in ray_dataset.iter_batches(batch_size= 1):
        actor_id = len(futures) % len(actors)
        futures.append(actors[actor_id].process_item.remote(batch))   
    
    results = ray.get(futures)

    if args.write_file:
        for result in results:
            for res in result:
                with open(result_file, 'a') as fout:
                    fout.write(json.dumps(res, default=set_default) + '\n')

def main():
    args = parse_arguments()
    if args.enable_dist:
        process_dataset_parallel(args)
    else:
        process_dataset_interactive(args)


if __name__ == "__main__":
    main()