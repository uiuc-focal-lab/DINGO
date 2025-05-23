#activate your environment first

CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True  --enable_oppurtunistic True 



CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 


CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset json_mode --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode ar_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True  --enable_oppurtunistic True 

