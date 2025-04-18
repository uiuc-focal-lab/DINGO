


CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True --constrain_at 127 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True --constrain_at 15

CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 32 --num_shots 4 --do_cot True --constrain_at 31

CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Instruct --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 64 --num_shots 4 --do_cot True --constrain_at 63



# CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True  --steps 128 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True 

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True --constrain_at 127

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 16 --num_shots 4 --do_cot True --constrain_at 15

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 32 --num_shots 4 --do_cot True --constrain_at 31

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model GSAI-ML/LLaDA-8B-Base --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 128 --gen_length 128 --block_length 64 --num_shots 4 --do_cot True --constrain_at 63







