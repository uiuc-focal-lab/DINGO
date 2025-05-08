source /home/tarun/denv2/bin/activate

CUDA_VISIBLE_DEVICES=0 python main.py --dataset gsm_symbolic --model Dream-org/Dream-v0-Base-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --do_cot True --constrain_at 63 --start_idx 56 --end_idx 57







