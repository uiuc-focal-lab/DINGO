source /home/tarun/denv2/bin/activate

#CUDA_VISIBLE_DEVICES=0 python main.py --dataset schema_bench --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode unconstrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4  --constrain_at 63 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset schema_bench --model Dream-org/Dream-v0-Instruct-7B --write_file True --overwrite_results True --constraint_mode diffusion_constrained --steps 64 --gen_length 128 --block_length 128 --num_shots 4 --constrain_at 63 --end_idx 20