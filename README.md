# DINGO: Constrained Inference for Diffusion LLMs

DINGO is a dynamic programming-based constrained decoding strategy for diffusion language models that enables efficient and provably distribution-preserving constraint enforcement. Unlike autoregressive models that generate tokens sequentially, diffusion LLMs predict blocks of tokens in parallel, making traditional constrained decoding algorithms ineffective. DINGO addresses this limitation by enabling sampling of output strings with the highest probability under the model's predicted distribution while strictly adhering to user-specified regular expressions and formal constraints such as fixed-schema JSON generation.

## Repository Structure

```
src/
├── main.py           # Main entry point and experiment runner
├── models.py         # Language model implementations
├── evaluators.py     # Dataset-specific evaluation functions
├── prompter.py       # Prompt formatting utilities
├── dataset.py        # Dataset loading and preprocessing
└── dfa.py           # Deterministic finite automaton for constraints
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion_constrain
```

2. Install dependencies:
```bash
pip install torch transformers datasets ray tqdm numpy
```

3. Set up environment variables:
```bash
export HF_CACHE=/path/to/huggingface/cache
export HF_HOME=$HF_CACHE
```

## Usage

### Basic Usage

```bash
python src/main.py \
  --dataset gsm8k \
  --model gpt-3.5-turbo \
  --constraint_mode unconstrained \
  --num_shots 8
```

### Diffusion-Constrained Generation

```bash
python src/main.py \
  --dataset spider \
  --model your-model \
  --constraint_mode diffusion_constrained \
  --steps 128 \
  --gen_length 128 \
  --block_length 128 \
  --constrain_at 5
```

### Distributed Inference

```bash
python src/main.py \
  --dataset gsm8k \
  --model your-model \
  --enable_dist true \
  --num_gpus 4 \
  --num_workers_per_gpu 2
```

## Key Parameters

- `--constraint_mode`: Generation mode (`unconstrained`, `diffusion_constrained`, `ar_constrained`)
- `--dataset`: Dataset name (e.g., `gsm8k`, `spider`)
- `--model`: Model name or path
- `--steps`: Number of diffusion steps
- `--gen_length`: Maximum tokens to generate
- `--do_cot`: Enable chain-of-thought reasoning
- `--enable_dist`: Enable distributed inference

## Output

Results are saved as JSONL files in the `logging/` directory, organized by dataset, model, constraint mode, and hyperparameters.

## License

[Add your license information here]
