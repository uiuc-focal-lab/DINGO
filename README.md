# DINGO: Constrained Inference for Diffusion LLMs

DINGO is a dynamic programming-based constrained decoding strategy for diffusion language models that enables efficient and provably distribution-preserving constraint enforcement. Unlike autoregressive models that generate tokens sequentially, diffusion LLMs predict blocks of tokens in parallel, making traditional constrained decoding algorithms ineffective. DINGO addresses this limitation by enabling sampling of output strings with the highest probability under the model's predicted distribution while strictly adhering to user-specified regular expressions and formal constraints such as fixed-schema JSON generation.



## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion_constrain
```

2. Install dependencies:
Setup 2 virtual environments or conda environments. In each environment, after activating it, install:
```bash
pip install -r requirements.txt
```
Then, for one environment (for DREAM model family),
```bash
pip install transformers==4.46.2
```

In the other environment (for LLaDA model family),
```bash
pip install transformers==4.38.2
```
These transformer versions are those specified in the official model release for each respective model family.
Then, for each environment,
```bash
cd diffusion_constrain/src/rust_dfa
maturin develop
```

3. Set up environment variables:
```bash
export HF_CACHE=/path/to/huggingface/cache
export HF_HOME=$HF_CACHE
```

## Usage
### Reproducing main experiments
Activate the respective environment for the model family.
```bash
cd diffusion_constrain/src/
bash run_exp.sh
```
Refer to `diffusion_constrain/src/main.py` for additional parameters.



Results are saved as JSONL files in the `logging/` directory, organized by dataset, model, constraint mode, and hyperparameters.

## License

[Add your license information here]
