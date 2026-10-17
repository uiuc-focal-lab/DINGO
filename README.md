# DINGO: Constrained Inference for Diffusion LLMs
[![arXiv](https://img.shields.io/badge/arXiv-2505.23061-red.svg)](https://arxiv.org/abs/2505.23061)

Official Implementation of [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) published at the [NeurIPS 2025](https://neurips.cc/).

DINGO is a dynamic programming-based constrained decoding strategy for diffusion language models that enables efficient and provably distribution-preserving constraint enforcement. Unlike autoregressive models that generate tokens sequentially, diffusion LLMs predict blocks of tokens in parallel, making traditional constrained decoding algorithms ineffective. DINGO addresses this limitation by enabling sampling of output strings with the highest probability under the model's predicted distribution while strictly adhering to user-specified regular expressions and formal constraints such as fixed-schema JSON generation.

## DINGO DEMO 

This is the visualization of the DINGO workflow, which seamlessly handles any regular expression, including those used in JSON Schema definitions. DINGO is optimal and always draws the highest probability string that is structurally valid.

> ![DINGO Workflow](./DINGODemo.gif)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/uiuc-focal-lab/DINGO.git
cd DINGO
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
cd src/rust_dfa
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
cd src/
bash run_exp.sh
```
Refer to `src/main.py` for additional parameters.

Results are saved as JSONL files in the `logging/` directory, organized by dataset, model, constraint mode, and hyperparameters.

## Citation

```bibtex
@misc{suresh2025dingoconstrainedinferencediffusion,
      title={DINGO: Constrained Inference for Diffusion LLMs}, 
      author={Tarun Suresh and Debangshu Banerjee and Shubham Ugare and Sasa Misailovic and Gagandeep Singh},
      year={2025},
      eprint={2505.23061},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23061}, 
}
```

## License

This project is licensed under the [MIT License](LICENSE).
