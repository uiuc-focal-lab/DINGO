# DINGO: Constrained Inference for Diffusion LLMs
[![arXiv](https://img.shields.io/badge/arXiv-2502.09061-red.svg)](https://arxiv.org/abs/2505.23061)

Official Implementation of [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) published at the [NeurIPS 2025](https://neurips.cc/).

# DINGO Workflow
## 1) Input JSON Schema (regex)

```json
{
  "Country": "^[A-Z][a-z]+(?:\\s[A-Z][a-z]+)*$",
  "Population": "^[1-9][0-9]*$",
  "Capital": "^[A-Z][a-z]+(?:\\s[A-Z][a-z]+)*$"
}
```

---

## 2) Workflow Animation

<div align="center">
<svg width="760" height="180" viewBox="0 0 760 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pipeline animation">
  <defs>
    <marker id="arrowhead" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#08306b" />
    </marker>
  </defs>

  <rect x="18" y="25" width="170" height="60" rx="8" ry="8" fill="#3b82c4" stroke="#08306b" stroke-width="2"/>
  <text x="28" y="48" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#ffffff">Input regex</text>
  <text x="28" y="66" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#ffffff">{"Country": "^[A-Z]...$"}</text>

  <rect x="215" y="10" width="160" height="90" rx="8" ry="8" fill="#f1f5f9" stroke="#08306b" stroke-width="2"/>
  <text x="235" y="38" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="14" fill="#08306b" font-weight="700">DFA</text>
  <text x="235" y="60" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#08306b">compressed automaton</text>

  <rect x="410" y="25" width="160" height="60" rx="8" ry="8" fill="#3b82c4" stroke="#08306b" stroke-width="2"/>
  <text x="420" y="48" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#ffffff">Diffusion LM</text>
  <text x="420" y="66" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#ffffff">distribution</text>

  <rect x="600" y="25" width="140" height="60" rx="8" ry="8" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
  <text x="610" y="48" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#92400e">Sample top string</text>

  <path d="M188 55 L215 55" stroke="#08306b" stroke-width="3" fill="none" marker-end="url(#arrowhead)">
    <animate attributeName="opacity" values="0.25;1;0.25" dur="2s" repeatCount="indefinite" />
  </path>

  <path d="M375 55 L410 55" stroke="#08306b" stroke-width="3" fill="none" marker-end="url(#arrowhead)">
    <animate attributeName="opacity" values="0.25;1;0.25" dur="2s" begin="0.4s" repeatCount="indefinite" />
  </path>

  <path d="M570 55 L600 55" stroke="#08306b" stroke-width="3" fill="none" marker-end="url(#arrowhead)">
    <animate attributeName="opacity" values="0.25;1;0.25" dur="2s" begin="0.8s" repeatCount="indefinite" />
  </path>

  <text x="380" y="150" font-family="Segoe UI, Roboto, Arial, sans-serif" font-size="12" fill="#08306b" text-anchor="middle">Regex → DFA → LM → Sample</text>
</svg>
</div>

---

## 3) Example Output

```json
{
  "Country": "India",
  "Population": "1408069540",
  "Capital": "New Delhi"
}
```

---

## 4) Tokenized Construction (Reverse Animation)

<div align="center">
<svg width="760" height="120" viewBox="0 0 760 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tokenized construction animation">
  <text id="s1" x="40" y="60" font-family="Courier New, monospace" font-size="16" fill="#041529">}</text>
  <animate xlink:href="#s1" attributeName="opacity" values="1;0" dur="1s" begin="0s" fill="freeze"/>

  <text id="s2" x="40" y="60" font-family="Courier New, monospace" font-size="16" fill="#041529">lhi}</text>
  <animate xlink:href="#s2" attributeName="opacity" values="0;1;0" dur="1s" begin="1s" fill="freeze"/>

  <text id="s3" x="40" y="60" font-family="Courier New, monospace" font-size="16" fill="#041529">Delhi}</text>
  <animate xlink:href="#s3" attributeName="opacity" values="0;1;0" dur="1s" begin="2s" fill="freeze"/>

  <text id="s4" x="40" y="60" font-family="Courier New, monospace" font-size="16" fill="#041529">New Delhi}</text>
  <animate xlink:href="#s4" attributeName="opacity" values="0;1;0" dur="1s" begin="3s" fill="freeze"/>

  <text id="s5" x="40" y="60" font-family="Courier New, monospace" font-size="16" fill="#041529">"Capital": "New Delhi"}</text>
  <animate xlink:href="#s5" attributeName="opacity" values="0;1" dur="1s" begin="4s" fill="freeze"/>
</svg>
</div>



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
