---
title: SpectralGuard Demo
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: true
license: mit
tags:
  - security
  - ssm
  - mamba
  - spectral-analysis
  - adversarial-robustness
---

# 🔬 SpectralGuard — Interactive Spectral Safety Monitor

Runtime safety monitor for State Space Models (Mamba). Detects adversarial memory collapse attacks by tracking the spectral radius ρ(Ā_t) across all model layers in real time.

## Links

- 📄 [Paper](https://github.com/DaviBonetto/spectralguard/blob/main/paper/main2.pdf)
- 💻 [GitHub](https://github.com/DaviBonetto/spectralguard)
- 📊 [Dataset](https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset)

## How it works

1. **Input** — A text prompt is tokenized and fed into the Mamba SSM
2. **State Extraction** — SpectralGuard hooks into each layer's selective scan to capture Ā_t = exp(Δ_t · A)
3. **Spectral Computation** — Computes spectral radius ρ(Ā_t) via power iteration across all 24 layers
4. **Decision** — Flags prompts where ρ drops below the critical threshold (ρ\* = 0.90)

## Modes

| Mode         | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| `real_model` | Extracts actual hidden-state dynamics from Mamba-130M (requires GPU) |
| `demo_mode`  | Deterministic SHA-256-seeded fallback for demonstration purposes     |

## Citation

```bibtex
@article{bonetto2026spectralguard,
  title={SpectralGuard: Detecting Memory Collapse Attacks in State Space Models},
  author={Bonetto, Davi},
  year={2026},
  url={https://github.com/DaviBonetto/spectralguard}
}
```
