---
license: mit
task_categories:
  - text-classification
tags:
  - security
  - adversarial
  - spectral-analysis
  - mamba
  - ssm
  - safety
size_categories:
  - 1K<n<10K
---

# SpectralGuard Benchmark Dataset

Paired benign and adversarial prompts with layer-wise spectral features for evaluating SSM safety monitors.

## Description

This dataset contains **1,200 prompts** processed through Mamba-130M with extracted spectral radius values across all 24 layers. Each sample includes ground-truth labels (benign/adversarial) and multi-layer spectral features.

| Split | Samples |
| ----- | ------- |
| Train | 840     |
| Val   | 180     |
| Test  | 180     |

**Class balance:** 600 benign · 600 adversarial (balanced 50/50)

## Schema

| Column        | Type  | Description                                       |
| ------------- | ----- | ------------------------------------------------- |
| `prompt_id`   | str   | Unique prompt identifier                          |
| `prompt_text` | str   | Raw prompt text                                   |
| `label`       | int   | 0 = benign, 1 = adversarial                       |
| `model_id`    | str   | Source model (e.g., `state-spaces/mamba-130m-hf`) |
| `layer_idx`   | int   | Layer index (-1 for prompt-level)                 |
| `token_idx`   | int   | Token index (-1 for prompt-level)                 |
| `rho`         | float | Spectral radius ρ                                 |
| `sigma_rho`   | float | Standard deviation of ρ                           |
| `split`       | str   | `train`, `val`, or `test`                         |
| `source`      | str   | Data source identifier                            |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("DaviBonetto/spectralguard-dataset")
print(ds)

# Access training split
train = ds["train"]
print(f"Training samples: {len(train)}")
```

Or load directly from CSV:

```python
import pandas as pd

df = pd.read_csv("spectralguard_benchmark.csv")
print(f"Total rows: {len(df)}")
print(f"Benign: {(df['label'] == 0).sum()}, Adversarial: {(df['label'] == 1).sum()}")
```

## Links

- 📄 [Paper](https://github.com/DaviBonetto/spectralguard/blob/main/paper/main2.pdf)
- 💻 [GitHub](https://github.com/DaviBonetto/spectralguard)
- 🔬 [Interactive Demo](https://huggingface.co/spaces/DaviBonetto/spectralguard-demo)

## Citation

```bibtex
@article{bonetto2026spectralguard,
  title={SpectralGuard: Detecting Memory Collapse Attacks in State Space Models},
  author={Bonetto, Davi},
  year={2026},
  url={https://github.com/DaviBonetto/spectralguard}
}
```
