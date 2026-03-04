# Public Artifacts and Links

This document is the canonical source for public links shared with external reviewers.

## Core links

- GitHub repository (live): `https://github.com/DaviBonetto/spectralguard`
- Paper PDF in repo: `https://github.com/DaviBonetto/spectralguard/blob/main/paper/main.pdf`
- Hugging Face Space (target): `https://huggingface.co/spaces/DaviBonetto/spectralguard-demo`
- Hugging Face Dataset (target): `https://huggingface.co/datasets/DaviBonetto/spectralguard-benchmark`

## Status

- GitHub: ready
- Space: deployment-ready code (`app.py` + `requirements-space.txt`), publish pending
- Dataset: schema + summary ready (`data/dataset/*`), publish pending
- Zamba2 transfer bundle: runner ready, execution pending on GPU runtime (T4/A100)

## Evidence pointers

- API contract: `spectralguard.monitor(prompt, hidden_states)`
  - `spectralguard/detector.py`
  - `spectralguard/__init__.py`
  - `mamba_spectral/tests/test_spectralguard_api.py`
- Demo implementation:
  - `app.py`
  - `requirements-space.txt`
- Dataset release files:
  - `data/dataset/spectralguard_benchmark.csv`
  - `data/dataset/dataset_summary.csv`
  - `data/dataset/README.md`
- Zamba2 transfer outputs (created after GPU run):
  - `artifacts/zamba2_v4_transfer/zamba2_transfer_metrics.csv`
  - `artifacts/zamba2_v4_transfer/zamba2_transfer_summary.csv`
  - `artifacts/zamba2_v4_transfer/zamba2_transfer_samples.csv`

## Reviewer note

If the two Hugging Face links are still private or not yet created at review time, keep this file and add the final public URLs in-place without changing filenames elsewhere in the repo.
