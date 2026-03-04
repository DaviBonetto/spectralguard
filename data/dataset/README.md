# SpectralGuard Benchmark Dataset

Public benchmark schema for spectral defense research.

## Output files

- `spectralguard_benchmark.csv`
- `dataset_summary.csv`

## Required schema

- `prompt_id`
- `prompt_text`
- `label` (`0` benign, `1` adversarial)
- `model_id`
- `layer_idx`
- `token_idx`
- `rho`
- `sigma_rho`
- `split` (`train`, `val`, `test`)
- `source`

## Build command (Task 23 target)

```bash
python mamba_spectral/scripts/compile_dataset.py \
  --output dataset/spectralguard_benchmark.csv \
  --min-prompts 1200 \
  --target-benign 600 \
  --target-adversarial 600 \
  --allow-synthetic-fill
```

## Validation policy

- Build fails if required columns are missing.
- Build fails if either class label is absent.
- Summary is emitted to `dataset_summary.csv` and must report:
  - `rows >= 1000` (target default is 1200),
  - both labels present,
  - non-empty train/val/test splits.
  - composition transparency fields (`synthetic_rows`, `real_rows`, `synthetic_ratio`, `unique_prompts`, `unique_prompt_ratio`).

## Notes

- The compiler ingests prompt-level artifacts when full traces are unavailable.
- When real artifacts are insufficient, synthetic fill can be enabled to hit release-size targets while preserving schema compatibility.
- Release notes should always report synthetic composition from `dataset_summary.csv`.
