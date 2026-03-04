# Hugging Face Publish Checklist

Use this checklist to publish the demo and dataset with stable URLs referenced by the paper and README.

## Space (`DaviBonetto/spectralguard-demo`)

1. Create a new Gradio Space named `spectralguard-demo`.
2. Upload:
   - `app.py`
   - `requirements-space.txt`
3. Set hardware to CPU or T4.
4. Confirm the app starts in `demo_mode` without manual edits.
5. Optional: enable `real_model` by adding model access tokens/weights.

Target URL:
- `https://huggingface.co/spaces/DaviBonetto/spectralguard-demo`

## Dataset (`DaviBonetto/spectralguard-benchmark`)

1. Create dataset repo named `spectralguard-benchmark`.
2. Upload:
   - `data/dataset/spectralguard_benchmark.csv`
   - `data/dataset/dataset_summary.csv`
   - `data/dataset/README.md`
3. Verify row count and schema in `dataset_summary.csv`.
4. Add short release notes describing synthetic ratio and source mix.

Target URL:
- `https://huggingface.co/datasets/DaviBonetto/spectralguard-benchmark`

## Post-publish sync

1. Update `docs/release/public_artifacts.md` status from `publish pending` to `live`.
2. If needed, update the artifact paragraph in `paper/main.tex`.
3. Commit and push.
