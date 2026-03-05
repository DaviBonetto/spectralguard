<div align="center">

# SpectralGuard

**Spectral anomaly detection for Mamba-based language models**

_Detecting adversarial prompt injections via hidden-state eigenvalue analysis_

---

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-NeurIPS%202026-red?logo=arxiv)](paper/paper.pdf)
[![HuggingFace](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/DaviBonetto/spectralguard-demo)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Paper**](paper/paper.pdf) · [**Quickstart**](#quickstart) · [**Experiments**](#experiment-matrix) · [**API Reference**](#public-api) · [**Reproducibility**](#reproducibility)

<img src="assets/diagram.png" alt="SpectralGuard Diagram" width="80%"/>

</div>

---

## What is SpectralGuard?

State Space Models (SSMs) like Mamba achieve linear-time sequence processing through input-dependent recurrence, but this mechanism introduces a critical vulnerability: **Spectral Collapse**.

SpectralGuard is a **runtime safety monitor** for recurrent and hybrid foundation models that tracks the spectral radius $\rho(\bar{A}_t)$ of the discretized transition operator across layers. It protects against attacks that silently destroy reasoning capacity without triggering output-level alarms.

### Theoretical Foundation

1. **The Spectral Horizon Bound:** We prove that effective memory horizon ($H_{\text{eff}}$) is tightly bounded by the spectral radius $\rho(\bar{A})$. When $\rho$ drops below a critical threshold ($\approx 0.90$), reasoning capacity collapses from millions of tokens to mere dozens.
2. **Evasion Existence Theorem:** We formally establish that **no defense operating solely on model outputs** can reliably detect spectral collapse attacks. Adversaries can optimize token sequences via Hidden State Poisoning (HiSPA) to crush internal dynamics while maintaining superficially plausible output logits.
3. **Spectral Geometry as a Monosemantic Signal:** Unlike complex feature activations, the eigenvalue spectrum is a low-dimensional, monosemantic quantity that perfectly dictates information retention. SpectralGuard acts as a gatekeeper against this specific failure mode with sub-15ms latency per token footprint.

<div align="center">
<br>
<img src="assets/hidden_state_trajectories_3d.png" alt="SpectralGuard – Phase-space trajectory of spectral hazard detection" width="70%"/>
<br><em>Phase-space trajectory of Mamba-130M hidden states. Benign dynamics maintain a stable orbit (blue), while a HiSPA attack forces contraction (red). SpectralGuard intervenes before complete memory collapse (green).</em>
<br><br>

<img src="assets/spectral_phase_transition.png" alt="Spectral Phase Transition" width="48%"/>
<img src="assets/accuracy_vs_distance.png" alt="Accuracy vs Distance" width="48%"/>
<br><em>Empirical evidence of the Spectral Phase Transition in Mamba-130M, demonstrating the sharp drop in reasoning capacity when spectral stability is compromised.</em>
<br><br>
</div>

### Key Results

Our extensive evaluation demonstrates that spectral monitoring provides a principled, deployable safety layer:

| Metric                                | Value         |
| ------------------------------------- | ------------- |
| Main Defense F1-Score                 | **0.961**     |
| Main Defense AUROC                    | **0.989**     |
| Adaptive Attack (HiSPA v4) F1-Score   | **0.842**     |
| Adaptive Attack AUROC                 | **0.903**     |
| Monitoring Overhead per token/layer   | **< 15 ms**   |
| Cross-architecture Transfer (Zamba-2) | **Validated** |

<div align="center">
<br>
<img src="assets/spectralguard_confusion_matrix.png" alt="SpectralGuard Confusion Matrix" width="70%"/>
<br><em>SpectralGuard multi-layer detection performance across 500 tasks.</em>
<br><br>
</div>

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/DaviBonetto/spectralguard
cd spectralguard
pip install -r requirements.txt

# 2. Run the canonical defense evaluation
python scripts/run_main_defense_evaluation.py \
    --n 500 --output-dir artifacts/main_defense

# 3. Verify everything passes
pytest -q
```

---

## Public Links

- GitHub: https://github.com/DaviBonetto/spectralguard
- Paper PDF: https://github.com/DaviBonetto/spectralguard/blob/main/paper/paper.pdf
- Hugging Face Space: https://huggingface.co/spaces/DaviBonetto/spectralguard-demo
- Hugging Face Dataset: https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset

## Public API

The stable public contract is provided by the `spectralguard` package:

```python
from spectralguard import monitor

is_safe, spectral_hazard_score = monitor(
    prompt="What is the capital of France?",
    hidden_states={"rho_layers": [0.97, 0.95, 0.94]},
)
# is_safe: bool — True if below detection threshold
# spectral_hazard_score: float in [0, 1]
```

Class-based interface:

```python
from spectralguard import SpectralGuardDetector

detector = SpectralGuardDetector(threshold=0.5)
is_safe, score = detector.monitor(prompt, rho_layers=[0.97, 0.95, 0.94])
```

> **Stability guarantee:** The `monitor(prompt, hidden_states)` signature is frozen. Downstream integrations will not break across minor versions.

---

## Repository Layout

```
spectralguard/             ← Public package (stable API)
│   __init__.py
│   detector.py
│
├── core/                  ← Mamba wrapper + state extraction
├── spectral/              ← Eigenvalue analyzer, gramian, horizon predictor
├── security/              ← SpectralGuard detector + adversarial generator
├── utils/                 ← Dataset utilities, validation helpers
├── visualization/         ← Spectral plots, trajectory visualization
├── scripts/               ← Canonical experiment scripts
│   └── archive/           ← Superseded scripts (retained for history)
├── tests/                 ← Unit + integration tests
└── notebooks/             ← Canonical analysis notebooks (01–06)
│
paper/                     ← LaTeX source + compiled PDF
│   paper.tex
│   paper.pdf
└── figures/               ← All paper figures (canonical filenames)
│
artifacts/                 ← Experiment outputs [gitignored]
data/                      ← Benchmark dataset
docs/                      ← Space and Dataset READMEs, documentation
│   DATASET_README.md
│   SPACE_README.md
│
app.py                     ← HuggingFace Spaces demo
setup.py
requirements.txt
Dockerfile                 ← Docker configuration for Spaces
```

---

## Experiment Matrix

Every claim in the paper maps to a canonical script and an artifact folder. Run any experiment end-to-end with the commands below.

| #   | Experiment                     | Paper § | Script                                  | Output                                |
| --- | ------------------------------ | ------- | --------------------------------------- | ------------------------------------- |
| E1  | Spectral horizon validation    | §3      | `01_Spectral_Horizon_Validation.ipynb`  | `artifacts/main_defense/`             |
| E2  | Main defense evaluation (OOF)  | §4.1    | `run_main_defense_evaluation.py`        | `artifacts/main_defense/`             |
| E3  | Adaptive attack v4 (HiSPA)     | §4.2    | `run_adaptive_v4.py`                    | `artifacts/adaptive_v4/`              |
| E4  | Stealth attack v4-S            | §4.3    | `run_adaptive_v4_stealth.py`            | `artifacts/adaptive_v4_stealth/`      |
| E5  | GPT-2 baseline (rigorous)      | §4.4    | `run_gpt2_baseline_rigorous.py`         | `artifacts/gpt2_baseline/`            |
| E6  | Multilayer regression          | §4.5    | `run_multilayer_regression_rigorous.py` | `artifacts/multilayer_regression/`    |
| E7  | Causal intervention study      | §5      | `run_causal_intervention.py`            | `artifacts/causal/`                   |
| E8  | Cross-model transfer (Zamba-2) | §6      | `run_stealthy_transfer_zamba2.py`       | `artifacts/zamba2_v4_transfer/`       |
| E9  | Latency benchmark              | App. B  | `benchmark_latency.py`                  | `paper/figures/latency_benchmark.png` |
| E10 | Threshold ablation             | App. C  | `run_threshold_ablation.py`             | `artifacts/`                          |
| D1  | Dataset compilation            | —       | `compile_dataset.py`                    | `data/spectralguard_benchmark.csv`    |

### Core commands

```bash
# E2 — Main defense
python scripts/run_main_defense_evaluation.py \
    --n 500 --output-dir artifacts/main_defense \
    --figure-path paper/figures/NIXEtm04.png

# E3 — Adaptive attack
python scripts/run_adaptive_v4.py \
    --n 200 --seeds 42,123,456 --output-dir artifacts/adaptive_v4

# E4 — Stealth attack
python scripts/run_adaptive_v4_stealth.py \
    --n 200 --seeds 42,123,456 --surface-config s1 --run-tag S1 \
    --output-dir artifacts/adaptive_v4_stealth

# E5 — GPT-2 baseline
python scripts/run_gpt2_baseline_rigorous.py \
    --n 200 --seeds 42,123,456

# E6 — Multilayer regression
python scripts/build_multilayer_regression_features_v2.py
python scripts/run_multilayer_regression_rigorous.py \
    --input artifacts/multilayer_regression/features_v2.csv \
    --target-col target_signal

# E8 — Zamba2 transfer (run on T4/A100 recommended)
python scripts/run_stealthy_transfer_zamba2.py \
    --model-id Zyphra/Zamba2-2.7B \
    --prompt-csv artifacts/adaptive_v4_stealth/B2_fix/paired_prompts_for_gpt2.csv \
    --prompt-col prompt_text --label-col label \
    --n 200 --seeds 42,123,456 \
    --output-dir artifacts/zamba2_v4_transfer

# D1 — Dataset
python scripts/compile_dataset.py \
    --output data/spectralguard_benchmark.csv --min-prompts 1200

```

---

## Canonical Notebooks

Open these for interactive exploration of each paper section:

| Notebook                                                                                 | Topic                       |
| ---------------------------------------------------------------------------------------- | --------------------------- |
| [`01_Spectral_Horizon_Validation.ipynb`](notebooks/01_Spectral_Horizon_Validation.ipynb) | §3 — Horizon prediction     |
| [`02_Adversarial_CoT_Collapse.ipynb`](notebooks/02_Adversarial_CoT_Collapse.ipynb)       | §3 — CoT collapse detection |
| [`03_Spectral_Guard_Defense.ipynb`](notebooks/03_Spectral_Guard_Defense.ipynb)           | §4.1 — Defense evaluation   |
| [`04_Adaptive_Attack_Evaluation.ipynb`](notebooks/04_Adaptive_Attack_Evaluation.ipynb)   | §4.2 — Adaptive attack      |
| [`05_Scaling_And_Robustness.ipynb`](notebooks/05_Scaling_And_Robustness.ipynb)           | §4.5 — Scaling laws         |
| [`pareto_sweep_results.ipynb`](notebooks/pareto_sweep_results.ipynb)                     | §6.3 — Pareto frontier      |
| [`demo.ipynb`](notebooks/demo.ipynb)                                                     | Interactive demo            |

---

## Reproducibility

This repository is designed for full reproducibility. Every paper claim maps to an artifact via [`docs/claims-evidence-matrix.md`](docs/claims-evidence-matrix.md).

**Checklist:**

- [x] `docs/claims-evidence-matrix.md` maps claims to canonical artifacts
- [x] Multi-seed experiment policy (`42, 123, 456`) is enforced in runners
- [x] CI workflow runs testing on push/PR
- [ ] Hugging Face Space and Dataset links switched from target to live URLs

---

## Demo

Run locally:

```bash
python app.py
```

HuggingFace Spaces deployment:

```bash
pip install -r requirements.txt
python app.py
```

Target Space URL:

- `https://huggingface.co/spaces/DaviBonetto/spectralguard-demo`

The demo supports two modes:

- **`real_model`** — extracts layer-wise spectral traces from Mamba when model weights are available
- **`demo_mode`** — deterministic fallback for environments without local model weights

---

## Installation (development)

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Citation

If you use SpectralGuard in your research, please cite:

```bibtex
@inproceedings{bonetto2025spectralguard,
  title     = {SpectralGuard: Detecting Memory Collapse Attacks in State Space Models},
  author    = {Bonetto, Davi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

---

## Contact

- **Email:** davi.bonetto100@gmail.com
- **GitHub:** [@DaviBonetto](https://github.com/DaviBonetto)

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built with eigenvalues and stubbornness.</sub>
</div>
