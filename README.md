<div align="center">

# SpectralGuard

**Detecting Memory Collapse Attacks in State Space Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/DaviBonetto/spectralguard-demo)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Paper**](paper/paper.pdf) · [**Quickstart**](#quickstart) · [**Experiments**](#experiment-matrix) · [**Public API**](#public-api)

<br>
<img src="assets/hidden_state_trajectories_3d.png" alt="SpectralGuard Phase-space trajectory" width="70%"/>
<br><em>Phase-space trajectory of Mamba-130M hidden states. Benign dynamics maintain a stable orbit (blue), while a Hidden State Poisoning Attack (HiSPA) forces contraction (red). SpectralGuard intervenes before complete memory collapse (green), preserving reasoning capacity.</em>

</div>

---

State Space Models (SSMs) like **Mamba** achieve linear-time sequence processing through input-dependent recurrence. While highly efficient, this mechanism introduces a critical, silent vulnerability: **Spectral Collapse**.

SpectralGuard is a real-time, lightweight safety monitor for recurrent and hybrid foundation models. It tracks the spectral radius $\rho(\bar{A}_t)$ of the discretized transition operator across layers to defend against attacks that destroy internal reasoning capacity without triggering output-level alarms.

## The Vulnerability: Spectral Collapse

In architectures like Mamba, information is compressed into a $d$-dimensional hidden state. The retention of this memory is mathematically bounded by the **spectral radius** $\rho(\bar{A}_t)$ of its recurrent transition matrix.

When $\rho(\bar{A}_t) \approx 1$, the model remembers and reasons over long sequences. When $\rho(\bar{A}_t) \to 0$, information contracts exponentially, wiping the model's memory in just a few tokens.

<div align="center">
<img src="assets/spectral_phase_transition.png" alt="Spectral Phase Transition" width="48%"/>
<img src="assets/accuracy_vs_distance.png" alt="Accuracy vs Distance" width="48%"/>
<br><em>The Spectral Phase Transition in Mamba-130M. When spectral stability is compromised ($\rho < 0.90$), reasoning capacity effectively collapses across different context distances.</em>
</div>

### The Impossibility of Output-Level Defense

We mathematically prove an **Evasion Existence Theorem**: no defense operating solely on model outputs (e.g., perplexity filters, toxicity classifiers) can reliably detect spectral collapse attacks. The internal recurrence bottleneck allows adversaries to decouple internal devastation from output perplexity.

## The Threat: Hidden State Poisoning (HiSPA)

By applying gradient-based adversarial optimization against the discrete step size ($\Delta_t$), an attacker can construct "chain-of-thought" prompts that maliciously minimize $\rho(\bar{A}_t)$.

This causes the effective memory horizon to drop from millions of tokens to mere dozens, inducing catastrophic failure (~50 point accuracy drops) on complex tasks while keeping the adversarial text entirely natural and lexically indistinguishable from benign input.

<div align="center">
<img src="assets/information_retention_curve.png" alt="Information Retention" width="60%"/>
<br><em>Adversarial tokens actively force the spectral radius down, accelerating information loss relative to a benign trajectory.</em>
</div>

## The Defense: SpectralGuard

SpectralGuard intercepts inference and extracts feature signals ($[\rho_1\dots\rho_L, \sigma_1\dots\sigma_L]$) across multiple layers using rapid eigenvalue approximation (power method). A logistic classifier evaluates hazard levels layer-by-layer and blocks generation if it detects a collapse.

<div align="center">
<img src="assets/diagram.png" alt="SpectralGuard Architecture Diagram" width="80%"/>
<br><em>System Architecture: Adversarial tokens $x_t$ attempt to manipulate $\Delta_t$. SpectralGuard dynamically estimates $\rho$ and blocks propagation before the internal state $h_t$ is destroyed.</em>
</div>

### Why it Works (Mechanistic Interpretability)

The attack leaves an impossible-to-hide structural signature. Our layer-wise analysis reveals that to successfully collapse memory, attackers create a stark "bottleneck" (layers 4–10) that starves subsequent layers of context. Benign text generation naturally preserves unit-norm operation.

<div align="center">
<img src="assets/layerwise_collapse.png" alt="Layerwise Collapse Signature" width="60%"/>
<br><em>Layer-wise spectral signature showing adversarial contraction forcing bottlenecks in mid-to-early layers, a mechanic not found in benign computation.</em>
</div>

<br>

<div align="center">
<img src="assets/spectralguard_confusion_matrix.png" alt="Confusion Matrix" width="60%"/>
<br><em>SpectralGuard achieves near-perfect F1=0.961 separation on multi-layer diagnostics without false alarms on benign texts.</em>
</div>

## Complete Results & Robustness

Our comprehensive evaluation proves the efficacy and bounds of SpectralGuard across diverse settings.

### Scaling & Reproducibility

The phase transition is uniform regardless of model size, and detection performance is verified across multiple random seeds, confirming that the defense generalizes completely through Mamba **130M**, **1.4B**, and **2.8B** parameter scale thresholds.

<div align="center">
<img src="assets/phase4_results.png" alt="Scaling and Robustness" width="90%"/>
<br><em>Scaling and multi-seed robustness curves validating that spectral phenomena generalize perfectly from 130M up to 2.8B configurations.</em>
</div>

### Cross-Architecture Transfer

The spectral monitoring principle is not limited to pure Mamba architectures. We successfully transferred the detection mechanisms to hybrid architectures like **Zamba2-2.7B** (interleaved Mamba/Attention layers). Since the underlying SSM block is what drives the recurrence, the spectral signature can still be intercepted accurately.

<div align="center">
<img src="assets/zamba2_validation.png" alt="Cross Architecture Validation" width="50%"/>
<br><em>Validation that hybrid networks remain analyzable via the local continuous matrices $A_l$.</em>
</div>

### The Adversary's Pareto Frontier

Can the attacker optimize against SpectralGuard directly? Yes, but they hit an intractable wall. We proved the existence of a **topological lock**: if an attack minimizes structural damage to evade detection ($\Delta\rho \to 0$), it completely sacrifices stealth, destroying the token manifold. Attackers cannot achieve high lexical stealth and lethal spectral impact simultaneously.

<div align="center">
<img src="assets/pareto_curve.png" alt="Pareto Frontier" width="50%"/>
<br><em>Pareto frontier mapping lexical evasion against spectral damage. The frontier is capped; evasion implies low impact.</em>
</div>

### Deployment Viability

Throughput benchmarking confirms that the required layer-wise eigenvalue power iteration introduces only a $+15\%$ constant overhead relative to normal Mamba auto-regressive generation, ensuring viability in live production applications.

<div align="center">
<img src="assets/latency_benchmark.png" alt="Latency Overhead" width="50%"/>
<br><em>Inference auto-regressive latency benchmark per token output displaying negligible computational footprint for high-risk monitoring.</em>
</div>

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/DaviBonetto/spectralguard.git
cd spectralguard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the canonical defense evaluation over 500 tasks
python scripts/run_main_defense_evaluation.py \
    --n 500 --output-dir artifacts/main_defense

# 4. Verify local unit tests pass safely
python -m pytest tests/ -v -q
```

---

## Public Links

- **GitHub Source Code:** [https://github.com/DaviBonetto/spectralguard](https://github.com/DaviBonetto/spectralguard)
- **Pre-compiled Paper PDF:** [`paper/paper.pdf`](paper/paper.pdf)
- **Live Hugging Face Space (Demo):** [https://huggingface.co/spaces/DaviBonetto/spectralguard-demo](https://huggingface.co/spaces/DaviBonetto/spectralguard-demo)
- **Dataset Benchmark:** [https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset](https://huggingface.co/datasets/DaviBonetto/spectralguard-dataset)

---

## Public API

The primary interface into the SpectralGuard defense mechanism exposes the underlying model. The static structure allows it to adapt regardless of intermediate framework shifts.

```python
from security.spectral_guard import SpectralGuard

# Initialize defense layer with selected sensitivity threshold
defender = SpectralGuard(threshold=0.30)

# Simulate token streaming, passing the computed hidden state layer values
is_safe, hazard_score = defender.check_prompt(
    prompt_text="Explain quantum mechanics.",
    rho_values=[0.97, 0.95, 0.94]
)

# is_safe: bool — True if the prompt is above the collapse threshold
# hazard_score: float in [0, 1] - Probabilistic confidence of manipulation
```

> **Stability guarantee:** Downstream integrations mapping into the `SpectralGuard` interface will not break across minor version iterations.

---

## Repository Layout

```text
spectralguard/
├── core/                  ← Mamba wrapper + state extraction layers
├── spectral/              ← Eigenvalue analyzer & horizon predictor
├── security/              ← SpectralGuard detector logic
├── utils/                 ← Dataset utilities & validation helpers
├── visualization/         ← Visual tools for trajectory mapping
├── scripts/               ← Canonical experiment pipeline scripts
├── tests/                 ← Unit and integration tests for APIs
├── notebooks/             ← Canonical analysis notebooks (01–06)
├── paper/                 ← LaTeX source code + compiled paper.pdf
│   └── figures/           ← Rendered output plots and diagrams
├── artifacts/             ← Output of experiments [gitignored]
├── data/                  ← Stored benchmark datasets
├── docs/                  ← Supplementary text & matrices
├── app.py                 ← Main execution for HuggingFace Spaces
├── requirements.txt       ← Pinning model / system dependencies
└── Dockerfile             ← Configuration file for container building
```

---

## Experiment Matrix

We built this repository specifically around reproducibility. You can launch any phase of our paper evaluation using the canonical pipelines linked below. All configuration specifics correspond precisely with the metrics documented in section _§4_ of our publication.

| #   | Experiment                     | Script Command                          | Reference |
| --- | ------------------------------ | --------------------------------------- | --------- |
| E1  | Spectral horizon validation    | `01_Spectral_Horizon_Validation.ipynb`  | §3        |
| E2  | Main defense evaluation        | `python scripts/run_main_defense_...`   | §4.1      |
| E3  | Adaptive attack v4 (HiSPA)     | `python scripts/run_adaptive_v4.py`     | §4.2      |
| E4  | Stealth attack v4-S            | `python scripts/run_adaptive_v4_...`    | §4.3      |
| E5  | GPT-2 baseline                 | `python scripts/run_gpt2_baseline_...`  | §4.4      |
| E6  | Multilayer regression          | `python scripts/build_multilayer...`    | §4.5      |
| E7  | Causal intervention study      | `python scripts/run_causal_interven...` | §5        |
| E8  | Cross-model transfer (Zamba-2) | `python scripts/run_stealthy_transf...` | §6        |
| E9  | Latency benchmark              | `python scripts/benchmark_latency.py`   | App. B    |

Check our interactive **Jupyter Notebooks**, such as [`pareto_sweep_results.ipynb`](notebooks/pareto_sweep_results.ipynb), to dig deeper into visualization rendering steps or the threshold ablation boundaries.

---

## Demo

Deploy locally via standard Python invocation:

```bash
python app.py
```

The system will automatically initialize the Gradio Web-Server at port `:7860`. The Space features two diagnostic settings:

1. `real_model`: Hooks seamlessly into Mamba local layer parameters.
2. `demo_mode`: Pure visual simulation mimicking spectral drop-offs for lower-bandwidth environments.

---

## Installation (Development)

To modify source scripts, establish a local interactive package:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Citation

If our mechanistic tracking paradigm or attack surface quantification aids your continuous state space research, please cite:

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

- **Email:** [davi.bonetto100@gmail.com](mailto:davi.bonetto100@gmail.com)
- **GitHub:** [@DaviBonetto](https://github.com/DaviBonetto)

---

## License

**MIT License** — complete provisions available under [`LICENSE`](LICENSE).

<br>
<div align="center">
<sub>Built with eigenvalues and stubbornness.</sub>
</div>
