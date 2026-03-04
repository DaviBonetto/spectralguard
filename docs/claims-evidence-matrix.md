# Claim -> Evidence Matrix

This matrix is the source of truth for paper claims and their backing artifacts.

| Claim ID | Claim (short) | Paper Location | Evidence Type | Artifact |
|---|---|---|---|---|
| C01 | Multi-layer SpectralGuard main protocol is synchronized at N=500 with F1 0.961, FPR 0.060, and counts TN=235/FP=15/FN=5/TP=245 | Section Results, Figure `guard_performance` + Table `consolidated_defense` | Metrics table + canonical confusion artifact | `paper/main.tex` + `artifacts/main_defense/main_defense_metrics.csv` |
| C02 | Single-layer baseline on Mamba-130M is F1 0.619 | Section Results, Table `scaling` + `consolidated_defense` | Metrics table | `paper/main.tex` |
| C03 | Adaptive v1 reduces single-layer to F1 0.370 | Section Results + Discussion | Metrics section | `paper/main.tex` |
| C04 | Multi-layer counter-defense reaches adaptive v2 F1 0.950 (Prec 0.941 / Rec 0.960) | Discussion + consolidated table | CSV metrics | `artifacts/adaptive_v2/adaptive_v2_metrics.csv` |
| C05 | Zamba2 cross-architecture detection F1 is 0.8911 | Experiment 5 + consolidated table | Metrics section | `paper/main.tex` |
| C06 | Causal clamping run with N=100 shows all-layer collapse and single-layer monotonic degradation | Experiment 3B + causal table | CSV + method | `artifacts/causal/causal_intervention_results.csv` |
| C07 | Adaptive v2 confusion-level evidence (TP=48, TN=47, FP=3, FN=2) | Adaptive analysis | CSV | `artifacts/adaptive_v2/adaptive_v2_defense_results.csv` |
| C08 | Hardware declarations are consolidated | Appendix A.4, table `hardware_map` | Documentation table | `paper/main.tex` |
| C09 | Placeholder bibliography IDs removed | Bibliography | Static check | `paper/main.tex` |
| C10 | AdaptiveHiSPA v3 yields F1 0.842 / AUC 0.903 (N=200) | Discussion + consolidated table | CSV metrics | `artifacts/adaptive_v3/adaptive_v3_metrics.csv` |
| C11 | Robust regression remains non-predictive under strict split (mean multivariate test R2 0.0195; CI [-0.2531, 0.1405]; promotion flag=0) | Experiment 1B | CSV metrics | `artifacts/multilayer_regression/multilayer_regression_metrics.csv` + `artifacts/multilayer_regression/multilayer_regression_summary.csv` |
| C12 | Paired GPT-2 proxy is perfectly separable in this run (attention AUC 1.00, hidden AUC 1.00), but lexical control is also saturated (AUC=1.00), indicating unresolved lexical leakage | Experiment 8 + Discussion | CSV metrics | `artifacts/gpt2_baseline/gpt2_baseline_rigorous_metrics.csv` + `artifacts/gpt2_baseline/gpt2_baseline_rigorous_summary.csv` |
| C13 | Qualitative FP/FN examples are grounded in executed adaptive v3 outputs | Discussion (Qualitative) | CSV examples | `artifacts/adaptive_v3/example_fp.csv` + `artifacts/adaptive_v3/example_fn.csv` |
| C14 | Task 11 robust protocol executed with stable target signal (`target_signal`) and multi-seed train/test split | Methods + scripts | Code + final artifacts | `mamba_spectral/scripts/build_multilayer_regression_features_v2.py` + `mamba_spectral/scripts/run_multilayer_regression.py` + `artifacts/multilayer_regression/multilayer_regression_summary.csv` |
| C15 | Task 18 rigorous protocol executed on 3 seeds with paired v4 prompt CSV and lexical audit | Experiment 8 methodology + scripts | Code + final artifacts | `mamba_spectral/scripts/run_gpt2_baseline_rigorous.py` + `artifacts/gpt2_baseline/gpt2_baseline_rigorous_summary.csv` |
| C16 | Public package API exposes `monitor(prompt, hidden_states)` with contract tests | Package + reproducibility | Code + tests | `spectralguard/detector.py` + `spectralguard/__init__.py` + `mamba_spectral/tests/test_spectralguard_api.py` |
| C17 | Gradio demo consumes package API with explicit `real_model`/`demo_mode` | Demo section / release notes | Code | `app.py` |
| C18 | Public benchmark build reached 1200 rows with schema valid and balanced labels (600/600), with explicit transparency on synthetic/real composition | Dataset section | CSV summary | `data/dataset/dataset_summary.csv` |
| C19 | Figure 4 confusion matrix is synchronized with main protocol counts (N=500) | Experiment 3 figure | Predictions + generated image | `artifacts/main_defense/main_defense_predictions.csv` + `paper/figures/NIXEtm04.png` |
| C20 | Legacy HiSPA v4 comparison run (pre-stress) is archived as non-promoted (`claim_promoted_lexical_stealth`=0; lexical AUC=1.00; mean spectral damage $\Delta\\rho$=0.0048 < 0.02 threshold) | Adaptive section history | CSV metrics + summary | `artifacts/adaptive_v4/adaptive_v4_metrics.csv` + `artifacts/adaptive_v4/adaptive_v4_summary.csv` |
| C21 | Stealthy transfer protocol (AdaptiveHiSPA v4 -> Zamba2) is implemented with standardized outputs and CLI validation | Phase 1.3 execution pipeline | Script + CLI test + Colab notebook | `mamba_spectral/scripts/run_stealthy_transfer_zamba2.py` + `mamba_spectral/tests/test_transfer_cli.py` + `mamba_spectral/notebooks/run_stealthy_transfer_t4.ipynb` |
| C22 | AdaptiveHiSPA v4 non-identity stress runs B2_fix/B3_fix (N=200, 3 seeds) remain non-promoted: lexical separability is saturated (`AUC_word_eff`=1.00, `AUC_char_eff`=1.00) while spectral damage stays below threshold ($\Delta\\rho=0.0047$ for B2_fix, $0.0051$ for B3_fix, both <0.02). Identity-selection collapse is resolved (0/600 exact-copy adversarial rows in each run). | Adaptive section + limitations | CSV metrics + summary + sample audit + dedicated rerun notebooks | `artifacts/adaptive_v4_stealth/stealth_gate_runs.csv` + `artifacts/adaptive_v4_stealth/B2_fix/adaptive_v4_stealth_summary.csv` + `artifacts/adaptive_v4_stealth/B3_fix/adaptive_v4_stealth_summary.csv` + `artifacts/adaptive_v4_stealth/B2_fix/adaptive_v4_stealth_samples.csv` + `artifacts/adaptive_v4_stealth/B3_fix/adaptive_v4_stealth_samples.csv` + `mamba_spectral/notebooks/run_v4_stealth_b2_fix_only_t4.ipynb` + `mamba_spectral/notebooks/run_v4_stealth_b3_fix_only_t4.ipynb` |
| C23 | AdaptiveHiSPA v5 trade-off frontier (surface_only / semantic_preserve / hybrid_budgeted; N=200, 3 seeds) also remains non-promoted. Best char-level point (`semantic_preserve`) has `AUC_char_eff`=0.567 but `AUC_word_eff`=0.706 and $\Delta\\rho=0.0017<0.02$; best word-level point (`hybrid_budgeted`) has `AUC_word_eff`=0.615 but `AUC_char_eff`=0.98 and negative $\Delta\\rho=-0.0033$. | Adaptive section + conclusion | Trade-off summary + frontier CSV | `artifacts/adaptive_v5_tradeoff/v5_tradeoff_main/tradeoff_summary.csv` + `artifacts/adaptive_v5_tradeoff/v5_tradeoff_main/tradeoff_frontier.csv` + `artifacts/adaptive_v5_tradeoff/tradeoff_runs.csv` + `mamba_spectral/scripts/run_adaptive_v5_tradeoff.py` |
| C24 | Public release endpoints and publish checklist are versioned in-repo for GitHub, Hugging Face Space, and Hugging Face Dataset. | Reproducibility and release notes | Documentation | `docs/release/public_artifacts.md` + `docs/release/hf_publish_checklist.md` + `README.md` |

## Reproducibility policy

- No numeric claim is promoted without a reproducible artifact (script + output file).
- Numeric claims C11/C12/C22/C23 are tied to canonical copies in `artifacts/multilayer_regression`, `artifacts/gpt2_baseline`, `artifacts/adaptive_v4_stealth`, and `artifacts/adaptive_v5_tradeoff`.
- Claims without precision/recall values are reported as `---` and excluded from inferential statements.

