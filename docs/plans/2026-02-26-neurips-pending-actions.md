# Pending Actions - NeurIPS (SpectralGuard)

Last update: 2026-02-27

This file tracks what is still open after the latest B2/B3 execution round.

## Current status summary

- Bugs #2 to #9 (Claude list) are already addressed in text/code and validated.
- Bug #1 (lexical leakage / stealth gate) is still OPEN.
- B2/B3 stress runs finished and did not promote the gate.

## Bug #1 status after B2/B3

Executed artifacts:

- `artifacts/adaptive_v4_stealth/B2/adaptive_v4_stealth_summary.csv`
- `artifacts/adaptive_v4_stealth/B3/adaptive_v4_stealth_summary.csv`
- `artifacts/adaptive_v4_stealth/stealth_gate_runs.csv`

Observed mean outcomes (B2 and B3):

- `lexical_auc_word_effective = 0.7783`
- `lexical_auc_char_effective = 0.8083`
- `delta_rho_mean = -8.8e-05`
- `claim_promoted_lexical_stealth = 0`
- `claim_promoted_all_seeds = 0`

Additional audit:

- Identity-selection collapse: `300/300` adversarial rows are identical to base prompt in both B2 and B3 (`prompt_text == base_prompt`).
- This indicates objective degeneracy (no effective perturbation), not a successful stealth attacker.

## Open tasks (decision-complete order)

### 1) Fix attacker degeneracy (highest priority)

- Prevent identity candidate from winning in stealth search.
- Enforce perturbation floor (token edit distance / structural change minimum).
- Log and report `identity_selection_rate` per run.
- Canonical rerun notebook: `mamba_spectral/notebooks/run_v4_stealth_b2_b3_nonidentity_t4.ipynb`.

### 2) Run tradeoff frontier experiment (new v5 track)

- Replace one-shot "pass/fail only" narrative with a measured frontier:
  - lexical separability (`AUC_word_effective`, `AUC_char_effective`)
  - spectral damage (`delta_rho_mean`)
- Execute 3 seeds on T4 and save canonical frontier CSV.

### 3) Update paper claims based on gate result

- If gate passes: promote stealth claim with strict evidence.
- If gate fails again: keep conservative negative-result framing and explicitly document frontier/limits.

## Remaining impact track items (post Bug #1)

- Package polish (`pip install spectralguard`) and API contract stability.
- HF Spaces demo hardening (`real_model` + `demo_mode`).
- Public dataset release packaging and metadata transparency.
- Real jailbreak scenario experiment (separate from capability-only degradation).

## Acceptance criteria to mark Bug #1 closed

All must hold simultaneously on 3 seeds:

- `AUC_lex_word_effective < 0.60`
- `AUC_lex_char_effective < 0.60`
- `delta_rho_mean >= 0.02`
- No identity-collapse behavior in adversarial rows.
