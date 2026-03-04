# Prompt for Claude - Professional Repository Cleanup (Non-Destructive)

You are acting as a repository organization architect for an ML research project.

## Goal

Produce a professional, non-destructive cleanup and structure plan for this repository, inspired by high-quality research/engineering repos like:

- https://github.com/google-deepmind/alphafold
- https://github.com/huggingface/transformers

Do not execute deletions directly. First output a decision-complete manifest that can be applied safely.

## Hard constraints

1. Do not delete scientific evidence files that support paper claims.
2. Prefer archiving old/duplicate notebooks and outputs instead of removing them.
3. Preserve canonical paper files and canonical experiment scripts.
4. Keep current public API compatibility for `spectralguard.monitor(prompt, hidden_states)`.
5. Keep reproducibility traceability: every claim must still map to an artifact.

## Required output format

Return exactly these sections:

1. `Current State Diagnosis`
2. `Target Repository Layout`
3. `Canonical vs Archive Classification`
4. `File Move Manifest (source -> destination)`
5. `Safe Delete List (only low-risk generated clutter)`
6. `README Upgrade Draft`
7. `Migration Risk Checklist`

## Classification policy

Use these categories for each major file/folder:

- `CANONICAL`: must stay in primary tree
- `ARCHIVE`: move to `archive/` (retain history)
- `GENERATED`: can be ignored or deleted safely
- `AMBIGUOUS`: needs human confirmation

## Repository-specific priorities

1. Normalize notebooks:
   - keep only active canonical runbooks in `mamba_spectral/notebooks/`
   - move old result-heavy notebooks to `mamba_spectral/notebooks/archive/`
2. Normalize artifacts:
   - keep canonical paths under `artifacts/<experiment_name>/`
   - remove duplicate result folders with inconsistent naming
3. Improve top-level readability:
   - concise top-level directories
   - clear separation of `paper`, `scripts`, `package`, `data`, `docs`
4. Propose `.gitignore` refinements for generated outputs and notebook checkpoints.
5. Draft a professional README with:
   - project summary
   - quickstart in 3 commands
   - experiment matrix
   - reproducibility checklist
   - citation block

## Deliverable quality bar

Your output must be executable by another engineer without extra decisions.
If unsure about a file classification, mark it as `AMBIGUOUS` and explain why.
