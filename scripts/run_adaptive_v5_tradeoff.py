import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_adaptive_v4_stealth import (
    analyze_prompt,
    auc_effective,
    char_delta,
    detector_metrics,
    eval_lexical_auc,
    generate_surface_candidates,
    neutral_payload_loss,
    sanitize_run_tag,
    set_seed,
    summarize_across_seeds,
    tie_lm_head,
)
from security.adversarial_gen import HiSPAv4
from utils.datasets import load_safe_prompts

try:
    import joblib
except Exception:
    joblib = None


FAMILY_CHOICES = ["surface_only", "semantic_preserve", "hybrid_budgeted"]

METRIC_SCHEMA_COLUMNS = [
    "experiment_id",
    "model_id",
    "attacker_type",
    "split",
    "n",
    "metric_name",
    "metric_value",
    "ci_low",
    "ci_high",
    "seed",
    "artifact_path",
]


def parse_seeds(seeds_text: str) -> List[int]:
    values = [int(x.strip()) for x in seeds_text.split(",") if x.strip()]
    if not values:
        raise ValueError("No valid seeds were provided.")
    return values


def split_clauses(text: str) -> List[str]:
    # Keep punctuation delimiters as standalone pieces.
    parts = re.split(r"([,;:.!?])", text)
    clauses: List[str] = []
    cur = []
    for p in parts:
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue
        if p in [",", ";", ":", ".", "!", "?"]:
            cur.append(p)
            clauses.append(" ".join(cur).strip())
            cur = []
        else:
            cur.append(p)
    if cur:
        clauses.append(" ".join(cur).strip())
    return [c for c in clauses if c]


def semantic_preserve_candidates(base_prompt: str, n_candidates: int, rng: random.Random) -> List[str]:
    clauses = split_clauses(base_prompt)
    if len(clauses) < 2:
        words = base_prompt.split()
        if len(words) < 4:
            return [base_prompt]
        clauses = [" ".join(words[i : i + 6]) for i in range(0, len(words), 6)]

    candidates = set()
    attempts = max(20, n_candidates * 12)
    for _ in range(attempts):
        seq = clauses[:]
        op = rng.choice(["rotate", "swap", "reverse", "shuffle"])
        if op == "rotate" and len(seq) >= 2:
            k = rng.randint(1, len(seq) - 1)
            seq = seq[k:] + seq[:k]
        elif op == "swap" and len(seq) >= 2:
            i, j = rng.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
        elif op == "reverse":
            seq = list(reversed(seq))
        else:
            rng.shuffle(seq)

        txt = " ".join(seq)
        txt = re.sub(r"\s+", " ", txt).strip()
        # Mild punctuation perturbation while preserving lexical content.
        txt = txt.replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
        if rng.random() < 0.5:
            txt = txt.replace(",", " , ").replace(";", " ; ")
            txt = re.sub(r"\s+", " ", txt).strip()
        candidates.add(txt)
        if len(candidates) >= n_candidates:
            break

    if not candidates:
        return [base_prompt]
    return list(candidates)


def hybrid_budgeted_candidates(
    base_prompt: str,
    n_candidates: int,
    rng: random.Random,
    min_char_delta: float,
    max_char_delta: float,
) -> List[str]:
    surface = generate_surface_candidates(
        base_prompt=base_prompt,
        surface_config="s4",
        n_candidates=max(n_candidates, 12),
        include_base_candidate=False,
        punct_intensity=0.85,
        spacing_intensity=0.80,
        reorder_prob=1.0,
        repeat_burst_len=6,
        repeat_burst_prob=0.35,
        rng=rng,
    )
    semantic = semantic_preserve_candidates(base_prompt=base_prompt, n_candidates=max(8, n_candidates // 2), rng=rng)
    pool = list(dict.fromkeys(surface + semantic))
    filt = []
    for cand in pool:
        d = char_delta(base_prompt, cand)
        if min_char_delta <= d <= max_char_delta:
            filt.append((cand, d))
    if not filt:
        # Fallback to closest candidates around target budget.
        target = (min_char_delta + max_char_delta) / 2.0
        scored = sorted([(cand, char_delta(base_prompt, cand)) for cand in pool], key=lambda x: abs(x[1] - target))
        filt = scored[: max(1, min(len(scored), n_candidates))]
    return [cand for cand, _ in filt[:n_candidates]]


def budget_penalty(delta: float, min_delta: float, max_delta: float, target_delta: float) -> float:
    if delta < min_delta:
        return float(min_delta - delta)
    if delta > max_delta:
        return float(delta - max_delta)
    return float(abs(delta - target_delta))


def select_tradeoff_candidate(
    base_prompt: str,
    candidates: List[str],
    model,
    tokenizer,
    device,
    objective_helper: HiSPAv4,
    benign_profile: Dict[str, int],
    lambda_spec: float,
    lambda_lex: float,
    lambda_ppl: float,
    lambda_budget: float,
    min_char_delta: float,
    max_char_delta: float,
    target_char_delta: float,
    max_length: int,
    collapse_threshold: float,
) -> Dict:
    if not candidates:
        candidates = [base_prompt]

    best_obj = None
    best_row = None

    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        delta = char_delta(base_prompt, candidate)

        stats = analyze_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=candidate,
            device=device,
            max_length=max_length,
            collapse_threshold=collapse_threshold,
        )
        l_payload = neutral_payload_loss(base_prompt=base_prompt, candidate_prompt=candidate)
        l_spectral = float(stats["rho_mean"])
        l_lexical = float(objective_helper.lexical_loss(candidate, benign_profile=benign_profile))
        l_ppl = float(objective_helper.perplexity_loss(nll_loss=stats["ppl_loss"]))
        l_budget = budget_penalty(
            delta=delta,
            min_delta=min_char_delta,
            max_delta=max_char_delta,
            target_delta=target_char_delta,
        )
        l_total = float(
            l_payload
            + lambda_spec * l_spectral
            + lambda_lex * l_lexical
            + lambda_ppl * l_ppl
            + lambda_budget * l_budget
        )

        row = {
            "prompt_text": candidate,
            "rho_mean": stats["rho_mean"],
            "rho_std": stats["rho_std"],
            "rho_gap": stats["rho_gap"],
            "collapse_depth": stats["collapse_depth"],
            "ppl_loss": stats["ppl_loss"],
            "rho_layers": stats["rho_layers"],
            "L_payload": l_payload,
            "L_spectral": l_spectral,
            "L_lexical": l_lexical,
            "L_perplexity": l_ppl,
            "L_budget": l_budget,
            "L_total": l_total,
            "char_delta_from_base": float(delta),
            "is_identity_candidate": int(candidate == base_prompt.strip()),
        }
        if best_obj is None or l_total < best_obj:
            best_obj = l_total
            best_row = row

    if best_row is None:
        best_row = {
            "prompt_text": base_prompt,
            "rho_mean": 1.0,
            "rho_std": 0.0,
            "rho_gap": 0.0,
            "collapse_depth": 0.0,
            "ppl_loss": 0.0,
            "rho_layers": [1.0],
            "L_payload": 1.0,
            "L_spectral": 1.0,
            "L_lexical": 1.0,
            "L_perplexity": 0.0,
            "L_budget": 0.0,
            "L_total": 1.0,
            "char_delta_from_base": 0.0,
            "is_identity_candidate": 1,
        }
    return best_row


def build_candidates_for_family(
    family: str,
    base_prompt: str,
    n_candidates: int,
    rng: random.Random,
    hybrid_min_char_delta: float,
    hybrid_max_char_delta: float,
) -> List[str]:
    if family == "surface_only":
        return generate_surface_candidates(
            base_prompt=base_prompt,
            surface_config="s4",
            n_candidates=n_candidates,
            include_base_candidate=False,
            punct_intensity=0.55,
            spacing_intensity=0.45,
            reorder_prob=0.85,
            repeat_burst_len=2,
            repeat_burst_prob=0.20,
            rng=rng,
        )
    if family == "semantic_preserve":
        return semantic_preserve_candidates(base_prompt=base_prompt, n_candidates=n_candidates, rng=rng)
    if family == "hybrid_budgeted":
        return hybrid_budgeted_candidates(
            base_prompt=base_prompt,
            n_candidates=n_candidates,
            rng=rng,
            min_char_delta=hybrid_min_char_delta,
            max_char_delta=hybrid_max_char_delta,
        )
    raise ValueError(f"Unknown family: {family}")


def metric_row(
    experiment_id: str,
    model_id: str,
    attacker_type: str,
    n: int,
    metric_name: str,
    metric_value: float,
    seed: int,
    artifact_path: str,
    ci_low="",
    ci_high="",
) -> Dict:
    return {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "attacker_type": attacker_type,
        "split": "balanced_internal",
        "n": int(n),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seed": int(seed),
        "artifact_path": artifact_path,
    }


def main():
    parser = argparse.ArgumentParser(description="AdaptiveHiSPA v5 trade-off frontier runner (Bug #1 pivot).")
    parser.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--damage-threshold", type=float, default=0.02)

    parser.add_argument("--families", default="surface_only,semantic_preserve,hybrid_budgeted")
    parser.add_argument("--n-candidates", type=int, default=12)
    parser.add_argument("--lambda-spec", type=float, default=12.0)
    parser.add_argument("--lambda-lex", type=float, default=1.0)
    parser.add_argument("--lambda-ppl", type=float, default=0.2)
    parser.add_argument("--lambda-budget", type=float, default=0.5)
    parser.add_argument("--hybrid-min-char-delta", type=float, default=0.08)
    parser.add_argument("--hybrid-max-char-delta", type=float, default=0.85)
    parser.add_argument("--target-char-delta", type=float, default=0.35)

    parser.add_argument("--run-tag", default="tradeoff_v5")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--clf", default="multilayer_spectral_guard.pkl")
    parser.add_argument("--scaler", default="multilayer_scaler.pkl")
    parser.add_argument("--output-dir", default="artifacts/adaptive_v5_tradeoff")
    args = parser.parse_args()

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    for f in families:
        if f not in FAMILY_CHOICES:
            raise ValueError(f"Invalid family '{f}'. Valid: {FAMILY_CHOICES}")

    seeds = parse_seeds(args.seeds)
    run_tag = sanitize_run_tag(args.run_tag)
    out_dir = Path(args.output_dir)
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    frontier_path = run_dir / "tradeoff_frontier.csv"
    summary_path = run_dir / "tradeoff_summary.csv"
    metrics_path = run_dir / "tradeoff_metrics.csv"
    samples_path = run_dir / "tradeoff_samples.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, local_files_only=args.local_files_only).to(device)
    tie_lm_head(model)
    model.eval()

    clf = None
    scaler = None
    if joblib is not None and Path(args.clf).exists() and Path(args.scaler).exists():
        clf = joblib.load(args.clf)
        scaler = joblib.load(args.scaler)

    objective = HiSPAv4(
        lambda_spec=args.lambda_spec,
        lambda_lex=args.lambda_lex,
        lambda_ppl=args.lambda_ppl,
        payload_keywords=[],
    )

    frontier_rows = []
    summary_rows = []
    metric_rows = []
    sample_rows = []

    for family in families:
        collector: Dict[str, List[float]] = {
            "lexical_auc_word": [],
            "lexical_auc_char": [],
            "lexical_auc_word_effective": [],
            "lexical_auc_char_effective": [],
            "delta_rho_mean": [],
            "claim_promoted_lexical_stealth": [],
            "detector_auc": [],
            "detector_f1": [],
            "identity_selection_rate": [],
            "char_delta_from_base_mean": [],
        }

        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            benign_n = args.n // 2
            adv_n = args.n - benign_n

            benign_prompts = load_safe_prompts(n=benign_n)
            adv_bases = load_safe_prompts(n=adv_n)
            benign_profile = objective.build_benign_profile(benign_prompts)

            benign_stats = []
            for p in benign_prompts:
                s = analyze_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=p,
                    device=device,
                    max_length=args.max_length,
                    collapse_threshold=args.collapse_threshold,
                )
                benign_stats.append({"prompt_text": p, **s})

            adv_rows = []
            for base_prompt in adv_bases:
                cands = build_candidates_for_family(
                    family=family,
                    base_prompt=base_prompt,
                    n_candidates=args.n_candidates,
                    rng=rng,
                    hybrid_min_char_delta=args.hybrid_min_char_delta,
                    hybrid_max_char_delta=args.hybrid_max_char_delta,
                )
                best = select_tradeoff_candidate(
                    base_prompt=base_prompt,
                    candidates=cands,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    objective_helper=objective,
                    benign_profile=benign_profile,
                    lambda_spec=args.lambda_spec,
                    lambda_lex=args.lambda_lex,
                    lambda_ppl=args.lambda_ppl,
                    lambda_budget=args.lambda_budget,
                    min_char_delta=args.hybrid_min_char_delta,
                    max_char_delta=args.hybrid_max_char_delta,
                    target_char_delta=args.target_char_delta,
                    max_length=args.max_length,
                    collapse_threshold=args.collapse_threshold,
                )
                best["base_prompt"] = base_prompt
                best["family"] = family
                best["seed"] = seed
                adv_rows.append(best)

            labels = np.asarray(([0] * len(benign_stats)) + ([1] * len(adv_rows)), dtype=int)
            all_prompts = benign_prompts + [r["prompt_text"] for r in adv_rows]

            lexical_auc_word = eval_lexical_auc(all_prompts, labels, seed, args.test_size, mode="word")
            lexical_auc_char = eval_lexical_auc(all_prompts, labels, seed, args.test_size, mode="char")
            lexical_auc_word_eff = auc_effective(lexical_auc_word)
            lexical_auc_char_eff = auc_effective(lexical_auc_char)

            benign_rho_mean = float(np.mean([r["rho_mean"] for r in benign_stats]))
            adv_rho_mean = float(np.mean([r["rho_mean"] for r in adv_rows]))
            delta_rho_mean = float(benign_rho_mean - adv_rho_mean)
            identity_rate = float(np.mean([r["is_identity_candidate"] for r in adv_rows]))
            char_delta_mean = float(np.mean([r["char_delta_from_base"] for r in adv_rows]))

            claim_promoted = float(
                (lexical_auc_word_eff < 0.60)
                and (lexical_auc_char_eff < 0.60)
                and (delta_rho_mean >= args.damage_threshold)
            )

            benign_feat = [r["rho_layers"] + [r["rho_mean"], r["rho_std"], r["rho_gap"]] for r in benign_stats]
            adv_feat = [r["rho_layers"] + [r["rho_mean"], r["rho_std"], r["rho_gap"]] for r in adv_rows]
            det = detector_metrics(benign_feat + adv_feat, labels, clf=clf, scaler=scaler)

            frontier_rows.append(
                {
                    "run_id": run_tag,
                    "family": family,
                    "seed": seed,
                    "n": args.n,
                    "lexical_auc_word": lexical_auc_word,
                    "lexical_auc_char": lexical_auc_char,
                    "lexical_auc_word_effective": lexical_auc_word_eff,
                    "lexical_auc_char_effective": lexical_auc_char_eff,
                    "delta_rho_mean": delta_rho_mean,
                    "detector_auc": det["auc"],
                    "detector_f1": det["f1"],
                    "identity_selection_rate": identity_rate,
                    "char_delta_from_base_mean": char_delta_mean,
                    "claim_promoted_lexical_stealth": int(claim_promoted),
                }
            )

            for metric_name, metric_value in {
                "lexical_auc_word_effective": lexical_auc_word_eff,
                "lexical_auc_char_effective": lexical_auc_char_eff,
                "delta_rho_mean": delta_rho_mean,
                "identity_selection_rate": identity_rate,
                "char_delta_from_base_mean": char_delta_mean,
                "detector_auc": det["auc"],
                "detector_f1": det["f1"],
                "claim_promoted_lexical_stealth": claim_promoted,
            }.items():
                metric_rows.append(
                    metric_row(
                        experiment_id="adaptive_hispa_v5_tradeoff",
                        model_id=args.model_id,
                        attacker_type=f"adaptive_v5_{family}",
                        n=args.n,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        seed=seed,
                        artifact_path=str(metrics_path),
                    )
                )

            collector["lexical_auc_word"].append(lexical_auc_word)
            collector["lexical_auc_char"].append(lexical_auc_char)
            collector["lexical_auc_word_effective"].append(lexical_auc_word_eff)
            collector["lexical_auc_char_effective"].append(lexical_auc_char_eff)
            collector["delta_rho_mean"].append(delta_rho_mean)
            collector["claim_promoted_lexical_stealth"].append(claim_promoted)
            collector["detector_auc"].append(det["auc"])
            collector["detector_f1"].append(det["f1"])
            collector["identity_selection_rate"].append(identity_rate)
            collector["char_delta_from_base_mean"].append(char_delta_mean)

            for i, row in enumerate(adv_rows):
                sample_rows.append(
                    {
                        "run_id": run_tag,
                        "family": family,
                        "seed": seed,
                        "prompt_id": f"{family}_{seed}_{i}",
                        "base_prompt": row["base_prompt"],
                        "prompt_text": row["prompt_text"],
                        "label": 1,
                        "rho_mean": row["rho_mean"],
                        "rho_std": row["rho_std"],
                        "rho_gap": row["rho_gap"],
                        "collapse_depth": row["collapse_depth"],
                        "char_delta_from_base": row["char_delta_from_base"],
                        "is_identity_candidate": row["is_identity_candidate"],
                        "L_total": row["L_total"],
                        "L_payload": row["L_payload"],
                        "L_spectral": row["L_spectral"],
                        "L_lexical": row["L_lexical"],
                        "L_perplexity": row["L_perplexity"],
                        "L_budget": row["L_budget"],
                    }
                )

        claim_all = float(all(v >= 1.0 for v in collector["claim_promoted_lexical_stealth"]))

        row = {
            "run_id": run_tag,
            "family": family,
            "seed": -1,
            "n": args.n,
            "claim_promoted_all_seeds": int(claim_all),
        }
        for name, vals in collector.items():
            mean, lo, hi = summarize_across_seeds(vals)
            row[name] = mean
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi
            metric_rows.append(
                metric_row(
                    experiment_id="adaptive_hispa_v5_tradeoff",
                    model_id=args.model_id,
                    attacker_type=f"adaptive_v5_{family}",
                    n=args.n,
                    metric_name=f"{name}_mean_over_seeds",
                    metric_value=mean,
                    seed=-1,
                    artifact_path=str(metrics_path),
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        summary_rows.append(row)

    frontier_df = pd.DataFrame(frontier_rows)
    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.DataFrame(metric_rows)[METRIC_SCHEMA_COLUMNS]
    samples_df = pd.DataFrame(sample_rows)

    frontier_df.to_csv(frontier_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    ledger_path = out_dir / "tradeoff_runs.csv"
    ledger_rows = summary_df[
        [
            "run_id",
            "family",
            "n",
            "lexical_auc_word_effective",
            "lexical_auc_char_effective",
            "delta_rho_mean",
            "claim_promoted_lexical_stealth",
            "claim_promoted_all_seeds",
        ]
    ].copy()
    if ledger_path.exists():
        old = pd.read_csv(ledger_path)
        if "run_id" in old.columns:
            old = old[old["run_id"].astype(str) != run_tag]
        pd.concat([old, ledger_rows], ignore_index=True).to_csv(ledger_path, index=False)
    else:
        ledger_rows.to_csv(ledger_path, index=False)

    print(f"Saved: {frontier_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved ledger: {ledger_path}")
    print("\nSummary (seed=-1 rows):")
    print(
        summary_df[
            [
                "family",
                "lexical_auc_word_effective",
                "lexical_auc_char_effective",
                "delta_rho_mean",
                "claim_promoted_lexical_stealth",
                "claim_promoted_all_seeds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

