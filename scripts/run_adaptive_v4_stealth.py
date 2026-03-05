import argparse
import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from security.adversarial_gen import HiSPAv4
from utils.datasets import load_safe_prompts

try:
    import joblib
except Exception:
    joblib = None

from transformers import AutoModelForCausalLM, AutoTokenizer


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


SURFACE_CONFIG_FLAGS = {
    "s1": {"punct": True, "spacing": False, "reorder": False},
    "s2": {"punct": True, "spacing": True, "reorder": False},
    "s3": {"punct": True, "spacing": True, "reorder": True},
    "s4": {"punct": True, "spacing": True, "reorder": True},
}

PUNCT_CHOICES = [", ", "; ", ": ", " - ", " ... ", " / ", " | "]
SPACE_CHOICES = [" ", "  ", "   ", " \n ", " \t "]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tie_lm_head(model) -> None:
    if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
        model.lm_head.weight = model.get_input_embeddings().weight


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def sanitize_run_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned or "default"


def char_delta(base_text: str, candidate_text: str) -> float:
    base = (base_text or "").strip()
    cand = (candidate_text or "").strip()
    if not base and not cand:
        return 0.0
    return float(1.0 - SequenceMatcher(None, base, cand).ratio())


def apply_punctuation(words: List[str], rng: random.Random, intensity: float) -> str:
    if len(words) <= 1:
        return " ".join(words)
    parts: List[str] = []
    for i, word in enumerate(words):
        parts.append(word)
        if i == len(words) - 1:
            break
        if rng.random() < intensity:
            parts.append(rng.choice(PUNCT_CHOICES))
        else:
            parts.append(" ")
    return "".join(parts).strip()


def apply_spacing(text: str, rng: random.Random, intensity: float) -> str:
    items = re.split(r"\s+", text.strip())
    if len(items) <= 1:
        return text.strip()
    out: List[str] = []
    for i, token in enumerate(items):
        out.append(token)
        if i == len(items) - 1:
            break
        if rng.random() < intensity:
            out.append(rng.choice(SPACE_CHOICES))
        else:
            out.append(" ")
    return "".join(out).strip()


def apply_reorder(words: List[str], rng: random.Random) -> List[str]:
    if len(words) < 6:
        return words[:]
    chunk_size = rng.choice([2, 3, 4])
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
    if len(chunks) < 2:
        return words[:]

    mode = rng.choice(["rotate", "swap", "reverse"])
    if mode == "rotate":
        chunks = chunks[1:] + chunks[:1]
    elif mode == "swap":
        chunks[0], chunks[1] = chunks[1], chunks[0]
    else:
        chunks = list(reversed(chunks))
    return [token for chunk in chunks for token in chunk]


def generate_surface_candidates(
    base_prompt: str,
    surface_config: str,
    n_candidates: int,
    include_base_candidate: bool,
    punct_intensity: float,
    spacing_intensity: float,
    reorder_prob: float,
    repeat_burst_len: int,
    repeat_burst_prob: float,
    rng: random.Random,
) -> List[str]:
    flags = SURFACE_CONFIG_FLAGS[surface_config]
    base_words = base_prompt.strip().split()
    if not base_words:
        return [base_prompt]

    candidates = set()
    if include_base_candidate:
        candidates.add(base_prompt.strip())
    max_attempts = max(20, n_candidates * 10)

    for _ in range(max_attempts):
        words = base_words[:]
        if flags["reorder"] and rng.random() < reorder_prob:
            words = apply_reorder(words=words, rng=rng)

        text = " ".join(words)
        if flags["punct"]:
            text = apply_punctuation(words=text.split(), rng=rng, intensity=punct_intensity)
        if flags["spacing"]:
            text = apply_spacing(text=text, rng=rng, intensity=spacing_intensity)
        if repeat_burst_len > 0 and rng.random() < repeat_burst_prob:
            burst: List[str] = []
            for j in range(repeat_burst_len):
                burst.append(rng.choice(base_words))
                if j < repeat_burst_len - 1 and rng.random() < 0.35:
                    burst.append(rng.choice([",", ";", ":", "-", "..."]))
            text = f"{text} {' '.join(burst)}"

        text = text.strip()
        if text:
            candidates.add(text)
        if len(candidates) >= n_candidates:
            break

    return list(candidates)


def neutral_payload_loss(base_prompt: str, candidate_prompt: str) -> float:
    base_tokens = simple_tokens(base_prompt)
    cand_tokens = simple_tokens(candidate_prompt)
    if not base_tokens or not cand_tokens:
        return 1.0

    base_counter = Counter(base_tokens)
    cand_counter = Counter(cand_tokens)
    overlap = sum(min(base_counter[token], cand_counter[token]) for token in base_counter.keys())
    precision = overlap / max(len(cand_tokens), 1)
    recall = overlap / max(len(base_tokens), 1)
    f1_like = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    length_penalty = abs(len(candidate_prompt) - len(base_prompt)) / max(len(base_prompt), 1)
    return float((1.0 - f1_like) + 0.1 * length_penalty)


def analyze_prompt(model, tokenizer, prompt: str, device, max_length: int, collapse_threshold: float = 0.90) -> Dict[str, float]:
    n_layers = len(model.backbone.layers)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    layer_deltas = {i: [] for i in range(n_layers)}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            layer_deltas[layer_idx].append(tensor.detach().float().cpu())

        return hook

    for i in range(n_layers):
        handles.append(model.backbone.layers[i].mixer.dt_proj.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])

    for handle in handles:
        handle.remove()

    rho_layers = []
    for i in range(n_layers):
        mixer = model.backbone.layers[i].mixer
        A = -torch.exp(mixer.A_log.detach().float())
        deltas = layer_deltas[i]
        if not deltas:
            rho_layers.append(0.95)
            continue
        dt = F.softplus(deltas[0]).unsqueeze(-1)
        A_bar = torch.exp(dt * A.cpu())
        rho_token = A_bar.max(dim=-1).values.mean(dim=-1)
        rho_layers.append(float(rho_token.min().item()))

    rho_arr = np.asarray(rho_layers, dtype=np.float32)
    return {
        "rho_mean": float(np.mean(rho_arr)),
        "rho_std": float(np.std(rho_arr)),
        "rho_gap": float(np.max(rho_arr) - np.min(rho_arr)),
        "collapse_depth": float(np.sum(rho_arr < collapse_threshold)),
        "ppl_loss": float(out.loss.item()),
        "rho_layers": rho_arr.tolist(),
    }


def eval_lexical_auc(prompts: List[str], labels: np.ndarray, seed: int, test_size: float, mode: str) -> float:
    idx = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    if mode == "word":
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    elif mode == "char":
        vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    else:
        raise ValueError(f"Unknown lexical mode: {mode}")

    x_train = vectorizer.fit_transform([prompts[i] for i in train_idx])
    x_test = vectorizer.transform([prompts[i] for i in test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def detector_metrics(feature_rows: List[List[float]], labels: np.ndarray, clf, scaler) -> Dict[str, float]:
    X = np.asarray(feature_rows, dtype=np.float32)
    if clf is not None and scaler is not None:
        try:
            Xs = scaler.transform(X)
            y_prob = clf.predict_proba(Xs)[:, 1]
            y_pred = clf.predict(Xs)
        except Exception:
            rho_mean = X[:, -3]
            rho_std = X[:, -2]
            rho_gap = X[:, -1]
            y_prob = np.clip((0.95 - rho_mean) + 0.4 * rho_std + 0.2 * rho_gap, 0.0, 1.0)
            y_pred = (y_prob >= 0.5).astype(int)
    else:
        rho_mean = X[:, -3]
        rho_std = X[:, -2]
        rho_gap = X[:, -1]
        y_prob = np.clip((0.95 - rho_mean) + 0.4 * rho_std + 0.2 * rho_gap, 0.0, 1.0)
        y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred, zero_division=0)),
        "recall": float(recall_score(labels, y_pred, zero_division=0)),
        "f1": float(f1_score(labels, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(labels, y_prob)),
    }


def summarize_across_seeds(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    lo, hi = np.percentile(arr, [2.5, 97.5]).astype(float)
    return mean, float(lo), float(hi)


def auc_effective(auc_raw: float) -> float:
    # AUC < 0.5 can still imply separability with inverted orientation.
    # For lexical stealth we use orientation-invariant separability.
    return float(max(auc_raw, 1.0 - auc_raw))


def select_best_candidate(
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
    max_length: int,
    collapse_threshold: float,
    allow_identity_candidate: bool,
    min_char_delta: float,
) -> Dict:
    if not candidates:
        candidates = [base_prompt]

    filtered_candidates: List[Tuple[str, float]] = []
    base_clean = base_prompt.strip()
    for candidate in candidates:
        cand_clean = (candidate or "").strip()
        if not cand_clean:
            continue
        delta = char_delta(base_clean, cand_clean)
        if (not allow_identity_candidate) and cand_clean == base_clean:
            continue
        if delta < min_char_delta:
            continue
        filtered_candidates.append((cand_clean, delta))

    # Fallback: never fail hard; pick most-changed candidates when strict filters remove all.
    if not filtered_candidates:
        fallback = []
        for candidate in candidates:
            cand_clean = (candidate or "").strip()
            if not cand_clean:
                continue
            if (not allow_identity_candidate) and cand_clean == base_clean:
                continue
            fallback.append((cand_clean, char_delta(base_clean, cand_clean)))
        if not fallback:
            fallback = [(base_clean, 0.0)]
        fallback = sorted(fallback, key=lambda x: x[1], reverse=True)
        filtered_candidates = fallback[: max(1, min(len(fallback), 5))]

    best_row = None
    best_obj = None

    for candidate, delta in filtered_candidates:
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
        l_total = float(l_payload + lambda_spec * l_spectral + lambda_lex * l_lexical + lambda_ppl * l_ppl)

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
            "L_total": l_total,
            "char_delta_from_base": float(delta),
            "is_identity_candidate": int(candidate.strip() == base_clean),
        }
        if best_obj is None or l_total < best_obj:
            best_obj = l_total
            best_row = row

    return best_row


def main():
    parser = argparse.ArgumentParser(
        description="AdaptiveHiSPA v4 stealth redesign with dual lexical controls and hard promotion gate."
    )
    parser.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--lambda-spec", type=float, default=8.0)
    parser.add_argument("--lambda-lex", type=float, default=2.0)
    parser.add_argument("--lambda-ppl", type=float, default=0.3)
    parser.add_argument("--damage-threshold", type=float, default=0.02)
    parser.add_argument("--surface-config", choices=sorted(SURFACE_CONFIG_FLAGS.keys()), default="s1")
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--punct-intensity", type=float, default=0.35)
    parser.add_argument("--spacing-intensity", type=float, default=0.25)
    parser.add_argument("--reorder-prob", type=float, default=0.60)
    parser.add_argument("--repeat-burst-len", type=int, default=0)
    parser.add_argument("--repeat-burst-prob", type=float, default=0.0)
    parser.add_argument("--include-base-candidate", action="store_true")
    parser.add_argument("--allow-identity-candidate", action="store_true")
    parser.add_argument("--min-char-delta", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--clf", default="multilayer_spectral_guard.pkl")
    parser.add_argument("--scaler", default="multilayer_scaler.pkl")
    parser.add_argument("--output-dir", default="artifacts/adaptive_v4_stealth")
    args = parser.parse_args()

    seed_values = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seed_values:
        raise ValueError("No seeds provided.")

    run_tag = sanitize_run_tag(args.run_tag)
    run_dir = Path(args.output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "adaptive_v4_stealth_metrics.csv"
    samples_path = run_dir / "adaptive_v4_stealth_samples.csv"
    summary_path = run_dir / "adaptive_v4_stealth_summary.csv"
    paired_path = run_dir / "paired_prompts_for_gpt2.csv"

    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

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

    objective_helper = HiSPAv4(
        lambda_spec=args.lambda_spec,
        lambda_lex=args.lambda_lex,
        lambda_ppl=args.lambda_ppl,
        payload_keywords=[],
    )

    metric_rows = []
    sample_rows = []
    summary_rows = []
    paired_rows = []

    mean_collectors: Dict[str, List[float]] = {
        "lexical_auc_word": [],
        "lexical_auc_char": [],
        "lexical_auc_word_effective": [],
        "lexical_auc_char_effective": [],
        "delta_rho_mean": [],
        "identity_selection_rate": [],
        "char_delta_from_base_mean": [],
        "detector_f1": [],
        "detector_auc": [],
        "claim_promoted_lexical_stealth": [],
    }

    for seed in seed_values:
        set_seed(seed)
        rng = random.Random(seed)
        benign_n = args.n // 2
        adv_n = args.n - benign_n

        benign_prompts = load_safe_prompts(n=benign_n)
        adv_base_prompts = load_safe_prompts(n=adv_n)
        benign_profile = objective_helper.build_benign_profile(benign_prompts)

        benign_stats = []
        for prompt in benign_prompts:
            stats = analyze_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_length=args.max_length,
                collapse_threshold=args.collapse_threshold,
            )
            benign_stats.append({"prompt_text": prompt, **stats})

        adv_rows = []
        for base_prompt in adv_base_prompts:
            candidates = generate_surface_candidates(
                base_prompt=base_prompt,
                surface_config=args.surface_config,
                n_candidates=args.n_candidates,
                include_base_candidate=args.include_base_candidate,
                punct_intensity=args.punct_intensity,
                spacing_intensity=args.spacing_intensity,
                reorder_prob=args.reorder_prob,
                repeat_burst_len=args.repeat_burst_len,
                repeat_burst_prob=args.repeat_burst_prob,
                rng=rng,
            )
            best = select_best_candidate(
                base_prompt=base_prompt,
                candidates=candidates,
                model=model,
                tokenizer=tokenizer,
                device=device,
                objective_helper=objective_helper,
                benign_profile=benign_profile,
                lambda_spec=args.lambda_spec,
                lambda_lex=args.lambda_lex,
                lambda_ppl=args.lambda_ppl,
                max_length=args.max_length,
                collapse_threshold=args.collapse_threshold,
                allow_identity_candidate=args.allow_identity_candidate,
                min_char_delta=args.min_char_delta,
            )
            best["base_prompt"] = base_prompt
            adv_rows.append(best)

        benign_rho_mean = float(np.mean([row["rho_mean"] for row in benign_stats]))
        adv_rho_mean = float(np.mean([row["rho_mean"] for row in adv_rows]))
        delta_rho_mean = float(benign_rho_mean - adv_rho_mean)
        identity_selection_rate = float(np.mean([row.get("is_identity_candidate", 0) for row in adv_rows]))
        char_delta_from_base_mean = float(np.mean([row.get("char_delta_from_base", 0.0) for row in adv_rows]))

        all_prompts = benign_prompts + [row["prompt_text"] for row in adv_rows]
        labels = np.asarray(([0] * len(benign_prompts)) + ([1] * len(adv_rows)), dtype=int)

        lexical_auc_word = eval_lexical_auc(
            prompts=all_prompts, labels=labels, seed=seed, test_size=args.test_size, mode="word"
        )
        lexical_auc_char = eval_lexical_auc(
            prompts=all_prompts, labels=labels, seed=seed, test_size=args.test_size, mode="char"
        )
        lexical_auc_word_effective = auc_effective(lexical_auc_word)
        lexical_auc_char_effective = auc_effective(lexical_auc_char)
        claim_promoted = float(
            (lexical_auc_word_effective < 0.60)
            and (lexical_auc_char_effective < 0.60)
            and (delta_rho_mean >= args.damage_threshold)
        )

        benign_feat = [row["rho_layers"] + [row["rho_mean"], row["rho_std"], row["rho_gap"]] for row in benign_stats]
        adv_feat = [row["rho_layers"] + [row["rho_mean"], row["rho_std"], row["rho_gap"]] for row in adv_rows]
        det = detector_metrics(feature_rows=(benign_feat + adv_feat), labels=labels, clf=clf, scaler=scaler)

        metric_pairs = {
            "lexical_auc_word": lexical_auc_word,
            "lexical_auc_char": lexical_auc_char,
            "lexical_auc_word_effective": lexical_auc_word_effective,
            "lexical_auc_char_effective": lexical_auc_char_effective,
            "delta_rho_mean": delta_rho_mean,
            "identity_selection_rate": identity_selection_rate,
            "char_delta_from_base_mean": char_delta_from_base_mean,
            "rho_mean_benign": benign_rho_mean,
            "rho_mean_adversarial": adv_rho_mean,
            "claim_promoted_lexical_stealth": claim_promoted,
            "detector_accuracy": det["accuracy"],
            "detector_precision": det["precision"],
            "detector_recall": det["recall"],
            "detector_f1": det["f1"],
            "detector_auc": det["auc"],
        }
        for metric_name, metric_value in metric_pairs.items():
            metric_rows.append(
                {
                    "experiment_id": "adaptive_hispa_v4_stealth_v1",
                    "model_id": args.model_id,
                    "attacker_type": "adaptive_v4_stealth",
                    "split": "balanced_internal",
                    "n": int(args.n),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "ci_low": "",
                    "ci_high": "",
                    "seed": int(seed),
                    "artifact_path": str(metrics_path),
                }
            )

        summary_rows.append(
            {
                "seed": seed,
                "run_tag": run_tag,
                "surface_config": args.surface_config,
                "n_total": args.n,
                "lexical_auc_word": lexical_auc_word,
                "lexical_auc_char": lexical_auc_char,
                "lexical_auc_word_effective": lexical_auc_word_effective,
                "lexical_auc_char_effective": lexical_auc_char_effective,
                "delta_rho_mean": delta_rho_mean,
                "identity_selection_rate": identity_selection_rate,
                "char_delta_from_base_mean": char_delta_from_base_mean,
                "damage_threshold": args.damage_threshold,
                "claim_promoted_lexical_stealth": int(claim_promoted),
                "detector_f1": det["f1"],
                "detector_auc": det["auc"],
            }
        )

        for metric_name in mean_collectors.keys():
            if metric_name == "claim_promoted_lexical_stealth":
                mean_collectors[metric_name].append(float(claim_promoted))
            elif metric_name == "detector_f1":
                mean_collectors[metric_name].append(float(det["f1"]))
            elif metric_name == "detector_auc":
                mean_collectors[metric_name].append(float(det["auc"]))
            elif metric_name == "lexical_auc_word":
                mean_collectors[metric_name].append(float(lexical_auc_word))
            elif metric_name == "lexical_auc_char":
                mean_collectors[metric_name].append(float(lexical_auc_char))
            elif metric_name == "lexical_auc_word_effective":
                mean_collectors[metric_name].append(float(lexical_auc_word_effective))
            elif metric_name == "lexical_auc_char_effective":
                mean_collectors[metric_name].append(float(lexical_auc_char_effective))
            elif metric_name == "delta_rho_mean":
                mean_collectors[metric_name].append(float(delta_rho_mean))
            elif metric_name == "identity_selection_rate":
                mean_collectors[metric_name].append(float(identity_selection_rate))
            elif metric_name == "char_delta_from_base_mean":
                mean_collectors[metric_name].append(float(char_delta_from_base_mean))

        for i, row in enumerate(benign_stats):
            sample_rows.append(
                {
                    "seed": seed,
                    "run_tag": run_tag,
                    "surface_config": args.surface_config,
                    "attacker_type": "benign_reference",
                    "prompt_id": f"benign_{seed}_{i}",
                    "base_prompt": row["prompt_text"],
                    "prompt_text": row["prompt_text"],
                    "label": 0,
                    "rho_mean": row["rho_mean"],
                    "rho_std": row["rho_std"],
                    "rho_gap": row["rho_gap"],
                    "collapse_depth": row["collapse_depth"],
                    "ppl_loss": row["ppl_loss"],
                    "L_payload": 0.0,
                    "L_spectral": row["rho_mean"],
                    "L_lexical": 0.0,
                    "L_perplexity": row["ppl_loss"],
                    "L_total": row["rho_mean"],
                    "char_delta_from_base": 0.0,
                    "is_identity_candidate": 1,
                }
            )
            paired_rows.append(
                {
                    "seed": seed,
                    "run_tag": run_tag,
                    "attacker_type": "benign_reference",
                    "prompt_text": row["prompt_text"],
                    "label": 0,
                }
            )

        for i, row in enumerate(adv_rows):
            sample_rows.append(
                {
                    "seed": seed,
                    "run_tag": run_tag,
                    "surface_config": args.surface_config,
                    "attacker_type": "adaptive_v4_stealth",
                    "prompt_id": f"adaptive_v4_stealth_{seed}_{i}",
                    "base_prompt": row["base_prompt"],
                    "prompt_text": row["prompt_text"],
                    "label": 1,
                    "rho_mean": row["rho_mean"],
                    "rho_std": row["rho_std"],
                    "rho_gap": row["rho_gap"],
                    "collapse_depth": row["collapse_depth"],
                    "ppl_loss": row["ppl_loss"],
                    "L_payload": row["L_payload"],
                    "L_spectral": row["L_spectral"],
                    "L_lexical": row["L_lexical"],
                    "L_perplexity": row["L_perplexity"],
                    "L_total": row["L_total"],
                    "char_delta_from_base": row.get("char_delta_from_base", 0.0),
                    "is_identity_candidate": row.get("is_identity_candidate", 0),
                }
            )
            paired_rows.append(
                {
                    "seed": seed,
                    "run_tag": run_tag,
                    "attacker_type": "adaptive_v4_stealth",
                    "prompt_text": row["prompt_text"],
                    "label": 1,
                }
            )

    for metric_name, values in mean_collectors.items():
        mean, lo, hi = summarize_across_seeds(values)
        metric_rows.append(
            {
                "experiment_id": "adaptive_hispa_v4_stealth_v1",
                "model_id": args.model_id,
                "attacker_type": "adaptive_v4_stealth",
                "split": "balanced_internal",
                "n": int(args.n),
                "metric_name": f"{metric_name}_mean_over_seeds",
                "metric_value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "seed": -1,
                "artifact_path": str(metrics_path),
            }
        )

    lexical_word_mean = float(np.mean(mean_collectors["lexical_auc_word"]))
    lexical_char_mean = float(np.mean(mean_collectors["lexical_auc_char"]))
    lexical_word_eff_mean = float(np.mean(mean_collectors["lexical_auc_word_effective"]))
    lexical_char_eff_mean = float(np.mean(mean_collectors["lexical_auc_char_effective"]))
    delta_mean = float(np.mean(mean_collectors["delta_rho_mean"]))
    identity_rate_mean = float(np.mean(mean_collectors["identity_selection_rate"]))
    char_delta_mean = float(np.mean(mean_collectors["char_delta_from_base_mean"]))
    claim_promoted_mean = float(
        (lexical_word_eff_mean < 0.60)
        and (lexical_char_eff_mean < 0.60)
        and (delta_mean >= args.damage_threshold)
    )
    claim_all_seeds = float(all(v >= 1.0 for v in mean_collectors["claim_promoted_lexical_stealth"]))

    metric_rows.append(
        {
            "experiment_id": "adaptive_hispa_v4_stealth_v1",
            "model_id": args.model_id,
            "attacker_type": "adaptive_v4_stealth",
            "split": "balanced_internal",
            "n": int(args.n),
            "metric_name": "claim_promoted_lexical_stealth_mean_gate",
            "metric_value": claim_promoted_mean,
            "ci_low": "",
            "ci_high": "",
            "seed": -1,
            "artifact_path": str(metrics_path),
        }
    )
    metric_rows.append(
        {
            "experiment_id": "adaptive_hispa_v4_stealth_v1",
            "model_id": args.model_id,
            "attacker_type": "adaptive_v4_stealth",
            "split": "balanced_internal",
            "n": int(args.n),
            "metric_name": "claim_promoted_lexical_stealth_all_seeds",
            "metric_value": claim_all_seeds,
            "ci_low": "",
            "ci_high": "",
            "seed": -1,
            "artifact_path": str(metrics_path),
        }
    )

    summary_rows.append(
        {
            "seed": -1,
            "run_tag": run_tag,
            "surface_config": args.surface_config,
            "n_total": args.n,
            "lexical_auc_word": lexical_word_mean,
            "lexical_auc_char": lexical_char_mean,
            "lexical_auc_word_effective": lexical_word_eff_mean,
            "lexical_auc_char_effective": lexical_char_eff_mean,
            "delta_rho_mean": delta_mean,
            "identity_selection_rate": identity_rate_mean,
            "char_delta_from_base_mean": char_delta_mean,
            "damage_threshold": args.damage_threshold,
            "claim_promoted_lexical_stealth": int(claim_promoted_mean),
            "claim_promoted_all_seeds": int(claim_all_seeds),
            "detector_f1": float(np.mean(mean_collectors["detector_f1"])),
            "detector_auc": float(np.mean(mean_collectors["detector_auc"])),
        }
    )

    metrics_df = pd.DataFrame(metric_rows)[METRIC_SCHEMA_COLUMNS]
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)

    ledger_path = Path(args.output_dir) / "stealth_gate_runs.csv"
    ledger_row = pd.DataFrame(
        [
            {
                "run_tag": run_tag,
                "surface_config": args.surface_config,
                "model_id": args.model_id,
                "n_total": args.n,
                "seeds": args.seeds,
                "lexical_auc_word_mean": lexical_word_mean,
                "lexical_auc_char_mean": lexical_char_mean,
                "lexical_auc_word_effective_mean": lexical_word_eff_mean,
                "lexical_auc_char_effective_mean": lexical_char_eff_mean,
                "delta_rho_mean": delta_mean,
                "identity_selection_rate_mean": identity_rate_mean,
                "char_delta_from_base_mean": char_delta_mean,
                "damage_threshold": args.damage_threshold,
                "claim_promoted_lexical_stealth": int(claim_promoted_mean),
                "run_dir": str(run_dir),
            }
        ]
    )
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        if "run_tag" in existing.columns:
            existing = existing[existing["run_tag"].astype(str) != run_tag]
        merged = pd.concat([existing, ledger_row], ignore_index=True)
        merged.to_csv(ledger_path, index=False)
    else:
        ledger_row.to_csv(ledger_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved samples: {samples_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired prompts: {paired_path}")
    print(f"Saved run ledger: {ledger_path}")
    print(
        "Hard gate: "
        f"word_auc_raw={lexical_word_mean:.4f}, "
        f"char_auc_raw={lexical_char_mean:.4f}, "
        f"word_auc_effective={lexical_word_eff_mean:.4f}, "
        f"char_auc_effective={lexical_char_eff_mean:.4f}, "
        f"delta_rho_mean={delta_mean:.4f}, "
        f"identity_rate={identity_rate_mean:.4f}, "
        f"char_delta_mean={char_delta_mean:.4f}, "
        f"promoted={bool(claim_promoted_mean)}"
    )


if __name__ == "__main__":
    main()
