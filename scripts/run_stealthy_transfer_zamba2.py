import argparse
import random
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
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mamba_spectral.utils.datasets import load_safe_prompts


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize_across_seeds(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    lo, hi = np.percentile(arr, [2.5, 97.5]).astype(float)
    return mean, float(lo), float(hi)


def _balanced_from_labeled_csv(
    df: pd.DataFrame,
    n: int,
    seed: int,
    prompt_col: str,
    label_col: str,
) -> Tuple[List[str], np.ndarray]:
    tmp = df[[prompt_col, label_col]].copy()
    tmp = tmp.dropna()
    tmp[label_col] = tmp[label_col].astype(int)
    tmp = tmp[tmp[label_col].isin([0, 1])].reset_index(drop=True)
    if tmp.empty:
        raise ValueError("Input CSV has no valid labeled rows.")

    n0 = int((tmp[label_col] == 0).sum())
    n1 = int((tmp[label_col] == 1).sum())
    k = min(n // 2, n0, n1)
    if k < 2:
        raise ValueError("Need at least two rows per class for balanced split.")

    benign = tmp[tmp[label_col] == 0].sample(n=k, random_state=seed)
    adv = tmp[tmp[label_col] == 1].sample(n=k, random_state=seed + 1)
    out = pd.concat([benign, adv], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    prompts = out[prompt_col].astype(str).tolist()
    labels = out[label_col].astype(int).to_numpy()
    return prompts, labels


def load_transfer_prompts(
    prompt_csv: str,
    n: int,
    seed: int,
    prompt_col: str,
    label_col: str,
    attacker_type: str,
) -> Tuple[List[str], np.ndarray]:
    path = Path(prompt_csv)
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Prompt CSV is empty.")

    if prompt_col in df.columns and label_col in df.columns:
        return _balanced_from_labeled_csv(df, n=n, seed=seed, prompt_col=prompt_col, label_col=label_col)

    if prompt_col not in df.columns:
        raise ValueError(f"Prompt column '{prompt_col}' not found.")

    adv_df = df.copy()
    if "attacker_type" in adv_df.columns:
        adv_df = adv_df[adv_df["attacker_type"] == attacker_type].copy()
    if "label" in adv_df.columns:
        adv_df = adv_df[adv_df["label"] == 1].copy()
    adv_df = adv_df.dropna(subset=[prompt_col])
    if adv_df.empty:
        raise ValueError("No adversarial prompts available for transfer input.")

    k = min(n // 2, int(adv_df.shape[0]))
    if k < 2:
        raise ValueError("Not enough adversarial prompts for balanced run.")

    adv_prompts = adv_df[prompt_col].sample(n=k, random_state=seed).astype(str).tolist()
    random.seed(seed)
    benign_prompts = load_safe_prompts(n=k)
    prompts = benign_prompts + adv_prompts
    labels = np.asarray(([0] * k) + ([1] * k), dtype=int)
    return prompts, labels


def setup_zamba_model(model_id: str, local_files_only: bool, device: torch.device):
    tokenizer = None
    tok_errors: List[str] = []
    for kwargs in (
        {"use_fast": True, "trust_remote_code": True},
        {"use_fast": False, "trust_remote_code": True},
        {"use_fast": False, "trust_remote_code": False},
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                **kwargs,
            )
            break
        except Exception as exc:
            tok_errors.append(f"{kwargs}: {exc}")
    if tokenizer is None:
        joined = " | ".join(tok_errors)
        raise RuntimeError(
            f"Failed to load tokenizer for {model_id}. "
            "Install tokenizer dependencies (sentencepiece/tiktoken) and ensure model files are cached. "
            f"Attempts: {joined}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Compatibility guard for some Zamba2 builds.
    if hasattr(model, "model") and not hasattr(model.model, "rotary_emb"):
        try:
            from transformers.models.zamba2.modeling_zamba2 import Zamba2RotaryEmbedding

            model.model.rotary_emb = Zamba2RotaryEmbedding(model.config, device=device)
        except Exception:
            pass

    return model, tokenizer


def find_zamba_mixers(model) -> List[Tuple[str, object]]:
    mixers: List[Tuple[str, object]] = []
    for name, module in model.named_modules():
        if hasattr(module, "A_log") and hasattr(module, "in_proj") and hasattr(module, "dt_bias"):
            mixers.append((name, module))
    if not mixers:
        raise RuntimeError(
            "No Zamba2-style Mamba mixers found. Expected modules with A_log, in_proj, and dt_bias."
        )
    return mixers


def _extract_dt_raw(projected: torch.Tensor, mixer) -> torch.Tensor:
    num_heads = int(getattr(mixer, "num_heads", projected.shape[-1]))
    p = projected.shape[-1]
    intermediate_size = int(getattr(mixer, "intermediate_size", 0))
    n_groups = int(getattr(mixer, "n_groups", 1))
    ssm_state_size = int(getattr(mixer, "ssm_state_size", 1))
    d_to_remove = 2 * intermediate_size + 2 * n_groups * ssm_state_size + num_heads
    d_mlp = (p - d_to_remove) // 2
    if d_mlp >= 0 and (d_mlp * 2 + intermediate_size + (2 * n_groups * ssm_state_size) + num_heads) <= p:
        return projected[..., -num_heads:]
    return projected[..., -num_heads:]


def extract_zamba_spectral_features(
    model,
    tokenizer,
    mixers: List[Tuple[str, object]],
    prompt: str,
    device: torch.device,
    max_length: int,
    collapse_threshold: float,
) -> Dict[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    projected_states: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def _hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            projected_states[name] = tensor.detach().float().cpu()

        return _hook

    for name, mixer in mixers:
        handles.append(mixer.in_proj.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    layer_rhos: List[float] = []
    for name, mixer in mixers:
        proj = projected_states.get(name)
        if proj is None or proj.ndim < 3:
            continue

        dt_raw = _extract_dt_raw(proj, mixer)
        dt_bias = mixer.dt_bias.detach().float().cpu().view(1, 1, -1)

        heads = min(dt_raw.shape[-1], dt_bias.shape[-1])
        if heads <= 0:
            continue
        dt_raw = dt_raw[..., :heads]
        dt_bias = dt_bias[..., :heads]
        dt = F.softplus(dt_raw + dt_bias)

        a = -torch.exp(mixer.A_log.detach().float().cpu())
        if a.ndim == 1:
            a_head = a.view(1, 1, -1)
        else:
            a_head = a.view(a.shape[0], -1).mean(dim=-1).view(1, 1, -1)
        a_head = a_head[..., :heads]

        rho_tokens = torch.exp(dt * a_head)
        layer_rho = float(rho_tokens.mean(dim=-1).min().item())
        layer_rhos.append(layer_rho)

    if not layer_rhos:
        layer_rhos = [0.95]

    rho = np.asarray(layer_rhos, dtype=np.float32)
    return {
        "rho_mean": float(np.mean(rho)),
        "sigma_rho": float(np.std(rho)),
        "rho_gap": float(np.max(rho) - np.min(rho)),
        "collapse_depth": float(np.sum(rho < collapse_threshold)),
        "n_layers": int(len(layer_rhos)),
    }


def eval_spectral_detector(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)

    y_prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
    y_pred = clf.predict(scaler.transform(x_test))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }
    return metrics, test_idx, y_pred, y_prob


def eval_lexical_auc(prompts: List[str], labels: np.ndarray, seed: int, test_size: float) -> float:
    idx = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    vect = CountVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = vect.fit_transform([prompts[i] for i in train_idx])
    x_test = vect.transform([prompts[i] for i in test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-architecture stealth transfer test (AdaptiveHiSPA v4 -> Zamba2)."
    )
    parser.add_argument("--model-id", default="Zyphra/Zamba2-2.7B")
    parser.add_argument("--prompt-csv", default="artifacts/adaptive_v4/adaptive_v4_samples.csv")
    parser.add_argument("--prompt-col", default="prompt_text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--attacker-type", default="adaptive_v4_stealth")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/zamba2_v4_transfer")
    args = parser.parse_args()

    seed_values = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not seed_values:
        raise ValueError("No seeds provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, tokenizer = setup_zamba_model(args.model_id, local_files_only=args.local_files_only, device=device)
    mixers = find_zamba_mixers(model)
    print(f"Discovered {len(mixers)} Zamba2 Mamba mixers.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "zamba2_transfer_metrics.csv"

    metric_rows = []
    sample_rows = []
    seed_summary = []
    auc_values = []
    lexical_auc_values = []

    for seed in seed_values:
        set_seed(seed)
        prompts, labels = load_transfer_prompts(
            prompt_csv=args.prompt_csv,
            n=args.n,
            seed=seed,
            prompt_col=args.prompt_col,
            label_col=args.label_col,
            attacker_type=args.attacker_type,
        )

        feat_rows = []
        for prompt in prompts:
            feat = extract_zamba_spectral_features(
                model=model,
                tokenizer=tokenizer,
                mixers=mixers,
                prompt=prompt,
                device=device,
                max_length=args.max_length,
                collapse_threshold=args.collapse_threshold,
            )
            feat_rows.append(feat)

        feat_df = pd.DataFrame(feat_rows)
        x = feat_df[["rho_mean", "sigma_rho", "rho_gap", "collapse_depth"]].to_numpy(dtype=np.float32)
        det_metrics, test_idx, y_pred, y_prob = eval_spectral_detector(
            features=x,
            labels=labels,
            seed=seed,
            test_size=args.test_size,
        )
        lexical_auc = eval_lexical_auc(prompts, labels, seed=seed, test_size=args.test_size)

        benign_mean = float(feat_df.iloc[np.where(labels == 0)[0]]["rho_mean"].mean())
        adv_mean = float(feat_df.iloc[np.where(labels == 1)[0]]["rho_mean"].mean())
        delta_rho = float(benign_mean - adv_mean)
        auc_values.append(det_metrics["auc"])
        lexical_auc_values.append(lexical_auc)

        metric_pairs = {
            "spectral_transfer_accuracy": det_metrics["accuracy"],
            "spectral_transfer_precision": det_metrics["precision"],
            "spectral_transfer_recall": det_metrics["recall"],
            "spectral_transfer_f1": det_metrics["f1"],
            "spectral_transfer_auc": det_metrics["auc"],
            "prompt_text_lexical_auc": lexical_auc,
            "rho_mean_benign": benign_mean,
            "rho_mean_adversarial": adv_mean,
            "delta_rho_mean": delta_rho,
            "n_mixers_detected": float(len(mixers)),
        }
        for metric_name, metric_value in metric_pairs.items():
            metric_rows.append(
                {
                    "experiment_id": "zamba2_transfer_v4_v1",
                    "model_id": args.model_id,
                    "attacker_type": args.attacker_type,
                    "split": "test",
                    "n": int(test_idx.shape[0]),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "ci_low": "",
                    "ci_high": "",
                    "seed": int(seed),
                    "artifact_path": str(metrics_path),
                }
            )

        for local_pos, idx in enumerate(test_idx.tolist()):
            sample_rows.append(
                {
                    "seed": seed,
                    "prompt_id": f"zamba2_s{seed}_{idx}",
                    "prompt_text": prompts[idx],
                    "label": int(labels[idx]),
                    "pred": int(y_pred[local_pos]),
                    "prob_adv": float(y_prob[local_pos]),
                    "rho_mean": float(feat_df.loc[idx, "rho_mean"]),
                    "sigma_rho": float(feat_df.loc[idx, "sigma_rho"]),
                    "rho_gap": float(feat_df.loc[idx, "rho_gap"]),
                    "collapse_depth": float(feat_df.loc[idx, "collapse_depth"]),
                    "split": "test",
                }
            )

        seed_summary.append(
            {
                "seed": seed,
                "model_id": args.model_id,
                "attacker_type": args.attacker_type,
                "n_total": int(len(prompts)),
                "spectral_auc": float(det_metrics["auc"]),
                "spectral_f1": float(det_metrics["f1"]),
                "lexical_auc": float(lexical_auc),
                "rho_mean_benign": benign_mean,
                "rho_mean_adversarial": adv_mean,
                "delta_rho_mean": delta_rho,
                "n_mixers_detected": int(len(mixers)),
            }
        )

    for metric_name in [
        "spectral_transfer_accuracy",
        "spectral_transfer_precision",
        "spectral_transfer_recall",
        "spectral_transfer_f1",
        "spectral_transfer_auc",
        "prompt_text_lexical_auc",
        "delta_rho_mean",
    ]:
        vals = [row["metric_value"] for row in metric_rows if row["metric_name"] == metric_name]
        if not vals:
            continue
        mean, lo, hi = summarize_across_seeds(vals)
        metric_rows.append(
            {
                "experiment_id": "zamba2_transfer_v4_v1",
                "model_id": args.model_id,
                "attacker_type": args.attacker_type,
                "split": "test",
                "n": int(args.n),
                "metric_name": f"{metric_name}_mean_over_seeds",
                "metric_value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "seed": -1,
                "artifact_path": str(metrics_path),
            }
        )

    transfer_detectable = bool(auc_values and min(auc_values) > 0.60)
    metric_rows.append(
        {
            "experiment_id": "zamba2_transfer_v4_v1",
            "model_id": args.model_id,
            "attacker_type": args.attacker_type,
            "split": "test",
            "n": int(args.n),
            "metric_name": "claim_transfer_detectable_auc_gt_0_60_consistent",
            "metric_value": 1.0 if transfer_detectable else 0.0,
            "ci_low": "",
            "ci_high": "",
            "seed": -1,
            "artifact_path": str(metrics_path),
        }
    )

    metrics_df = pd.DataFrame(metric_rows)[METRIC_SCHEMA_COLUMNS]
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(sample_rows).to_csv(out_dir / "zamba2_transfer_samples.csv", index=False)

    summary_df = pd.DataFrame(seed_summary)
    summary_df.to_csv(out_dir / "zamba2_transfer_summary.csv", index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved samples: {out_dir / 'zamba2_transfer_samples.csv'}")
    print(f"Saved summary: {out_dir / 'zamba2_transfer_summary.csv'}")
    print(f"Transfer detectable (AUC>0.60 all seeds): {transfer_detectable}")


if __name__ == "__main__":
    main()
