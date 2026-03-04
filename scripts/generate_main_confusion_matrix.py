import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def build_df_from_counts(tp: int, tn: int, fp: int, fn: int) -> pd.DataFrame:
    rows = []
    rows.extend([{"label": 1, "pred": 1}] * int(tp))
    rows.extend([{"label": 0, "pred": 0}] * int(tn))
    rows.extend([{"label": 0, "pred": 1}] * int(fp))
    rows.extend([{"label": 1, "pred": 0}] * int(fn))
    return pd.DataFrame(rows)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / float(n_pos * n_neg)


def compute_metrics(df: pd.DataFrame, label_col: str, pred_col: str, prob_col: Optional[str]) -> dict:
    y_true = df[label_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_counts(y_true=y_true, y_pred=y_pred)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / y_true.shape[0]) if y_true.shape[0] > 0 else float("nan")
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(df.shape[0]),
    }
    if prob_col and prob_col in df.columns:
        try:
            metrics["auc"] = float(binary_auc(y_true=y_true, y_score=df[prob_col].astype(float).to_numpy()))
        except Exception:
            metrics["auc"] = float("nan")
    return metrics


def save_confusion_image(metrics: dict, output_image: Path) -> None:
    cm = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Predicted Safe", "Predicted Attack"])
    ax.set_yticks([0, 1], labels=["Actual Safe", "Actual Attack"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("SpectralGuard Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=12)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_image, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate canonical Figure 4 confusion matrix for the main N=500 protocol."
    )
    parser.add_argument("--predictions-csv", default="", help="Optional path with columns label/pred (and optionally prob).")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--prob-col", default="")
    parser.add_argument("--tp", type=int, default=245)
    parser.add_argument("--tn", type=int, default=235)
    parser.add_argument("--fp", type=int, default=15)
    parser.add_argument("--fn", type=int, default=5)
    parser.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--attacker-type", default="non_adaptive_main_protocol")
    parser.add_argument("--output-image", default="mamba_spectral/Estado_atual/NIXEtm04.png")
    parser.add_argument("--output-predictions", default="artifacts/main_defense/main_defense_predictions.csv")
    parser.add_argument("--output-metrics", default="artifacts/main_defense/main_defense_metrics.csv")
    args = parser.parse_args()

    if args.predictions_csv:
        pred_path = Path(args.predictions_csv)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        df = pd.read_csv(pred_path)
    else:
        df = build_df_from_counts(tp=args.tp, tn=args.tn, fp=args.fp, fn=args.fn)

    if args.label_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(f"Missing required columns. Need '{args.label_col}' and '{args.pred_col}'.")

    prob_col = args.prob_col if args.prob_col else None
    metrics = compute_metrics(df, label_col=args.label_col, pred_col=args.pred_col, prob_col=prob_col)

    output_predictions = Path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_predictions, index=False)

    output_image = Path(args.output_image)
    save_confusion_image(metrics, output_image=output_image)

    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]:
        if key not in metrics:
            continue
        rows.append(
            {
                "experiment_id": "main_defense_nonadaptive_v1",
                "model_id": args.model_id,
                "attacker_type": args.attacker_type,
                "split": "balanced_main",
                "n": metrics["n"],
                "metric_name": key,
                "metric_value": metrics[key],
                "ci_low": "",
                "ci_high": "",
                "seed": -1,
                "artifact_path": str(output_metrics),
            }
        )
    for key in ["tp", "tn", "fp", "fn"]:
        rows.append(
            {
                "experiment_id": "main_defense_nonadaptive_v1",
                "model_id": args.model_id,
                "attacker_type": args.attacker_type,
                "split": "balanced_main",
                "n": metrics["n"],
                "metric_name": key,
                "metric_value": float(metrics[key]),
                "ci_low": "",
                "ci_high": "",
                "seed": -1,
                "artifact_path": str(output_metrics),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df = out_df[METRIC_SCHEMA_COLUMNS]
    out_df.to_csv(output_metrics, index=False)

    print(f"Saved image: {output_image}")
    print(f"Saved predictions: {output_predictions}")
    print(f"Saved metrics: {output_metrics}")
    print(
        "Main metrics: "
        f"Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, "
        f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, FPR={metrics['fpr']:.3f}"
    )


if __name__ == "__main__":
    main()
