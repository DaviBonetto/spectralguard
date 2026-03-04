import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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


def parse_seeds(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one seed is required.")
    return vals


def ensure_split_columns(df: pd.DataFrame, seed_values: Sequence[int], test_size: float) -> pd.DataFrame:
    out = df.copy()
    if "seed" not in out.columns:
        parts = []
        for s in seed_values:
            tmp = out.copy()
            tmp["seed"] = s
            parts.append(tmp)
        out = pd.concat(parts, ignore_index=True)

    if "split" in out.columns:
        out["split"] = out["split"].astype(str).str.lower()
        return out

    out["split"] = "train"
    for s in sorted(out["seed"].unique()):
        idx = out.index[out["seed"] == s].to_numpy()
        if idx.shape[0] < 10:
            continue
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=int(s))
        out.loc[train_idx, "split"] = "train"
        out.loc[test_idx, "split"] = "test"
    return out


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    n_boot: int,
    r2_floor: float,
    r2_ceil: float,
) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    vals = []
    max_trials = max(n_boot * 10, 1000)
    trials = 0
    while len(vals) < n_boot and trials < max_trials:
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        if float(np.var(y_bs)) < 1e-10:
            trials += 1
            continue
        v = float(r2_score(y_bs, y_pred[idx]))
        if np.isnan(v) or np.isinf(v):
            trials += 1
            continue
        vals.append(float(np.clip(v, r2_floor, r2_ceil)))
        trials += 1

    if not vals:
        vals = [r2_floor]
    arr = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5]).astype(float)
    return float(lo), float(hi), arr


def fit_seed_models(seed_df: pd.DataFrame, target_col: str) -> Dict[str, object]:
    train_df = seed_df[seed_df["split"] == "train"].copy()
    test_df = seed_df[seed_df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Each seed partition must contain both train and test rows.")

    if target_col not in seed_df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if float(np.var(train_df[target_col].to_numpy(dtype=float))) < 1e-10:
        raise ValueError(f"Target variance is near zero in train split for target_col={target_col}.")

    uni_cols = ["rho_mean"]
    multi_cols = ["rho_mean", "sigma_rho", "eig_gap", "collapse_depth"]
    multi_cols = [c for c in multi_cols if c in seed_df.columns]
    if len(multi_cols) < 3:
        raise ValueError("Missing multivariate columns: need rho_mean, sigma_rho, eig_gap at least.")

    uni_model = LinearRegression().fit(train_df[uni_cols].to_numpy(dtype=float), train_df[target_col].to_numpy(dtype=float))
    multi_model = LinearRegression().fit(
        train_df[multi_cols].to_numpy(dtype=float),
        train_df[target_col].to_numpy(dtype=float),
    )

    y_test = test_df[target_col].to_numpy(dtype=float)
    y_uni = uni_model.predict(test_df[uni_cols].to_numpy(dtype=float))
    y_multi = multi_model.predict(test_df[multi_cols].to_numpy(dtype=float))

    return {
        "train_n": int(train_df.shape[0]),
        "test_n": int(test_df.shape[0]),
        "uni_cols": uni_cols,
        "multi_cols": multi_cols,
        "uni_model": uni_model,
        "multi_model": multi_model,
        "y_test": y_test,
        "y_uni": y_uni,
        "y_multi": y_multi,
        "r2_uni": float(r2_score(y_test, y_uni)),
        "r2_multi": float(r2_score(y_test, y_multi)),
    }


def ci_from_values(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5]).astype(float)
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="Rigorous multi-seed train/test R2 analysis")
    parser.add_argument("--input", default="artifacts/multilayer_regression/features_rigorous.csv")
    parser.add_argument("--output-dir", default="artifacts/multilayer_regression")
    parser.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--target-col", default="target_signal")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--r2-floor", type=float, default=-5.0)
    parser.add_argument("--r2-ceil", type=float, default=1.0)
    parser.add_argument("--target-r2", type=float, default=0.20)
    parser.add_argument("--target-ci-low", type=float, default=0.10)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(in_path)
    required = ["rho_mean", "sigma_rho", "eig_gap"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seed_values = parse_seeds(args.seeds)
    df = ensure_split_columns(raw, seed_values, args.test_size)
    df = df[df["seed"].isin(seed_values)].copy()
    if df.empty:
        raise RuntimeError("No rows available after seed filtering.")

    metrics_rows = []
    coef_rows = []

    all_seed_r2_uni = []
    all_seed_r2_multi = []
    all_boot_uni = []
    all_boot_multi = []

    metrics_path = out_dir / "multilayer_regression_rigorous_metrics.csv"

    for seed in seed_values:
        seed_df = df[df["seed"] == seed].copy()
        if seed_df.empty:
            continue
        res = fit_seed_models(seed_df, target_col=args.target_col)

        lo_uni, hi_uni, boot_uni = bootstrap_r2(
            res["y_test"],
            res["y_uni"],
            seed=seed,
            n_boot=args.n_boot,
            r2_floor=args.r2_floor,
            r2_ceil=args.r2_ceil,
        )
        lo_multi, hi_multi, boot_multi = bootstrap_r2(
            res["y_test"],
            res["y_multi"],
            seed=seed + 10_000,
            n_boot=args.n_boot,
            r2_floor=args.r2_floor,
            r2_ceil=args.r2_ceil,
        )

        all_seed_r2_uni.append(res["r2_uni"])
        all_seed_r2_multi.append(res["r2_multi"])
        all_boot_uni.extend(boot_uni.tolist())
        all_boot_multi.extend(boot_multi.tolist())

        metrics_rows.extend(
            [
                {
                    "experiment_id": "multilayer_regression_rigorous_v1",
                    "model_id": args.model_id,
                    "attacker_type": "mixed_regimes",
                    "split": "test",
                    "n": res["test_n"],
                    "metric_name": "r2_univariate_test",
                    "metric_value": res["r2_uni"],
                    "ci_low": lo_uni,
                    "ci_high": hi_uni,
                    "seed": seed,
                    "artifact_path": str(metrics_path),
                },
                {
                    "experiment_id": "multilayer_regression_rigorous_v1",
                    "model_id": args.model_id,
                    "attacker_type": "mixed_regimes",
                    "split": "test",
                    "n": res["test_n"],
                    "metric_name": "r2_multivariate_test",
                    "metric_value": res["r2_multi"],
                    "ci_low": lo_multi,
                    "ci_high": hi_multi,
                    "seed": seed,
                    "artifact_path": str(metrics_path),
                },
                {
                    "experiment_id": "multilayer_regression_rigorous_v1",
                    "model_id": args.model_id,
                    "attacker_type": "mixed_regimes",
                    "split": "test",
                    "n": res["test_n"],
                    "metric_name": "delta_r2_multi_minus_uni",
                    "metric_value": float(res["r2_multi"] - res["r2_uni"]),
                    "ci_low": "",
                    "ci_high": "",
                    "seed": seed,
                    "artifact_path": str(metrics_path),
                },
            ]
        )

        for feat, coef in zip(res["multi_cols"], res["multi_model"].coef_.tolist()):
            coef_rows.append(
                {
                    "seed": seed,
                    "feature": feat,
                    "coefficient": float(coef),
                }
            )

    if not all_seed_r2_multi:
        raise RuntimeError("No valid seed partitions were evaluated.")

    mean_uni = float(np.mean(all_seed_r2_uni))
    mean_multi = float(np.mean(all_seed_r2_multi))
    ci_uni_lo, ci_uni_hi = ci_from_values(all_boot_uni)
    ci_multi_lo, ci_multi_hi = ci_from_values(all_boot_multi)
    claim_promoted = bool(mean_multi >= args.target_r2 and ci_multi_lo >= args.target_ci_low)

    total_test_n = int(
        sum(row["n"] for row in metrics_rows if row["metric_name"] == "r2_multivariate_test")
    )

    metrics_rows.extend(
        [
            {
                "experiment_id": "multilayer_regression_rigorous_v1",
                "model_id": args.model_id,
                "attacker_type": "mixed_regimes",
                "split": "test",
                "n": total_test_n,
                "metric_name": "r2_univariate_test_mean_over_seeds",
                "metric_value": mean_uni,
                "ci_low": ci_uni_lo,
                "ci_high": ci_uni_hi,
                "seed": -1,
                "artifact_path": str(metrics_path),
            },
            {
                "experiment_id": "multilayer_regression_rigorous_v1",
                "model_id": args.model_id,
                "attacker_type": "mixed_regimes",
                "split": "test",
                "n": total_test_n,
                "metric_name": "r2_multivariate_test_mean_over_seeds",
                "metric_value": mean_multi,
                "ci_low": ci_multi_lo,
                "ci_high": ci_multi_hi,
                "seed": -1,
                "artifact_path": str(metrics_path),
            },
            {
                "experiment_id": "multilayer_regression_rigorous_v1",
                "model_id": args.model_id,
                "attacker_type": "mixed_regimes",
                "split": "test",
                "n": total_test_n,
                "metric_name": "claim_promoted_r2_threshold",
                "metric_value": 1.0 if claim_promoted else 0.0,
                "ci_low": "",
                "ci_high": "",
                "seed": -1,
                "artifact_path": str(metrics_path),
            },
        ]
    )

    metrics_df = pd.DataFrame(metrics_rows)[METRIC_SCHEMA_COLUMNS]
    metrics_df.to_csv(metrics_path, index=False)

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(out_dir / "multilayer_regression_rigorous_coefficients.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "target_col": args.target_col,
                "mean_r2_univariate_test": mean_uni,
                "mean_r2_multivariate_test": mean_multi,
                "ci_low_multivariate": ci_multi_lo,
                "ci_high_multivariate": ci_multi_hi,
                "target_r2": float(args.target_r2),
                "target_ci_low": float(args.target_ci_low),
                "claim_promoted": claim_promoted,
                "seeds": ",".join(str(s) for s in seed_values),
                "input_path": str(in_path),
            }
        ]
    )
    summary.to_csv(out_dir / "multilayer_regression_rigorous_summary.csv", index=False)

    print(f"Rigorous multivariate R2(test) mean over seeds: {mean_multi:.4f}")
    print(f"Rigorous multivariate bootstrap 95% CI: [{ci_multi_lo:.4f}, {ci_multi_hi:.4f}]")
    print(f"Claim promoted: {claim_promoted}")


if __name__ == "__main__":
    main()
