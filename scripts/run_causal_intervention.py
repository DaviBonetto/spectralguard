import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_ar_prompt(rng: random.Random):
    words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elder",
        "fig",
        "grape",
        "honey",
        "kiwi",
        "lemon",
        "mango",
        "nut",
        "orange",
        "pear",
        "plum",
        "rose",
        "star",
        "tree",
        "vine",
        "wood",
    ]
    rng.shuffle(words)
    seq = words[:10]
    prefix = " ".join(seq)
    target = seq[-1]
    repeated_prefix = " ".join(seq[:-1]) + " "
    prompt = f"{prefix} . {prefix} . {prefix} . {repeated_prefix}"
    return prompt, target


def tie_lm_head(model):
    if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
        model.lm_head.weight = model.get_input_embeddings().weight


def apply_clamp_hooks(model, layer_indices, target_rho, device):
    handles = []

    for idx in layer_indices:
        mixer = model.backbone.layers[idx].mixer
        A = -torch.exp(mixer.A_log.detach().float()).to(device)
        min_abs_A = torch.abs(A).min(dim=-1).values
        min_dt_required = -math.log(target_rho) / torch.clamp(min_abs_A, min=1e-8)

        def make_hook(min_dt_local):
            def hook(module, args, output):
                dt_raw = output[0] if isinstance(output, tuple) else output
                dt = F.softplus(dt_raw)
                dt_clamped = torch.max(
                    dt,
                    min_dt_local.unsqueeze(0).unsqueeze(0)
                    .to(dt.device)
                    .to(dt.dtype),
                )
                dt_raw_clamped = dt_clamped + torch.log1p(
                    -torch.exp(-dt_clamped.float())
                ).to(dt_clamped.dtype)

                if isinstance(output, tuple):
                    return (dt_raw_clamped,) + output[1:]
                return dt_raw_clamped

            return hook

        handles.append(mixer.dt_proj.register_forward_hook(make_hook(min_dt_required)))

    return handles


def evaluate_accuracy(model, tokenizer, prompts, target_rho, protocol, layer_idx, device, max_new_tokens):
    if protocol == "all_layer":
        layers = list(range(len(model.backbone.layers)))
    elif protocol == "single_layer":
        layers = [layer_idx]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    handles = apply_clamp_hooks(model, layers, target_rho, device)

    correct = 0
    try:
        for prompt, target in tqdm(prompts, desc=f"{protocol} rho={target_rho:.2f}"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            if target in decoded:
                correct += 1
    finally:
        for h in handles:
            h.remove()

    n = len(prompts)
    acc = correct / n if n > 0 else 0.0
    ci_low, ci_high = wilson_interval(correct, n)
    return acc, ci_low, ci_high, correct


def mine_validated_prompts(model, tokenizer, n_samples, device, seed, max_attempts, max_new_tokens):
    rng = random.Random(seed)
    validated = []
    attempts = 0

    pbar = tqdm(total=n_samples, desc="Mining solvable prompts")
    while len(validated) < n_samples and attempts < max_attempts:
        attempts += 1
        prompt, target = generate_ar_prompt(rng)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if target in decoded:
            validated.append((prompt, target))
            pbar.update(1)
    pbar.close()

    if len(validated) < n_samples:
        print(f"Warning: collected {len(validated)} validated prompts (target={n_samples}).")

    return validated


def main():
    parser = argparse.ArgumentParser(description="Causal intervention for spectral clamp protocols.")
    parser.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--protocol",
        choices=["single_layer", "all_layer", "both"],
        default="both",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.99, 0.95, 0.90, 0.70, 0.50],
    )
    parser.add_argument("--max-attempts", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="causal_intervention_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, local_files_only=args.local_files_only).to(device)
    tie_lm_head(model)
    model.eval()

    prompts = mine_validated_prompts(
        model,
        tokenizer,
        args.n_samples,
        device,
        args.seed,
        args.max_attempts,
        args.max_new_tokens,
    )

    rows = []
    run_protocols = [args.protocol] if args.protocol != "both" else ["single_layer", "all_layer"]

    for protocol in run_protocols:
        n = len(prompts)
        rows.append(
            {
                "experiment_id": "causal_clamp",
                "model_id": args.model_id,
                "attacker_type": "none",
                "split": "validated_ar",
                "n": n,
                "metric_name": "accuracy",
                "metric_value": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
                "seed": args.seed,
                "artifact_path": args.output,
                "protocol": protocol,
                "rho_target": "baseline",
                "correct": n,
            }
        )

        for rho in args.thresholds:
            acc, ci_low, ci_high, correct = evaluate_accuracy(
                model,
                tokenizer,
                prompts,
                rho,
                protocol,
                args.layer_idx,
                device,
                args.max_new_tokens,
            )
            rows.append(
                {
                    "experiment_id": "causal_clamp",
                    "model_id": args.model_id,
                    "attacker_type": "none",
                    "split": "validated_ar",
                    "n": n,
                    "metric_name": "accuracy",
                    "metric_value": round(acc, 6),
                    "ci_low": round(ci_low, 6),
                    "ci_high": round(ci_high, 6),
                    "seed": args.seed,
                    "artifact_path": args.output,
                    "protocol": protocol,
                    "rho_target": rho,
                    "correct": correct,
                }
            )

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
