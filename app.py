import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from spectralguard import monitor as spectral_monitor

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None
    F = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


MODEL_ID = os.getenv("SPECTRALGUARD_MODEL_ID", "state-spaces/mamba-130m-hf")


@dataclass
class RuntimeState:
    tokenizer: Optional[object]
    model: Optional[object]
    device: Optional[object]
    ready: bool


def _deterministic_fallback(prompt: str, n_layers: int = 24) -> np.ndarray:
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.88, 0.97, size=n_layers)
    drift = rng.uniform(-0.08, 0.02, size=n_layers)
    return np.clip(baseline + drift, 0.2, 1.0).astype(np.float32)


def _load_model_once() -> RuntimeState:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return RuntimeState(tokenizer=None, model=None, device=None, ready=False)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        model.eval()
        return RuntimeState(tokenizer=tok, model=model, device=device, ready=True)
    except Exception:
        return RuntimeState(tokenizer=None, model=None, device=None, ready=False)


STATE = _load_model_once()


def _extract_layer_rho_real(prompt: str) -> np.ndarray:
    if not STATE.ready:
        raise RuntimeError("real_model mode is unavailable in current environment.")

    inputs = STATE.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(STATE.device)
    n_layers = len(STATE.model.backbone.layers)
    layer_deltas = {i: [] for i in range(n_layers)}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _args, output):
            t = output[0] if isinstance(output, tuple) else output
            layer_deltas[layer_idx].append(t.detach().float().cpu())

        return hook

    for i in range(n_layers):
        handles.append(STATE.model.backbone.layers[i].mixer.dt_proj.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        STATE.model(**inputs)

    for h in handles:
        h.remove()

    rho_layers = []
    for i in range(n_layers):
        mixer = STATE.model.backbone.layers[i].mixer
        A = -torch.exp(mixer.A_log.detach().float())
        if len(layer_deltas[i]) == 0:
            rho_layers.append(0.95)
            continue
        dt = F.softplus(layer_deltas[i][0]).unsqueeze(-1)
        A_bar = torch.exp(dt * A.cpu())
        rho = A_bar.max(dim=-1).values.mean(dim=-1)
        rho_layers.append(float(rho.min().item()))

    return np.asarray(rho_layers, dtype=np.float32)


def _extract_layer_rho(prompt: str, mode: str) -> Tuple[np.ndarray, str]:
    requested_mode = mode if mode in {"real_model", "demo_mode"} else "demo_mode"
    if requested_mode == "real_model":
        if STATE.ready:
            return _extract_layer_rho_real(prompt), "real_model"
        return _deterministic_fallback(prompt), "demo_mode (fallback: model unavailable)"
    return _deterministic_fallback(prompt), "demo_mode"


def analyze_prompt(prompt: str, mode: str):
    if not prompt or not prompt.strip():
        return "Enter a prompt.", None, None

    rho, used_mode = _extract_layer_rho(prompt, mode)
    is_safe, hazard = spectral_monitor(prompt, {"rho_layers": rho.tolist()})

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(rho))
    ax.plot(x, rho, marker="o", linewidth=1.5)
    ax.axhline(0.90, color="orange", linestyle="--", linewidth=1.0, label="critical band")
    ax.set_xlabel("Layer")
    ax.set_ylabel("min token rho")
    ax.set_title("Layer-wise spectral trace")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    summary = (
        f"Mode: {used_mode}\n"
        f"Safety: {'PASS' if is_safe else 'BLOCK'}\n"
        f"Spectral hazard score: {hazard:.3f}\n"
        f"Mean rho: {np.mean(rho):.3f} | Std rho: {np.std(rho):.3f}"
    )

    table = [[int(i), float(v)] for i, v in enumerate(rho)]
    return summary, fig, table


with gr.Blocks(title="SpectralGuard Demo") as demo:
    gr.Markdown("# SpectralGuard Interactive Demo")
    gr.Markdown("Inspect spectral traces using either real model extraction or a declared demo fallback.")

    prompt = gr.Textbox(lines=4, label="Prompt", placeholder="Ask a question or paste a candidate prompt...")
    mode = gr.Radio(
        choices=["demo_mode", "real_model"],
        value="demo_mode",
        label="Execution mode",
        info="Use real_model when weights/runtime are available; otherwise demo_mode is deterministic fallback.",
    )
    run_btn = gr.Button("Analyze")

    summary = gr.Textbox(label="Decision")
    plot = gr.Plot(label="Layer-wise rho")
    table = gr.Dataframe(headers=["layer_idx", "rho_min"], datatype=["number", "number"], label="Trace values")

    run_btn.click(analyze_prompt, inputs=[prompt, mode], outputs=[summary, plot, table])


if __name__ == "__main__":
    demo.launch()
