import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_id = 'state-spaces/mamba-130m-hf'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

num_layers = len(model.backbone.layers)
batch_sizes = [1, 4, 16, 32]
seq_len = 50
n_warmup = 1
n_runs = 3

results_baseline = []
results_guarded = []

def run_baseline(batch_size):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time() - start

def run_guarded(batch_size):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
    
    handles = []
    layer_deltas = {i: [] for i in range(num_layers)}
    
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            layer_deltas[layer_idx].append(output[0].detach())
        return hook_fn
        
    for i in range(num_layers):
        layer = model.backbone.layers[i].mixer
        handles.append(layer.dt_proj.register_forward_hook(get_hook(i)))
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        _ = model(input_ids)
        
        # Simulate extraction overhead (the actual defense logic)
        for i in range(num_layers):
            if len(layer_deltas[i]) > 0:
                dt_raw = layer_deltas[i][0]
                A = -torch.exp(model.backbone.layers[i].mixer.A_log.detach().float())
                dt = F.softplus(dt_raw).unsqueeze(-1)
                A_bar = torch.exp(dt * A)
                rho = A_bar.max(dim=-1).values.mean(dim=-1)
                
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    for handle in handles:
        handle.remove()
        
    return end - start

print("\nStarting Benchmark...")
for bs in batch_sizes:
    print(f"\nBatch Size: {bs}")
    
    # Warmup Baseline
    for _ in range(n_warmup):
        run_baseline(bs)
        
    # Measure Baseline
    times_b = []
    for _ in range(n_runs):
        times_b.append(run_baseline(bs))
    avg_b = np.mean(times_b)
    tokens_per_sec_b = (bs * seq_len) / avg_b
    results_baseline.append((avg_b * 1000, tokens_per_sec_b)) # ms and t/s
    print(f"  Baseline: {avg_b*1000:.2f} ms | {tokens_per_sec_b:.2f} tokens/sec")

    # Warmup Guarded
    for _ in range(n_warmup):
        run_guarded(bs)
        
    # Measure Guarded
    times_g = []
    for _ in range(n_runs):
        times_g.append(run_guarded(bs))
    avg_g = np.mean(times_g)
    tokens_per_sec_g = (bs * seq_len) / avg_g
    results_guarded.append((avg_g * 1000, tokens_per_sec_g))
    print(f"  Guarded:  {avg_g*1000:.2f} ms | {tokens_per_sec_g:.2f} tokens/sec")
    print(f"  Overhead: {((avg_g - avg_b) / avg_b) * 100:.2f}%")

# Plotting
ms_b = [r[0] for r in results_baseline]
ms_g = [r[0] for r in results_guarded]

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, ms_b, marker='o', label='Baseline (Vanilla Mamba)', color='blue', linewidth=2)
plt.plot(batch_sizes, ms_g, marker='s', label='SpectralGuard (Multi-Layer Hooks)', color='red', linewidth=2, linestyle='--')

plt.title('Inference Latency vs. Batch Size (100 tokens)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Total Latency (ms)', fontsize=12)
plt.xticks(batch_sizes)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Add overhead percentage annotations
for i, bs in enumerate(batch_sizes):
    pct = ((ms_g[i] - ms_b[i]) / ms_b[i]) * 100
    plt.annotate(f"+{pct:.1f}%", 
                 (bs, ms_g[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 fontsize=9,
                 color='darkred')

plt.tight_layout()
import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Estado_atual', 'latency_benchmark.png')
# Save both to local and artifact explicitly to ensure accessibility
plt.savefig(save_path, dpi=300)
plt.savefig(r'C:\Users\Davib\.gemini\antigravity\brain\510b4e3e-1986-4e32-90f0-db86f84c63e6\latency_benchmark.png', dpi=300)
print(f"\nSaved benchmark plot to latency_benchmark.png")
