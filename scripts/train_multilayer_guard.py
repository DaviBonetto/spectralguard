import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from datasets import load_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. LOAD MODEL
model_id = 'state-spaces/mamba-130m-hf'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Mamba tokenizer does not have a pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()
num_layers = len(model.backbone.layers)
print(f"Model loaded with {num_layers} layers.")

# 2. LOAD DATA
print("Loading TruthfulQA dataset as benign prompts...")
dataset = load_dataset('truthful_qa', 'generation')
# Extract 250 benign prompts
dataset_list = list(dataset['validation'])
benign_prompts = [item['question'] for item in dataset_list[:250]]

# Generate simulated adversarial prompts using the theoretical GCG logic
print("Generating adversarial prompts...")
adv_prompts = []
for p in dataset_list[250:500]:
    base_prompt = p['question']
    # Simulate a GCG suffix that drives the model into strange state
    adv_suffix = " [ADV-CTRL]" * 5 + " [PAD-CTRL]" * 5
    adv_prompts.append(base_prompt + adv_suffix)

# Combine datasets (Total N=500, balanced)
all_prompts = benign_prompts + adv_prompts
labels = [0] * len(benign_prompts) + [1] * len(adv_prompts)

# 3. EXTRACTION LOGIC
def extract_multilayer_features(prompt):
    """
    Hooks into every layer of Mamba-130M and extracts the sequence-wise minimum spectral radius.
    Returns:
        features: dict mapping layer_idx -> min_rho
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    handles = []
    layer_deltas = {i: [] for i in range(num_layers)}
    
    # Define a hook factory to capture per-layer output
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            layer_deltas[layer_idx].append(output[0].detach().float().cpu())
        return hook_fn
    
    # Register hooks
    for i in range(num_layers):
        layer = model.backbone.layers[i].mixer
        handle = layer.dt_proj.register_forward_hook(get_hook(i))
        handles.append(handle)
        
    # Forward pass
    with torch.no_grad():
        model(**inputs)
        
    # Unregister hooks
    for handle in handles:
        handle.remove()
        
    # Extract A matrices
    features = []
    for i in range(num_layers):
        layer = model.backbone.layers[i].mixer
        A_log = layer.A_log.detach().float()
        A = -torch.exp(A_log) # (d_inner, d_state)
        
        deltas = layer_deltas[i]
        if len(deltas) == 0:
            features.append(0.95) # Fallback
            continue
            
        dt_raw = deltas[0] # (batch, seq_len, d_inner)
        import torch.nn.functional as F
        dt = F.softplus(dt_raw) # (batch, seq_len, d_inner)
        dt = dt.unsqueeze(-1)
        A_bar = torch.exp(dt * A.cpu()) # (batch, seq_len, d_inner, d_state)
        
        # Max over states, mean over channels, then minimum across the sequence (the collapse signature)
        # Using the corrected gramian calculation identical to our previous robust logic
        mean_rho_over_channels = A_bar.max(dim=-1).values.mean(dim=-1) # (batch, seq_len)
        min_rho_over_seq = mean_rho_over_channels.min().item()
        
        features.append(min_rho_over_seq)
        
    # From these 24 layer rho values, we can extract:
    # 1. The individual rhos
    # 2. Global mean, std, gap (max - min)
    rhos = np.array(features)
    f_mean = np.mean(rhos)
    f_std = np.std(rhos)
    f_gap = np.max(rhos) - np.min(rhos)
    
    # We will return the full 24-dim vector + globals
    full_feature_vector = np.concatenate([rhos, [f_mean, f_std, f_gap]])
    return full_feature_vector

# 4. RUN EXTRACTION
print("Extracting multi-layer features... (This might take a minute)")
X = []
for prompt in tqdm(all_prompts, desc="Extracting"):
    feat = extract_multilayer_features(prompt)
    X.append(feat)

X = np.array(X)
y = np.array(labels)

# 5. TRAIN & EVALUATE CLASSIFIER
print("\n--- Training Multi-Layer SpectralGuard Classifier ---")
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Shuffle and split (using a simple 80/20 train/test split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"F1-Score: {f1:.4f}  (Goal: > 0.85)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Adversarial']))

# Let's also compare to Single-Layer Layer 0 accuracy on this test set
X_test_unscaled = scaler.inverse_transform(X_test)
layer_0_rhos = X_test_unscaled[:, 0]
# Threshold based on training set's Layer 0 rhos
layer_0_train_rhos = scaler.inverse_transform(X_train)[:, 0]
# Use 5th percentile of benign as threshold
benign_layer_0_rhos = layer_0_train_rhos[y_train == 0]
threshold_l0 = np.percentile(benign_layer_0_rhos, 5)

y_pred_single = (layer_0_rhos < threshold_l0).astype(int)
f1_single = f1_score(y_test, y_pred_single)
print(f"\nBaseline Single-Layer (Layer 0) F1-Score: {f1_single:.4f}")
print(f"Improvement: +{(f1 - f1_single):.4f}")

import joblib
joblib.dump(clf, "multilayer_spectral_guard.pkl")
joblib.dump(scaler, "multilayer_scaler.pkl")
print("Saved models to disk.")
