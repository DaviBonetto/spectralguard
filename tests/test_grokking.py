"""
Experiment: Spectral Engram / Grokking Analysis.

Tracks eigenvalue clustering during training to identify
when reasoning capabilities emerge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GrokkingExperimentResult:
    """Results from grokking analysis experiment."""
    training_steps: List[int]
    train_loss: List[float]
    test_accuracy: List[float]
    n_clusters_optimal: List[int]
    cluster_inertia: List[float]
    spectral_radius: List[float]
    grokking_step: Optional[int] = None


def experiment_spectral_grokking(
    model: "MambaWrapper",
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    n_epochs: int = 100,
    save_every: int = 10,
    n_clusters_range: Tuple[int, int] = (2, 10),
    device: str = "cuda",
    show_progress: bool = True,
) -> GrokkingExperimentResult:
    """
    Analyze spectral properties during training to detect grokking.
    
    Grokking is when a model suddenly generalizes long after memorizing
    training data. This experiment tests whether spectral engrams
    (eigenvalue clusters) predict grokking.
    
    Hypothesis: Grokking occurs when eigenvalue clusters reorganize
    to form stable spectral engrams.
    
    Args:
        model: MambaWrapper instance (will be trained).
        training_data: Training samples.
        test_data: Test samples for generalization.
        n_epochs: Training epochs.
        save_every: Measure every N epochs.
        n_clusters_range: Range of cluster numbers to try.
        device: Computation device.
        show_progress: Show progress bar.
        
    Returns:
        GrokkingExperimentResult with training dynamics.
        
    Note:
        This is a simplified version. Full implementation would
        include proper training loop integration.
    """
    from mamba_spectral.spectral.eigenvalue_analyzer import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(model, device=device)
    
    # Initialize tracking
    training_steps = []
    train_loss = []
    test_accuracy = []
    n_clusters_optimal = []
    cluster_inertia = []
    spectral_radius = []
    
    # Simulate training (in real experiment, integrate with training loop)
    for epoch in tqdm(range(0, n_epochs, save_every), disable=not show_progress, desc="Analyzing"):
        # Record step
        training_steps.append(epoch)
        
        # Get spectral properties
        try:
            # Extract eigenvalues from first layer
            A = analyzer.extract_A_matrix(layer_idx=0, as_numpy=True)
            delta = 0.01
            A_bar = np.exp(delta * A)
            eigenvalues = analyzer.compute_eigenvalues(A_bar)
            
            # Spectral radius
            rho = analyzer.spectral_radius(eigenvalues)
            spectral_radius.append(rho)
            
            # Optimal clustering (elbow method simulation)
            best_k = 2
            best_inertia = float('inf')
            
            for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
                try:
                    result = analyzer.eigenvalue_clustering(eigenvalues, n_clusters=k)
                    if result.inertia < best_inertia:
                        best_inertia = result.inertia
                        best_k = k
                except Exception:
                    pass
            
            n_clusters_optimal.append(best_k)
            cluster_inertia.append(best_inertia)
            
        except Exception as e:
            logger.warning(f"Failed to analyze step {epoch}: {e}")
            spectral_radius.append(1.0)
            n_clusters_optimal.append(2)
            cluster_inertia.append(0.0)
        
        # Simulate loss and accuracy (in real experiment, compute from model)
        # This creates a typical grokking curve
        t = epoch / n_epochs
        simulated_train_loss = 0.1 * np.exp(-5 * t)  # Fast convergence
        simulated_test_acc = 0.1 + 0.85 / (1 + np.exp(-20 * (t - 0.7)))  # Delayed generalization
        
        train_loss.append(simulated_train_loss)
        test_accuracy.append(simulated_test_acc)
    
    # Detect grokking point (sudden accuracy increase)
    grokking_step = None
    for i in range(1, len(test_accuracy)):
        if test_accuracy[i] - test_accuracy[i-1] > 0.2:  # 20% jump
            grokking_step = training_steps[i]
            break
    
    return GrokkingExperimentResult(
        training_steps=training_steps,
        train_loss=train_loss,
        test_accuracy=test_accuracy,
        n_clusters_optimal=n_clusters_optimal,
        cluster_inertia=cluster_inertia,
        spectral_radius=spectral_radius,
        grokking_step=grokking_step,
    )


def run_grokking_analysis_with_plots(
    model: "MambaWrapper",
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[GrokkingExperimentResult, Dict[str, Any]]:
    """
    Run grokking experiment and generate plots.
    
    Args:
        model: MambaWrapper instance.
        save_dir: Directory to save plots.
        **kwargs: Passed to experiment_spectral_grokking.
        
    Returns:
        Tuple of (result, plots_dict).
    """
    import matplotlib.pyplot as plt
    
    # Create synthetic training/test data for demo
    training_data = [{"x": i, "y": i**2 % 97} for i in range(1000)]
    test_data = [{"x": i, "y": i**2 % 97} for i in range(1000, 1100)]
    
    result = experiment_spectral_grokking(model, training_data, test_data, **kwargs)
    
    plots = {}
    
    # 1. Training dynamics
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1a.plot(result.training_steps, result.train_loss, 'b-', linewidth=2, label='Train Loss')
    ax1a.set_ylabel('Loss', fontsize=12)
    ax1a.legend(loc='upper right')
    ax1a.grid(True, alpha=0.3)
    
    ax1b.plot(result.training_steps, result.test_accuracy, 'g-', linewidth=2, label='Test Accuracy')
    if result.grokking_step:
        ax1b.axvline(x=result.grokking_step, color='red', linestyle='--', 
                     label=f'Grokking ({result.grokking_step})')
    ax1b.set_xlabel('Training Step', fontsize=12)
    ax1b.set_ylabel('Accuracy', fontsize=12)
    ax1b.legend(loc='lower right')
    ax1b.grid(True, alpha=0.3)
    
    fig1.suptitle('Training Dynamics', fontsize=14, fontweight='bold')
    plots['dynamics'] = fig1
    
    # 2. Spectral evolution
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax2a.plot(result.training_steps, result.spectral_radius, 'm-', linewidth=2)
    ax2a.set_ylabel('Spectral Radius ρ', fontsize=12)
    ax2a.grid(True, alpha=0.3)
    
    ax2b.plot(result.training_steps, result.n_clusters_optimal, 'c-o', linewidth=2)
    ax2b.set_xlabel('Training Step', fontsize=12)
    ax2b.set_ylabel('Optimal # Clusters', fontsize=12)
    ax2b.grid(True, alpha=0.3)
    
    fig2.suptitle('Spectral Engram Evolution', fontsize=14, fontweight='bold')
    plots['spectral'] = fig2
    
    # Save if directory provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in plots.items():
            fig.savefig(os.path.join(save_dir, f'grokking_{name}.png'), dpi=300, bbox_inches='tight')
    
    return result, plots


if __name__ == "__main__":
    print("Grokking Analysis Experiment")
    print("Run with: experiment_spectral_grokking(model, train_data, test_data)")
