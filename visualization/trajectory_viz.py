"""
Trajectory Visualization for Mamba SSM.

Animated visualizations and state dynamics plots.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def animate_eigenvalue_evolution(
    eigenvalue_trajectories: List[np.ndarray],
    interval: int = 100,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    fps: int = 10,
) -> animation.FuncAnimation:
    """
    Create animation of eigenvalue movement over tokens.
    
    Shows how eigenvalues migrate in the complex plane as the
    model processes the input sequence.
    
    Args:
        eigenvalue_trajectories: List of eigenvalue arrays at each timestep.
        interval: Milliseconds between frames.
        figsize: Figure size.
        save_path: Optional path to save animation (e.g., .gif, .mp4).
        fps: Frames per second for saved video.
        
    Returns:
        matplotlib FuncAnimation object.
        
    Example:
        >>> trajectory = analyzer.track_evolution(prompt)
        >>> anim = animate_eigenvalue_evolution(trajectory.eigenvalues)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='Unit circle')
    
    # Initialize scatter
    scat = ax.scatter([], [], c='blue', s=50, alpha=0.7)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         fontsize=12, verticalalignment='top')
    
    # Set limits based on data
    all_real = np.concatenate([np.real(e) for e in eigenvalue_trajectories])
    all_imag = np.concatenate([np.imag(e) for e in eigenvalue_trajectories])
    
    margin = 0.2
    xlim = (all_real.min() - margin, all_real.max() + margin)
    ylim = (all_imag.min() - margin, all_imag.max() + margin)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')
    ax.set_title('Eigenvalue Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, time_text
    
    def update(frame):
        eigvals = eigenvalue_trajectories[frame]
        positions = np.column_stack([np.real(eigvals), np.imag(eigvals)])
        scat.set_offsets(positions)
        
        # Compute spectral radius
        rho = np.max(np.abs(eigvals))
        time_text.set_text(f'Token: {frame}, ρ = {rho:.4f}')
        
        return scat, time_text
    
    anim = animation.FuncAnimation(
        fig, update,
        init_func=init,
        frames=len(eigenvalue_trajectories),
        interval=interval,
        blit=True,
    )
    
    if save_path:
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            else:
                anim.save(save_path, fps=fps)
            logger.info(f"Saved animation to {save_path}")
        except Exception as e:
            logger.warning(f"Could not save animation: {e}")
    
    return anim


def plot_state_dynamics(
    states: List[np.ndarray],
    state_indices: Optional[List[int]] = None,
    n_states: int = 5,
    title: str = "Hidden State Dynamics",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot hidden state evolution over tokens.
    
    Shows how individual state components change during processing.
    
    Args:
        states: List of state vectors at each timestep.
        state_indices: Which state indices to plot (None = auto-select).
        n_states: Number of states to plot if auto-selecting.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to array
    states_array = np.array(states)  # [timesteps, state_dim]
    timesteps = np.arange(len(states))
    
    # Select state indices
    if state_indices is None:
        # Auto-select: pick states with highest variance
        variances = np.var(states_array, axis=0)
        state_indices = np.argsort(variances)[-n_states:]
    
    # Plot each state
    colors = plt.cm.tab10(np.linspace(0, 1, len(state_indices)))
    for i, (idx, color) in enumerate(zip(state_indices, colors)):
        ax.plot(
            timesteps, states_array[:, idx],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f'State {idx}',
        )
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('State Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spectral_heatmap_over_layers(
    layer_trajectories: Dict[int, List[float]],
    title: str = "Spectral Radius Across Layers",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Heatmap of spectral radius across layers and tokens.
    
    Args:
        layer_trajectories: Dict mapping layer_idx to trajectory list.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Build matrix
    layer_indices = sorted(layer_trajectories.keys())
    max_len = max(len(t) for t in layer_trajectories.values())
    
    matrix = np.zeros((len(layer_indices), max_len))
    for i, layer_idx in enumerate(layer_indices):
        traj = layer_trajectories[layer_idx]
        matrix[i, :len(traj)] = traj
    
    # Heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spectral Radius ρ(Ā)', fontsize=12)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(layer_indices)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attack_comparison(
    original_trajectory: List[float],
    attacked_trajectory: List[float],
    attack_name: str = "HiSPA",
    title: str = "Attack Impact on Spectral Radius",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Compare spectral trajectories before and after attack.
    
    Args:
        original_trajectory: Spectral trajectory of benign prompt.
        attacked_trajectory: Spectral trajectory of adversarial prompt.
        attack_name: Name of the attack.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both trajectories
    ax.plot(
        original_trajectory,
        'b-', linewidth=2, marker='o', markersize=4,
        label='Original (benign)',
    )
    ax.plot(
        attacked_trajectory,
        'r-', linewidth=2, marker='x', markersize=6,
        label=f'After {attack_name} attack',
    )
    
    # Fill area between
    min_len = min(len(original_trajectory), len(attacked_trajectory))
    ax.fill_between(
        range(min_len),
        original_trajectory[:min_len],
        attacked_trajectory[:min_len],
        alpha=0.2,
        color='red',
        label='Collapse region',
    )
    
    # Stability line
    ax.axhline(y=1.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Spectral Radius ρ(Ā)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add collapse annotation
    if len(attacked_trajectory) > 0:
        min_idx = np.argmin(attacked_trajectory)
        min_val = attacked_trajectory[min_idx]
        ax.annotate(
            f'Collapse: ρ = {min_val:.3f}',
            xy=(min_idx, min_val),
            xytext=(min_idx + 5, min_val + 0.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10,
            color='red',
        )
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_security_check_results(
    results: List[Dict[str, Any]],
    title: str = "SpectralGuard Results",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Visualize security check results.
    
    Args:
        results: List of SecurityCheckResult as dicts.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure with two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Pie chart of safe vs blocked
    n_safe = sum(1 for r in results if r.get('is_safe', True))
    n_blocked = len(results) - n_safe
    
    ax1.pie(
        [n_safe, n_blocked],
        labels=['Safe', 'Blocked'],
        colors=['green', 'red'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.05),
    )
    ax1.set_title('Detection Results', fontsize=12)
    
    # 2. Histogram of mean radius
    mean_radii = [
        np.mean(r.get('trajectory', [1.0])) if r.get('trajectory') else 1.0
        for r in results
    ]
    
    ax2.hist(mean_radii, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax2.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Mean Spectral Radius', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Mean Radius Distribution', fontsize=12)
    ax2.legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
