"""
Spectral Plots for Mamba SSM Analysis.

Visualization functions for eigenvalue spectra, trajectories,
and gramian analysis.

References:
    matplotlib documentation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid", palette="deep")


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "Eigenvalue Spectrum",
    unit_circle: bool = True,
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot eigenvalues in the complex plane.
    
    Eigenvalues near the unit circle indicate long-term memory.
    Eigenvalues near the origin indicate fast forgetting.
    
    Args:
        eigenvalues: Array of complex eigenvalues.
        title: Plot title.
        unit_circle: Whether to draw the unit circle.
        colormap: Colormap for eigenvalue magnitude.
        figsize: Figure size.
        ax: Optional existing axes to plot on.
        save_path: Optional path to save figure.
        
    Returns:
        matplotlib Figure.
        
    Example:
        >>> eigenvalues = analyzer.compute_eigenvalues(A_bar)
        >>> fig = plot_eigenvalue_spectrum(eigenvalues)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract real and imaginary parts
    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)
    magnitudes = np.abs(eigenvalues)
    
    # Draw unit circle
    if unit_circle:
        circle = patches.Circle(
            (0, 0), 1,
            fill=False,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Unit Circle'
        )
        ax.add_patch(circle)
    
    # Scatter eigenvalues
    scatter = ax.scatter(
        real, imag,
        c=magnitudes,
        cmap=colormap,
        s=50,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5,
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|λ|', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Real(λ)', fontsize=12)
    ax.set_ylabel('Imag(λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Grid and legend
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    if unit_circle:
        ax.legend(loc='upper right')
    
    # Tight layout
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved eigenvalue spectrum to {save_path}")
    
    return fig


def plot_spectral_radius_trajectory(
    trajectory: List[float],
    timesteps: Optional[List[int]] = None,
    title: str = "Spectral Radius Evolution",
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 5),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot spectral radius over tokens.
    
    Shows how memory capacity evolves during prompt processing.
    
    Args:
        trajectory: List of spectral radius values.
        timesteps: Token indices (defaults to 0, 1, 2, ...).
        title: Plot title.
        threshold: Optional threshold line (e.g., for stability).
        figsize: Figure size.
        ax: Optional existing axes.
        save_path: Optional path to save figure.
        
    Returns:
        matplotlib Figure.
        
    Example:
        >>> trajectory = analyzer.track_evolution(prompt)
        >>> fig = plot_spectral_radius_trajectory(trajectory.spectral_radius)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if timesteps is None:
        timesteps = list(range(len(trajectory)))
    
    # Plot trajectory
    ax.plot(
        timesteps, trajectory,
        color='#1f77b4',
        linewidth=2,
        marker='o',
        markersize=4,
        label='ρ(Ā)',
    )
    
    # Fill area under curve
    ax.fill_between(
        timesteps, trajectory,
        alpha=0.2,
        color='#1f77b4',
    )
    
    # Threshold line
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold})',
        )
    
    # Stability line at 1.0
    ax.axhline(
        y=1.0,
        color='orange',
        linestyle=':',
        linewidth=1.5,
        alpha=0.7,
        label='Stability boundary',
    )
    
    # Labels
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Spectral Radius ρ(Ā)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trajectory to {save_path}")
    
    return fig


def plot_eigenvalue_clusters(
    eigenvalues: np.ndarray,
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    title: str = "Eigenvalue Clusters",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot eigenvalue clustering in the complex plane.
    
    Visualizes "spectral engrams" - groups of eigenvalues that
    may encode different types of information.
    
    Args:
        eigenvalues: Array of complex eigenvalues.
        labels: Cluster labels from clustering.
        centers: Optional cluster centers.
        title: Plot title.
        figsize: Figure size.
        ax: Optional existing axes.
        save_path: Optional path to save.
        
    Returns:
        matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)
    
    # Plot points colored by cluster
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(
            real[mask], imag[mask],
            color=colors[i],
            s=60,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f'Cluster {i} (n={mask.sum()})',
        )
    
    # Plot centers
    if centers is not None:
        ax.scatter(
            np.real(centers), np.imag(centers),
            c='red',
            s=200,
            marker='X',
            edgecolors='black',
            linewidth=2,
            label='Cluster Centers',
        )
    
    # Unit circle
    circle = patches.Circle(
        (0, 0), 1,
        fill=False,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.5,
    )
    ax.add_patch(circle)
    
    ax.set_xlabel('Real(λ)', fontsize=12)
    ax.set_ylabel('Imag(λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gramian_heatmap(
    gramian: np.ndarray,
    title: str = "Reachability Gramian",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot the reachability gramian as a heatmap.
    
    Args:
        gramian: Gramian matrix W_R.
        title: Plot title.
        log_scale: Use log scale for colormap.
        figsize: Figure size.
        ax: Optional existing axes.
        save_path: Optional path to save.
        
    Returns:
        matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Apply log scale if requested
    if log_scale:
        data = np.log10(np.abs(gramian) + 1e-10)
        cbar_label = 'log₁₀(|W_R|)'
    else:
        data = gramian
        cbar_label = 'W_R'
    
    # Heatmap
    im = ax.imshow(data, cmap='viridis', aspect='equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    ax.set_xlabel('State index i', fontsize=12)
    ax.set_ylabel('State index j', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_singular_value_trajectory(
    singular_values: List[np.ndarray],
    horizon: Optional[int] = None,
    title: str = "Singular Value Evolution",
    threshold: float = 1e-6,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot singular values of gramian over horizon.
    
    Args:
        singular_values: List of singular value arrays at each step.
        horizon: Maximum horizon to plot.
        title: Plot title.
        threshold: Threshold line.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizon is None:
        horizon = len(singular_values)
    
    # Extract min, max, and mean singular values
    steps = list(range(min(len(singular_values), horizon)))
    mins = [sv.min() if len(sv) > 0 else 0 for sv in singular_values[:horizon]]
    maxs = [sv.max() if len(sv) > 0 else 0 for sv in singular_values[:horizon]]
    means = [sv.mean() if len(sv) > 0 else 0 for sv in singular_values[:horizon]]
    
    # Plot
    ax.fill_between(steps, mins, maxs, alpha=0.2, color='blue', label='σ range')
    ax.plot(steps, means, 'b-', linewidth=2, label='σ mean')
    ax.plot(steps, mins, 'b--', linewidth=1, alpha=0.7, label='σ min')
    
    # Threshold
    ax.axhline(y=threshold, color='red', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Horizon Step', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_layer_comparison(
    layer_summaries: List[Dict[str, Any]],
    metric: str = "spectral_radius",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Compare spectral properties across layers.
    
    Args:
        layer_summaries: List of summary dicts from SpectralAnalyzer.
        metric: Which metric to compare.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = [s.get("layer_idx", i) for i, s in enumerate(layer_summaries)]
    values = [s.get(metric, 0) for s in layer_summaries]
    
    ax.bar(layers, values, color='steelblue', edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'{metric.replace("_", " ").title()} by Layer', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_spectral_dashboard(
    eigenvalues: np.ndarray,
    trajectory: List[float],
    gramian: Optional[np.ndarray] = None,
    title: str = "Spectral Analysis Dashboard",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        eigenvalues: Complex eigenvalues.
        trajectory: Spectral radius trajectory.
        gramian: Optional gramian matrix.
        title: Dashboard title.
        figsize: Figure size.
        save_path: Optional save path.
        
    Returns:
        matplotlib Figure with multiple subplots.
    """
    n_plots = 3 if gramian is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # 1. Eigenvalue spectrum
    plot_eigenvalue_spectrum(eigenvalues, ax=axes[0], title="Eigenvalue Spectrum")
    
    # 2. Trajectory
    plot_spectral_radius_trajectory(trajectory, ax=axes[1], title="Spectral Trajectory")
    
    # 3. Gramian (if provided)
    if gramian is not None:
        plot_gramian_heatmap(gramian, ax=axes[2], title="Reachability Gramian")
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dashboard to {save_path}")
    
    return fig
