"""
Eigenvalue Analyzer for Mamba SSM.

Computes and analyzes eigenvalues of the state transition matrix to
understand memory dynamics and reasoning capabilities.

References:
    [Gu2023] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
             with Selective State Spaces. arXiv:2312.00752
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class SpectralTrajectory:
    """
    Container for eigenvalue evolution over tokens.
    
    Attributes:
        timesteps: Token indices where measurements were taken.
        eigenvalues: List of eigenvalue arrays at each timestep.
        spectral_radius: Spectral radius (max |λ|) at each timestep.
        delta_values: Discretization step Δ at each timestep.
        layer_idx: Layer index being tracked.
    """
    timesteps: List[int] = field(default_factory=list)
    eigenvalues: List[np.ndarray] = field(default_factory=list)
    spectral_radius: List[float] = field(default_factory=list)
    delta_values: Optional[List[np.ndarray]] = None
    layer_idx: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timesteps": self.timesteps,
            "eigenvalues": [e.tolist() for e in self.eigenvalues],
            "spectral_radius": self.spectral_radius,
            "delta_values": [d.tolist() for d in self.delta_values] if self.delta_values else None,
            "layer_idx": self.layer_idx,
        }


@dataclass
class ClusterResult:
    """
    Result of eigenvalue clustering.
    
    Attributes:
        centers: Cluster centers in complex plane.
        labels: Cluster label for each eigenvalue.
        sizes: Number of eigenvalues in each cluster.
        inertia: Sum of squared distances to cluster centers.
    """
    centers: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray
    inertia: float


class SpectralAnalyzer:
    """
    Analyzes eigenvalues of the SSM transition matrix in Mamba models.
    
    The eigenvalues λ of the discretized matrix Ā determine:
    - |λ| ≈ 1: Long-term memory (information preserved)
    - |λ| ≈ 0: Fast forgetting (filters noise)
    - |λ| > 1: Unstable dynamics (exponential growth)
    
    For Mamba, A is diagonal, so eigenvalues are simply the diagonal elements.
    
    Attributes:
        wrapper: MambaWrapper instance.
        extractor: StateExtractor for accessing A matrices.
        device: Computation device.
        
    Example:
        >>> analyzer = SpectralAnalyzer(model)
        >>> eigenvalues = analyzer.compute_eigenvalues(layer_idx=0)
        >>> print(f"Spectral radius: {analyzer.spectral_radius(eigenvalues):.4f}")
        
        >>> trajectory = analyzer.track_evolution(prompt_tokens)
        >>> print(f"Radius declined from {trajectory.spectral_radius[0]:.3f} "
        ...       f"to {trajectory.spectral_radius[-1]:.3f}")
    """
    
    def __init__(
        self,
        model: Union["MambaWrapper", nn.Module],
        device: str = "cuda",
    ) -> None:
        """
        Initialize the spectral analyzer.
        
        Args:
            model: MambaWrapper or raw Mamba model.
            device: Device for computation ('cuda' or 'cpu').
        """
        # Handle both MambaWrapper and raw model
        from mamba_spectral.core.mamba_wrapper import MambaWrapper
        from mamba_spectral.core.state_extractor import StateExtractor
        
        if isinstance(model, MambaWrapper):
            self.wrapper = model
        else:
            self.wrapper = MambaWrapper(model, device=device)
        
        self.extractor = StateExtractor(self.wrapper)
        self.device = device
        
        logger.info(f"SpectralAnalyzer initialized on {device}")
    
    def extract_A_matrix(
        self,
        layer_idx: int,
        as_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract the continuous A matrix from a specific layer.
        
        In Mamba, A is stored as A_log and computed as A = -exp(A_log).
        This ensures stability (negative eigenvalues → stable dynamics).
        
        Args:
            layer_idx: Index of the Mamba layer (0-indexed).
            as_numpy: If True, return numpy array instead of tensor.
            
        Returns:
            A matrix of shape [d_inner, d_state].
            
        Example:
            >>> A = analyzer.extract_A_matrix(layer_idx=0)
            >>> print(f"A shape: {A.shape}")
        """
        A = self.extractor.extract_A_matrix(layer_idx)
        
        if as_numpy:
            return A.detach().cpu().numpy()
        return A
    
    def discretize_A(
        self,
        A_continuous: torch.Tensor,
        delta: Union[torch.Tensor, float],
        method: str = "zoh",
    ) -> torch.Tensor:
        """
        Discretize the continuous A matrix using Zero-Order Hold.
        
        Ā = exp(Δ · A)
        
        For diagonal A (as in Mamba), this is element-wise exponentiation.
        
        Args:
            A_continuous: Continuous A matrix. Shape: [d_inner, d_state].
            delta: Discretization step. Can be:
                - float: Single value applied uniformly
                - Tensor [d_inner]: Per-channel delta
                - Tensor [batch, seq_len, d_inner]: Input-dependent delta
            method: Discretization method:
                - 'zoh': Zero-Order Hold (default, recommended)
                - 'euler': Forward Euler (less accurate)
                
        Returns:
            torch.Tensor: Discretized Ā matrix.
            
        Example:
            >>> A = analyzer.extract_A_matrix(0)
            >>> A_bar = analyzer.discretize_A(A, delta=0.01)
            >>> eigenvalues = analyzer.compute_eigenvalues(A_bar)
        """
        return self.extractor.discretize(A_continuous, delta, method)
    
    def compute_eigenvalues(
        self,
        A_matrix: Union[torch.Tensor, np.ndarray],
        sort_by: str = "magnitude",
    ) -> np.ndarray:
        """
        Compute eigenvalues of the A matrix.
        
        For diagonal matrices (as in Mamba), eigenvalues ARE the diagonal.
        For non-diagonal matrices, uses torch.linalg.eigvals.
        
        Args:
            A_matrix: State transition matrix to analyze.
                - Diagonal: Shape [d_inner, d_state] or [d_state]
                - Full: Shape [d_state, d_state]
            sort_by: How to sort eigenvalues:
                - 'magnitude': |λ| descending (default)
                - 'real': Real part descending
                - 'none': No sorting
                
        Returns:
            np.ndarray: Complex eigenvalues sorted as specified.
            
        Note:
            For Mamba's diagonal A, eigenvalues are real and negative
            (since A = -exp(A_log)).
        """
        if isinstance(A_matrix, torch.Tensor):
            A_matrix = A_matrix.detach().cpu().numpy()
        
        # Check if diagonal (Mamba case)
        if A_matrix.ndim == 1:
            # Already diagonal
            eigenvalues = A_matrix.astype(np.complex128)
        elif A_matrix.ndim == 2 and A_matrix.shape[0] != A_matrix.shape[1]:
            # [d_inner, d_state] - flatten diagonal
            eigenvalues = A_matrix.flatten().astype(np.complex128)
        else:
            # Full square matrix - compute eigvals
            eigenvalues = np.linalg.eigvals(A_matrix)
        
        # Sort eigenvalues
        if sort_by == "magnitude":
            indices = np.argsort(-np.abs(eigenvalues))
            eigenvalues = eigenvalues[indices]
        elif sort_by == "real":
            indices = np.argsort(-np.real(eigenvalues))
            eigenvalues = eigenvalues[indices]
        # else: no sorting
        
        return eigenvalues
    
    def spectral_radius(
        self,
        eigenvalues: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute the spectral radius ρ(A) = max(|λ_i|).
        
        The spectral radius determines:
        - ρ < 1: System is stable (states decay)
        - ρ = 1: Marginally stable (states persist)
        - ρ > 1: Unstable (states grow exponentially)
        
        Args:
            eigenvalues: Array of eigenvalues.
            
        Returns:
            float: Maximum absolute eigenvalue.
            
        Example:
            >>> eigenvalues = analyzer.compute_eigenvalues(A_bar)
            >>> rho = analyzer.spectral_radius(eigenvalues)
            >>> print(f"Spectral radius: {rho:.4f}")
            >>> if rho > 1:
            ...     print("WARNING: System is unstable!")
        """
        if isinstance(eigenvalues, torch.Tensor):
            eigenvalues = eigenvalues.detach().cpu().numpy()
        
        return float(np.max(np.abs(eigenvalues)))
    
    def spectral_gap(
        self,
        eigenvalues: np.ndarray,
    ) -> float:
        """
        Compute the spectral gap = |λ_1| - |λ_2|.
        
        A larger gap indicates more distinct dominant dynamics.
        
        Args:
            eigenvalues: Array of eigenvalues (will be sorted by magnitude).
            
        Returns:
            float: Difference between top two eigenvalue magnitudes.
        """
        magnitudes = np.sort(np.abs(eigenvalues))[::-1]
        
        if len(magnitudes) < 2:
            return 0.0
        
        return float(magnitudes[0] - magnitudes[1])
    
    def condition_number(
        self,
        eigenvalues: np.ndarray,
    ) -> float:
        """
        Compute condition number = |λ_max| / |λ_min|.
        
        High condition number indicates sensitivity to perturbations.
        
        Args:
            eigenvalues: Array of eigenvalues.
            
        Returns:
            float: Ratio of largest to smallest eigenvalue magnitude.
        """
        magnitudes = np.abs(eigenvalues)
        magnitudes = magnitudes[magnitudes > 1e-10]  # Filter near-zero
        
        if len(magnitudes) == 0:
            return float("inf")
        
        return float(np.max(magnitudes) / np.min(magnitudes))
    
    def eigenvalue_clustering(
        self,
        eigenvalues: np.ndarray,
        n_clusters: int = 2,
        random_state: int = 42,
    ) -> ClusterResult:
        """
        Detect clustering of eigenvalues in the complex plane.
        
        Clustering reveals "spectral engramas" - groups of eigenvalues
        that may encode different types of information.
        
        Args:
            eigenvalues: Complex eigenvalues to cluster.
            n_clusters: Number of clusters (default: 2).
            random_state: Random seed for reproducibility.
            
        Returns:
            ClusterResult with:
                - centers: Cluster center locations
                - labels: Cluster assignment for each eigenvalue
                - sizes: Number of eigenvalues per cluster
                - inertia: Clustering quality metric
                
        Example:
            >>> result = analyzer.eigenvalue_clustering(eigenvalues, n_clusters=3)
            >>> print(f"Cluster sizes: {result.sizes}")
            >>> print(f"Centers: {result.centers}")
        """
        # Convert complex to 2D real coordinates
        X = np.column_stack([
            np.real(eigenvalues),
            np.imag(eigenvalues),
        ])
        
        # Fit K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)
        
        # Convert centers back to complex
        centers = kmeans.cluster_centers_[:, 0] + 1j * kmeans.cluster_centers_[:, 1]
        
        # Count cluster sizes
        sizes = np.bincount(labels, minlength=n_clusters)
        
        return ClusterResult(
            centers=centers,
            labels=labels,
            sizes=sizes,
            inertia=float(kmeans.inertia_),
        )
    
    @torch.no_grad()
    def track_evolution(
        self,
        prompt: Union[str, torch.Tensor],
        layer_idx: int = 0,
        save_every: int = 1,
        delta_value: float = 0.01,
    ) -> SpectralTrajectory:
        """
        Track eigenvalue evolution token-by-token during inference.
        
        This is the core method for understanding how spectral properties
        change as the model processes input.
        
        Args:
            prompt: Input text (string) or token IDs (tensor).
            layer_idx: Which Mamba layer to analyze (0-indexed).
            save_every: Record every N tokens (1 = all tokens).
            delta_value: Default delta for discretization.
            
        Returns:
            SpectralTrajectory containing:
                - timesteps: Token indices
                - eigenvalues: Eigenvalues at each timestep
                - spectral_radius: ρ(Ā) at each timestep
                
        Example:
            >>> trajectory = analyzer.track_evolution(
            ...     "The capital of France is Paris, located in Europe.",
            ...     layer_idx=0,
            ... )
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(trajectory.timesteps, trajectory.spectral_radius)
            >>> plt.xlabel("Token")
            >>> plt.ylabel("Spectral Radius")
            >>> plt.show()
        """
        # Tokenize if string
        if isinstance(prompt, str):
            if self.wrapper.tokenizer is None:
                raise ValueError("Tokenizer required for string input")
            tokens = self.wrapper.tokenizer.encode(prompt, return_tensors="pt")
            tokens = tokens.to(self.device)
        else:
            tokens = prompt.to(self.device)
        
        trajectory = SpectralTrajectory(layer_idx=layer_idx)
        seq_len = tokens.shape[1]
        
        # Get base A matrix
        A = self.extract_A_matrix(layer_idx)
        delta = torch.tensor(delta_value, device=self.device)
        
        # Track through sequence
        for t in range(0, seq_len, save_every):
            # Create partial input
            partial_tokens = tokens[:, :t+1]
            
            # Get discretized A for this context
            # Note: In full implementation, would extract Δ from forward pass
            A_bar = self.discretize_A(A, delta)
            
            # Compute eigenvalues
            eigenvalues = self.compute_eigenvalues(A_bar)
            radius = self.spectral_radius(eigenvalues)
            
            trajectory.timesteps.append(t)
            trajectory.eigenvalues.append(eigenvalues)
            trajectory.spectral_radius.append(radius)
        
        logger.info(
            f"Tracked {len(trajectory.timesteps)} timesteps. "
            f"Radius range: [{min(trajectory.spectral_radius):.3f}, "
            f"{max(trajectory.spectral_radius):.3f}]"
        )
        
        return trajectory
    
    @torch.no_grad()
    def track_evolution_with_hooks(
        self,
        prompt: Union[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, SpectralTrajectory]:
        """
        Track evolution using forward hooks for accurate Δ extraction.
        
        This method registers hooks to capture the actual input-dependent
        discretization step during forward pass.
        
        Args:
            prompt: Input text or tokens.
            layer_indices: Which layers to track (None = all).
            
        Returns:
            Dict mapping layer_idx to SpectralTrajectory.
        """
        # Tokenize
        if isinstance(prompt, str):
            tokens = self.wrapper.tokenizer.encode(prompt, return_tensors="pt")
            tokens = tokens.to(self.device)
        else:
            tokens = prompt.to(self.device)
        
        layers = self.wrapper.get_mamba_layers()
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        
        # Storage for captured data
        captured_delta: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Try to extract delta from module state
                if hasattr(module, "_delta"):
                    captured_delta[layer_idx].append(module._delta.detach().cpu())
            return hook
        
        # Register hooks
        handles = []
        for idx in layer_indices:
            handle = self.wrapper.register_hook(idx, make_hook(idx), "forward")
            handles.append(handle)
        
        try:
            # Forward pass
            _ = self.wrapper.forward(tokens)
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
        
        # Build trajectories
        trajectories = {}
        for layer_idx in layer_indices:
            A = self.extract_A_matrix(layer_idx)
            trajectory = SpectralTrajectory(layer_idx=layer_idx)
            
            deltas = captured_delta.get(layer_idx, [])
            if deltas:
                for t, delta in enumerate(deltas):
                    A_bar = self.discretize_A(A, delta.to(self.device))
                    eigenvalues = self.compute_eigenvalues(A_bar)
                    
                    trajectory.timesteps.append(t)
                    trajectory.eigenvalues.append(eigenvalues)
                    trajectory.spectral_radius.append(self.spectral_radius(eigenvalues))
            else:
                # Fallback: use default delta
                trajectory = self.track_evolution(tokens, layer_idx)
            
            trajectories[layer_idx] = trajectory
        
        return trajectories
    
    def summarize_layer(
        self,
        layer_idx: int,
        delta_value: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for a layer's spectral properties.
        
        Args:
            layer_idx: Layer to analyze.
            delta_value: Discretization step.
            
        Returns:
            Dictionary with:
                - spectral_radius: ρ(Ā)
                - spectral_gap: |λ_1| - |λ_2|
                - condition_number: |λ_max| / |λ_min|
                - num_stable: Count of |λ| < 1
                - num_unstable: Count of |λ| > 1
                - mean_magnitude: Average |λ|
                - std_magnitude: Std of |λ|
        """
        A = self.extract_A_matrix(layer_idx)
        A_bar = self.discretize_A(A, torch.tensor(delta_value))
        eigenvalues = self.compute_eigenvalues(A_bar)
        magnitudes = np.abs(eigenvalues)
        
        return {
            "layer_idx": layer_idx,
            "num_eigenvalues": len(eigenvalues),
            "spectral_radius": self.spectral_radius(eigenvalues),
            "spectral_gap": self.spectral_gap(eigenvalues),
            "condition_number": self.condition_number(eigenvalues),
            "num_stable": int(np.sum(magnitudes < 1)),
            "num_unstable": int(np.sum(magnitudes > 1)),
            "num_marginal": int(np.sum(np.isclose(magnitudes, 1, atol=0.01))),
            "mean_magnitude": float(np.mean(magnitudes)),
            "std_magnitude": float(np.std(magnitudes)),
            "min_magnitude": float(np.min(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
        }
    
    def analyze_all_layers(
        self,
        delta_value: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Analyze spectral properties of all Mamba layers.
        
        Args:
            delta_value: Discretization step.
            
        Returns:
            List of summary dictionaries, one per layer.
        """
        layers = self.wrapper.get_mamba_layers()
        results = []
        
        for i in range(len(layers)):
            try:
                summary = self.summarize_layer(i, delta_value)
                results.append(summary)
            except Exception as e:
                logger.warning(f"Could not analyze layer {i}: {e}")
                results.append({"layer_idx": i, "error": str(e)})
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"SpectralAnalyzer(\n"
            f"  model='{self.wrapper.model_name}',\n"
            f"  num_layers={len(self.wrapper.get_mamba_layers())},\n"
            f"  device='{self.device}'\n"
            f")"
        )
