"""
Reachability Gramian Computation for Mamba SSM.

The Reachability Gramian W_R determines which states are reachable from
input and quantifies the reasoning horizon of the model.

References:
    [Antsaklis2007] Antsaklis, P. J., & Michel, A. N. (2007). 
                    A Linear Systems Primer. Birkhäuser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GramianResult:
    """
    Result of Gramian computation.
    
    Attributes:
        gramian: The reachability gramian W_R.
        singular_values: Singular values of W_R at each step.
        rank: Numerical rank of W_R (σ > threshold).
        min_singular_value: Smallest singular value.
        is_full_rank: Whether gramian has full rank.
        horizon_reached: Step where reachability became limited.
    """
    gramian: np.ndarray
    singular_values: List[np.ndarray]
    rank: int
    min_singular_value: float
    is_full_rank: bool
    horizon_reached: Optional[int] = None


class ReachabilityGramian:
    """
    Computes the Reachability Gramian for SSM analysis.
    
    The Reachability Gramian is defined as:
        W_R(T) = Σ_{k=0}^{T-1} Ā^k · B̄ · B̄^T · (Ā^T)^k
    
    Properties:
    - If W_R is full rank, all states are reachable from input
    - If W_R is singular, some states are unreachable
    - The horizon H is where σ_min(W_R) drops below threshold
    
    This determines the Spectral Horizon - the maximum number of
    reasoning steps the model can effectively perform.
    
    Example:
        >>> gramian_calc = ReachabilityGramian(device='cuda')
        >>> result = gramian_calc.compute(A_bar, B_bar, horizon=100)
        >>> print(f"Rank: {result.rank}, Full rank: {result.is_full_rank}")
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Initialize the Gramian calculator.
        
        Args:
            device: Computation device.
            dtype: Data type (float64 recommended for numerical accuracy).
        """
        self.device = device
        self.dtype = dtype
        
        logger.info(f"ReachabilityGramian initialized on {device} with {dtype}")
    
    def compute(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        horizon: int = 100,
        threshold: float = 1e-10,
        track_singular_values: bool = True,
        show_progress: bool = False,
    ) -> GramianResult:
        """
        Compute the Reachability Gramian iteratively.
        
        W_R(T) = Σ_{k=0}^{T-1} Ā^k · B̄ · B̄^T · (Ā^T)^k
        
        Uses iterative accumulation to avoid O(T × n^3) matrix multiplications.
        
        Args:
            A_bar: Discretized state transition. Shape: [d_state, d_state] or [d_inner, d_state] (diagonal).
            B_bar: Discretized input matrix. Shape: [d_state, d_input] or [d_state].
            horizon: Maximum number of steps (T).
            threshold: Singular value threshold for rank determination.
            track_singular_values: If True, record σ at each step.
            show_progress: Show progress bar.
            
        Returns:
            GramianResult with gramian, singular values, and rank info.
            
        Note:
            For diagonal A (as in Mamba), computation is much more efficient
            as we only need elementwise operations.
        """
        # Ensure tensors are on device with proper dtype
        A_bar = A_bar.to(device=self.device, dtype=self.dtype)
        B_bar = B_bar.to(device=self.device, dtype=self.dtype)
        
        # Handle diagonal A case (Mamba)
        is_diagonal = (A_bar.ndim == 1) or (A_bar.ndim == 2 and A_bar.shape[0] != A_bar.shape[1])
        
        if is_diagonal:
            return self._compute_diagonal(A_bar, B_bar, horizon, threshold, track_singular_values, show_progress)
        else:
            return self._compute_full(A_bar, B_bar, horizon, threshold, track_singular_values, show_progress)
    
    def _compute_diagonal(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        horizon: int,
        threshold: float,
        track_singular_values: bool,
        show_progress: bool,
    ) -> GramianResult:
        """
        Efficient computation for diagonal A matrices.
        
        For diagonal A with elements a_i:
        W_R = Σ_k diag(a)^k B B^T diag(a)^k
            = Σ_k diag(a^{2k}) ⊙ (B B^T)  (for B as column vector)
            
        This allows elementwise operations instead of matrix multiplication.
        """
        # Flatten to diagonal
        if A_bar.ndim == 2:
            a = A_bar.flatten()  # [d_inner * d_state]
        else:
            a = A_bar
        
        n = len(a)
        
        # Ensure B is column vector
        if B_bar.ndim == 1:
            B = B_bar.view(-1, 1)
        else:
            B = B_bar
        
        # Adjust dimensions if needed
        if B.shape[0] != n:
            # Take first n elements or pad
            if B.shape[0] > n:
                B = B[:n]
            else:
                B = torch.nn.functional.pad(B, (0, 0, 0, n - B.shape[0]))
        
        # B B^T outer product
        BBT = B @ B.T  # [n, n]
        
        # Initialize accumulator
        W_R = torch.zeros(n, n, device=self.device, dtype=self.dtype)
        
        # Powers of a (elementwise)
        a_power = torch.ones_like(a)  # a^0 = 1
        
        # Track singular values
        singular_values = []
        horizon_reached = None
        
        iterator = range(horizon)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing Gramian")
        
        for k in iterator:
            # W_R += diag(a_power) @ BBT @ diag(a_power)
            # Which is: (a_power.unsqueeze(1) * a_power.unsqueeze(0)) * BBT
            outer = a_power.unsqueeze(1) * a_power.unsqueeze(0)
            W_R += outer * BBT
            
            # Update power
            a_power = a_power * a
            
            # Track singular values
            if track_singular_values:
                try:
                    s = torch.linalg.svdvals(W_R)
                    singular_values.append(s.cpu().numpy())
                    
                    # Check if horizon reached
                    if horizon_reached is None and s.min() < threshold:
                        horizon_reached = k
                except Exception as e:
                    logger.warning(f"SVD failed at step {k}: {e}")
        
        # Final analysis
        W_R_np = W_R.cpu().numpy()
        
        try:
            final_sv = np.linalg.svd(W_R_np, compute_uv=False)
            rank = int(np.sum(final_sv > threshold))
            min_sv = float(final_sv[-1]) if len(final_sv) > 0 else 0.0
        except Exception:
            final_sv = np.array([])
            rank = 0
            min_sv = 0.0
        
        return GramianResult(
            gramian=W_R_np,
            singular_values=singular_values,
            rank=rank,
            min_singular_value=min_sv,
            is_full_rank=(rank == n),
            horizon_reached=horizon_reached,
        )
    
    def _compute_full(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        horizon: int,
        threshold: float,
        track_singular_values: bool,
        show_progress: bool,
    ) -> GramianResult:
        """
        Standard computation for full (non-diagonal) A matrices.
        
        W_R = Σ_{k=0}^{T-1} A^k B B^T (A^T)^k
        """
        n = A_bar.shape[0]
        
        # Ensure B is proper shape
        if B_bar.ndim == 1:
            B = B_bar.view(-1, 1)
        else:
            B = B_bar
        
        BBT = B @ B.T
        
        # Initialize
        W_R = torch.zeros(n, n, device=self.device, dtype=self.dtype)
        A_power = torch.eye(n, device=self.device, dtype=self.dtype)
        
        singular_values = []
        horizon_reached = None
        
        iterator = range(horizon)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing Gramian")
        
        for k in iterator:
            # W_R += A^k B B^T (A^T)^k
            term = A_power @ BBT @ A_power.T
            W_R += term
            
            # Update A power
            A_power = A_bar @ A_power
            
            # Track singular values
            if track_singular_values:
                try:
                    s = torch.linalg.svdvals(W_R)
                    singular_values.append(s.cpu().numpy())
                    
                    if horizon_reached is None and s.min() < threshold:
                        horizon_reached = k
                except Exception as e:
                    logger.warning(f"SVD failed at step {k}: {e}")
        
        # Final analysis
        W_R_np = W_R.cpu().numpy()
        
        try:
            final_sv = np.linalg.svd(W_R_np, compute_uv=False)
            rank = int(np.sum(final_sv > threshold))
            min_sv = float(final_sv[-1]) if len(final_sv) > 0 else 0.0
        except Exception:
            rank = 0
            min_sv = 0.0
            final_sv = np.array([])
        
        return GramianResult(
            gramian=W_R_np,
            singular_values=singular_values,
            rank=rank,
            min_singular_value=min_sv,
            is_full_rank=(rank == n),
            horizon_reached=horizon_reached,
        )
    
    def compute_controllability_matrix(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute the controllability matrix C = [B, AB, A²B, ..., A^{n-1}B].
        
        Alternative to Gramian for checking reachability.
        
        Args:
            A_bar: State transition matrix.
            B_bar: Input matrix.
            n_steps: Number of steps (defaults to state dimension).
            
        Returns:
            Tuple of (controllability matrix, numerical rank).
        """
        A_bar = A_bar.to(device=self.device, dtype=self.dtype)
        B_bar = B_bar.to(device=self.device, dtype=self.dtype)
        
        n = A_bar.shape[0]
        if n_steps is None:
            n_steps = n
        
        # Ensure B is 2D
        if B_bar.ndim == 1:
            B = B_bar.view(-1, 1)
        else:
            B = B_bar
        
        m = B.shape[1]
        
        # Build controllability matrix
        C = torch.zeros(n, n_steps * m, device=self.device, dtype=self.dtype)
        
        A_power = torch.eye(n, device=self.device, dtype=self.dtype)
        for k in range(n_steps):
            C[:, k*m:(k+1)*m] = A_power @ B
            A_power = A_bar @ A_power
        
        C_np = C.cpu().numpy()
        
        # Compute rank
        rank = int(np.linalg.matrix_rank(C_np))
        
        return C_np, rank
    
    def analyze_reachability(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        horizon: int = 100,
        threshold: float = 1e-10,
    ) -> Dict[str, Any]:
        """
        Comprehensive reachability analysis.
        
        Args:
            A_bar: Discretized transition matrix.
            B_bar: Discretized input matrix.
            horizon: Analysis horizon.
            threshold: Numerical threshold.
            
        Returns:
            Dictionary with full analysis results.
        """
        # Compute gramian
        gramian_result = self.compute(
            A_bar, B_bar, horizon, threshold,
            track_singular_values=True, show_progress=False,
        )
        
        # Compute controllability matrix
        C, ctrl_rank = self.compute_controllability_matrix(A_bar, B_bar)
        
        # Get state dimension
        if A_bar.ndim == 1:
            n = len(A_bar)
        elif A_bar.ndim == 2 and A_bar.shape[0] != A_bar.shape[1]:
            n = A_bar.numel()
        else:
            n = A_bar.shape[0]
        
        return {
            "gramian_rank": gramian_result.rank,
            "controllability_rank": ctrl_rank,
            "state_dimension": n,
            "is_fully_reachable": gramian_result.is_full_rank,
            "min_singular_value": gramian_result.min_singular_value,
            "horizon_reached": gramian_result.horizon_reached,
            "rank_deficit": n - gramian_result.rank,
            "gramian": gramian_result.gramian,
            "singular_value_trajectory": gramian_result.singular_values,
        }
    
    def __repr__(self) -> str:
        return f"ReachabilityGramian(device='{self.device}', dtype={self.dtype})"
