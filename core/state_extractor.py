"""
State Extractor for Mamba SSM.

Extracts the internal state matrices (A, B, C, D) and discretization
parameters from Mamba layers for spectral analysis.

References:
    [Gu2023] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
             with Selective State Spaces. arXiv:2312.00752
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSMState:
    """
    Container for extracted SSM state matrices.
    
    Attributes:
        A: State transition matrix (continuous). Shape: [d_inner, d_state].
        A_bar: Discretized state transition matrix. Shape: [d_inner, d_state].
        B: Input matrix. Shape: [batch, seq_len, d_state].
        C: Output matrix. Shape: [batch, seq_len, d_state].
        D: Skip connection. Shape: [d_inner].
        delta: Discretization step. Shape: [batch, seq_len, d_inner].
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_inner: Inner dimension (d_model * expand).
    """
    A: torch.Tensor
    A_bar: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    D: Optional[torch.Tensor] = None
    delta: Optional[torch.Tensor] = None
    d_model: Optional[int] = None
    d_state: Optional[int] = None
    d_inner: Optional[int] = None
    
    def to(self, device: str) -> "SSMState":
        """Move all tensors to specified device."""
        return SSMState(
            A=self.A.to(device) if self.A is not None else None,
            A_bar=self.A_bar.to(device) if self.A_bar is not None else None,
            B=self.B.to(device) if self.B is not None else None,
            C=self.C.to(device) if self.C is not None else None,
            D=self.D.to(device) if self.D is not None else None,
            delta=self.delta.to(device) if self.delta is not None else None,
            d_model=self.d_model,
            d_state=self.d_state,
            d_inner=self.d_inner,
        )
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert all tensors to numpy arrays."""
        result = {}
        for field in ["A", "A_bar", "B", "C", "D", "delta"]:
            tensor = getattr(self, field)
            if tensor is not None:
                result[field] = tensor.detach().cpu().numpy()
        return result


class StateExtractor:
    """
    Extracts SSM state matrices from Mamba layers.
    
    The Mamba architecture uses a selective state space model with:
    - A: Diagonal state transition matrix (stored as A_log for stability)
    - B, C: Input/output matrices (input-dependent in Mamba)
    - Δ (delta): Discretization step (also input-dependent)
    
    Attributes:
        wrapper: MambaWrapper instance.
        device: Device for computation.
        
    Example:
        >>> extractor = StateExtractor(wrapper)
        >>> states = extractor.extract_from_layer(0)
        >>> print(f"A matrix shape: {states.A.shape}")
        >>> print(f"Spectral radius: {states.A.abs().max().item():.4f}")
    """
    
    def __init__(
        self,
        wrapper: "MambaWrapper",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the state extractor.
        
        Args:
            wrapper: MambaWrapper instance containing the model.
            device: Device for computation (defaults to wrapper's device).
        """
        self.wrapper = wrapper
        self.device = device or wrapper.device
        self._cached_states: Dict[int, SSMState] = {}
        
        logger.info(f"StateExtractor initialized on {self.device}")
    
    def extract_A_matrix(
        self,
        layer_idx: int,
        return_log: bool = False,
    ) -> torch.Tensor:
        """
        Extract the A matrix from a specific layer.
        
        In Mamba, A is stored as A_log (log space) for numerical stability.
        The actual A is computed as: A = -exp(A_log)
        
        The negative sign ensures stability (eigenvalues with negative real parts).
        
        Args:
            layer_idx: Index of the Mamba layer (0-indexed).
            return_log: If True, return A_log instead of A.
            
        Returns:
            torch.Tensor: A matrix of shape [d_inner, d_state].
            
        Note:
            A is diagonal, so this returns the diagonal elements only.
            For a full matrix, use torch.diag(A) on each d_inner slice.
        """
        layers = self.wrapper.get_mamba_layers()
        
        if layer_idx >= len(layers):
            raise IndexError(
                f"Layer {layer_idx} not found. Model has {len(layers)} layers."
            )
        
        layer = layers[layer_idx]
        
        # Try different attribute names used in mamba implementations
        A_log = None
        
        if hasattr(layer, "A_log"):
            A_log = layer.A_log
        elif hasattr(layer, "A"):
            # Some implementations store A directly
            return layer.A.detach().to(self.device)
        elif hasattr(layer, "ssm"):
            # Nested SSM structure
            if hasattr(layer.ssm, "A_log"):
                A_log = layer.ssm.A_log
        
        if A_log is None:
            raise AttributeError(
                f"Could not find A_log or A in layer {layer_idx}. "
                f"Available attributes: {dir(layer)}"
            )
        
        A_log = A_log.detach().to(self.device)
        
        if return_log:
            return A_log
        
        # A = -exp(A_log) to ensure stability
        # Negative because we want eigenvalues with negative real part
        A = -torch.exp(A_log)
        
        return A
    
    def extract_all_parameters(
        self,
        layer_idx: int,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> SSMState:
        """
        Extract all SSM parameters from a layer.
        
        For input-dependent parameters (B, C, delta), an input tensor
        must be provided to compute these projections.
        
        Args:
            layer_idx: Index of the Mamba layer.
            input_tensor: Optional input for computing B, C, delta.
                Shape: [batch, seq_len, d_model]
                
        Returns:
            SSMState: Container with all extracted matrices.
        """
        layers = self.wrapper.get_mamba_layers()
        layer = layers[layer_idx]
        
        # Extract A (always available)
        A = self.extract_A_matrix(layer_idx)
        
        # Initialize state
        state = SSMState(
            A=A,
            d_inner=A.shape[0] if len(A.shape) > 1 else A.shape[0],
            d_state=A.shape[-1],
        )
        
        # Extract D (skip connection) if available
        if hasattr(layer, "D"):
            state.D = layer.D.detach().to(self.device)
        
        # Extract other parameters if input is provided
        if input_tensor is not None:
            state = self._extract_input_dependent(layer, state, input_tensor)
        
        return state
    
    def _extract_input_dependent(
        self,
        layer: nn.Module,
        state: SSMState,
        x: torch.Tensor,
    ) -> SSMState:
        """
        Extract input-dependent parameters (B, C, delta).
        
        In Mamba, these are computed from the input via projections:
        - x_proj = layer.in_proj(x) or similar
        - B = x_proj[..., :d_state]
        - delta = softplus(x_proj[..., d_state:d_state+d_inner])
        
        Args:
            layer: Mamba layer module.
            state: SSMState to update.
            x: Input tensor of shape [batch, seq_len, d_model].
            
        Returns:
            Updated SSMState with B, C, delta.
        """
        x = x.to(self.device)
        
        try:
            # Method 1: Direct projection attributes
            if hasattr(layer, "x_proj") and hasattr(layer, "dt_proj"):
                # Compute input projection
                batch, seq_len, d_model = x.shape
                d_inner = state.d_inner or layer.d_inner
                d_state = state.d_state or layer.d_state
                
                # Input goes through in_proj first
                if hasattr(layer, "in_proj"):
                    xz = layer.in_proj(x)
                    x_inner = xz[..., :d_inner]
                else:
                    x_inner = x
                
                # x_proj extracts B, C, and dt
                x_dbl = layer.x_proj(x_inner)
                
                # Split into delta, B, C
                dt = x_dbl[..., :d_inner]
                BC = x_dbl[..., d_inner:]
                
                state.B = BC[..., :d_state]
                state.C = BC[..., d_state:2*d_state]
                
                # Delta through softplus
                dt = torch.nn.functional.softplus(layer.dt_proj(dt))
                state.delta = dt
                
            # Method 2: Hook-based extraction
            else:
                state = self._extract_via_hook(layer, state, x)
                
        except Exception as e:
            logger.warning(f"Could not extract input-dependent params: {e}")
        
        return state
    
    def _extract_via_hook(
        self,
        layer: nn.Module,
        state: SSMState,
        x: torch.Tensor,
    ) -> SSMState:
        """
        Extract parameters by registering a forward hook.
        
        This is a fallback method when direct attribute access fails.
        """
        captured = {}
        
        def hook_fn(module, input, output):
            # Try to capture from output
            if isinstance(output, tuple) and len(output) > 1:
                captured["output"] = output
        
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = layer(x)
        finally:
            handle.remove()
        
        # Process captured data if available
        # (Implementation depends on specific layer structure)
        
        return state
    
    def discretize(
        self,
        A: torch.Tensor,
        delta: torch.Tensor,
        method: str = "zoh",
    ) -> torch.Tensor:
        """
        Discretize continuous A matrix using Zero-Order Hold.
        
        Ā = exp(Δ · A)
        
        For diagonal A (as in Mamba), this is elementwise:
        Ā[i, j] = exp(delta[i] * A[i, j])
        
        Args:
            A: Continuous A matrix. Shape: [d_inner, d_state].
            delta: Discretization step. Shape: [batch, seq_len, d_inner] or [d_inner].
            method: Discretization method ('zoh' or 'euler').
            
        Returns:
            torch.Tensor: Discretized Ā matrix.
            
        Note:
            For stable systems, |exp(Δ·a)| < 1 requires Re(a) < 0.
            Mamba ensures this via A = -exp(A_log).
        """
        if method == "zoh":
            # Zero-Order Hold: Ā = exp(Δ·A)
            # Handle different delta shapes
            if delta.dim() == 1:
                # delta: [d_inner], A: [d_inner, d_state]
                A_bar = torch.exp(delta.unsqueeze(-1) * A)
            elif delta.dim() == 3:
                # delta: [batch, seq_len, d_inner], A: [d_inner, d_state]
                A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            else:
                # Assume broadcasted multiplication
                A_bar = torch.exp(delta.unsqueeze(-1) * A)
                
        elif method == "euler":
            # Euler: Ā = I + Δ·A
            if delta.dim() == 1:
                A_bar = 1.0 + delta.unsqueeze(-1) * A
            else:
                A_bar = 1.0 + delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
                
        else:
            raise ValueError(f"Unknown discretization method: {method}")
        
        return A_bar
    
    def discretize_B(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
        method: str = "zoh",
    ) -> torch.Tensor:
        """
        Discretize the B matrix using Zero-Order Hold.
        
        B̄ = (Δ·A)^{-1} (exp(Δ·A) - I) · Δ·B
        
        For diagonal A, this simplifies to:
        B̄ = (exp(Δ·A) - 1) / A · B
        
        Args:
            A: Continuous A matrix. Shape: [d_inner, d_state].
            B: Input matrix. Shape: [batch, seq_len, d_state].
            delta: Discretization step. Shape: [batch, seq_len, d_inner].
            method: Discretization method.
            
        Returns:
            torch.Tensor: Discretized B̄ matrix.
        """
        if method == "zoh":
            # For diagonal A: B_bar = (exp(delta*A) - 1) / A * B
            # But A is negative, so we need to handle carefully
            delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
            
            # Numerically stable computation
            # (exp(x) - 1) / x ≈ 1 + x/2 + x²/6 + ... for small x
            exp_term = torch.exp(delta_A) - 1
            
            # Avoid division by zero
            safe_A = A.unsqueeze(0).unsqueeze(0)
            safe_A = torch.where(
                safe_A.abs() > 1e-8,
                safe_A,
                torch.ones_like(safe_A) * 1e-8,
            )
            
            scaling = exp_term / safe_A
            B_bar = scaling * B.unsqueeze(-2)  # Broadcast B
            
        elif method == "euler":
            # Euler: B̄ = Δ · B
            B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return B_bar
    
    def get_effective_A(
        self,
        layer_idx: int,
        input_tensor: torch.Tensor,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Get the effective discretized A matrix for a specific position.
        
        This is the actual Ā = exp(Δ·A) used during inference,
        where Δ is input-dependent.
        
        Args:
            layer_idx: Mamba layer index.
            input_tensor: Input for computing Δ. Shape: [batch, seq_len, d_model].
            position: Token position (-1 for last).
            
        Returns:
            torch.Tensor: Ā at the specified position.
                Shape: [batch, d_inner, d_state]
        """
        state = self.extract_all_parameters(layer_idx, input_tensor)
        
        if state.delta is None:
            raise ValueError("Could not extract delta from layer")
        
        # Get delta at position
        delta_pos = state.delta[:, position, :]  # [batch, d_inner]
        
        # Discretize A
        A_bar = self.discretize(state.A, delta_pos)
        
        return A_bar
    
    def clear_cache(self) -> None:
        """Clear cached states."""
        self._cached_states.clear()
        logger.debug("Cleared state cache")
    
    def __repr__(self) -> str:
        return (
            f"StateExtractor(\n"
            f"  wrapper={self.wrapper.model_name},\n"
            f"  device='{self.device}'\n"
            f")"
        )
