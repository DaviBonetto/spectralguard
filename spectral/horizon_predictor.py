"""
Horizon Predictor for Mamba SSM.

Predicts the mathematical limit of reasoning (Spectral Horizon) using
eigenvalue analysis and reachability gramian.

References:
    [Gu2023] Spectral analysis of state space models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """
    Result of horizon prediction.
    
    Attributes:
        horizon: Predicted reasoning horizon (number of tokens).
        gramian_rank: Rank of reachability gramian at horizon.
        min_singular_value: Smallest singular value at horizon.
        spectral_radius: Spectral radius of Ā.
        is_reachable: Whether full state is reachable.
        confidence: Confidence score (0-1).
        method: Method used for prediction.
    """
    horizon: int
    gramian_rank: int
    min_singular_value: float
    spectral_radius: float
    is_reachable: bool
    confidence: float = 1.0
    method: str = "gramian"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon": self.horizon,
            "gramian_rank": self.gramian_rank,
            "min_singular_value": self.min_singular_value,
            "spectral_radius": self.spectral_radius,
            "is_reachable": self.is_reachable,
            "confidence": self.confidence,
            "method": self.method,
        }


class HorizonPredictor:
    """
    Predicts the Spectral Horizon - the mathematical limit of reasoning.
    
    The Spectral Horizon is determined by when the reachability gramian
    becomes singular, indicating that the model can no longer access
    information from earlier in the sequence.
    
    Theory:
    - W_R(T) = Σ_{k=0}^{T-1} Ā^k · B̄ · B̄^T · (Ā^T)^k
    - When σ_min(W_R) < ε, the horizon H = T is reached
    - Beyond H, reasoning about earlier context becomes unreliable
    
    Example:
        >>> predictor = HorizonPredictor(model)
        >>> result = predictor.predict_horizon(
        ...     "The capital of France is",
        ...     max_horizon=1000,
        ... )
        >>> print(f"Predicted horizon: {result.horizon} tokens")
        >>> print(f"Is fully reachable: {result.is_reachable}")
    """
    
    def __init__(
        self,
        model: Union["MambaWrapper", nn.Module],
        device: str = "cuda",
    ) -> None:
        """
        Initialize the horizon predictor.
        
        Args:
            model: Mamba model (MambaWrapper or raw model).
            device: Computation device.
        """
        from mamba_spectral.core.mamba_wrapper import MambaWrapper
        from mamba_spectral.core.state_extractor import StateExtractor
        from mamba_spectral.spectral.eigenvalue_analyzer import SpectralAnalyzer
        from mamba_spectral.spectral.gramian import ReachabilityGramian
        
        if isinstance(model, MambaWrapper):
            self.wrapper = model
        else:
            self.wrapper = MambaWrapper(model, device=device)
        
        self.extractor = StateExtractor(self.wrapper)
        self.analyzer = SpectralAnalyzer(self.wrapper, device=device)
        self.gramian_calc = ReachabilityGramian(device=device)
        self.device = device
        
        logger.info(f"HorizonPredictor initialized on {device}")
    
    def predict_horizon(
        self,
        prompt: Union[str, torch.Tensor],
        layer_idx: int = 0,
        threshold: float = 1e-6,
        max_horizon: int = 1000,
        delta_value: float = 0.01,
        method: str = "gramian",
    ) -> HorizonResult:
        """
        Predict the reasoning horizon for a given prompt.
        
        Computes the Spectral Horizon by analyzing when the reachability
        gramian becomes singular (σ_min < threshold).
        
        Args:
            prompt: Input text or token IDs.
            layer_idx: Which Mamba layer to analyze.
            threshold: Singular value threshold for horizon detection.
            max_horizon: Maximum horizon to search.
            delta_value: Discretization step.
            method: Prediction method:
                - 'gramian': Use reachability gramian (most accurate)
                - 'spectral': Use spectral radius decay (faster)
                - 'hybrid': Combine both methods
                
        Returns:
            HorizonResult with predicted horizon and analysis.
            
        Example:
            >>> result = predictor.predict_horizon(
            ...     "The key is APPLE. [distraction] Query: key was?",
            ...     threshold=1e-6,
            ... )
            >>> if result.horizon < 100:
            ...     print("Warning: Limited reasoning capacity!")
        """
        # Get SSM matrices
        A = self.extractor.extract_A_matrix(layer_idx)
        delta = torch.tensor(delta_value, device=self.device)
        A_bar = self.extractor.discretize(A, delta)
        
        # Create synthetic B (identity-like for analysis)
        d_state = A.shape[-1]
        B_bar = torch.eye(d_state, device=self.device, dtype=A_bar.dtype)[:, :min(d_state, 10)]
        
        # Compute spectral radius
        eigenvalues = self.analyzer.compute_eigenvalues(A_bar)
        spectral_radius = self.analyzer.spectral_radius(eigenvalues)
        
        if method == "gramian":
            return self._predict_via_gramian(
                A_bar, B_bar, threshold, max_horizon, spectral_radius
            )
        elif method == "spectral":
            return self._predict_via_spectral(
                A_bar, spectral_radius, threshold, max_horizon
            )
        elif method == "hybrid":
            return self._predict_hybrid(
                A_bar, B_bar, spectral_radius, threshold, max_horizon
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_via_gramian(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        threshold: float,
        max_horizon: int,
        spectral_radius: float,
    ) -> HorizonResult:
        """Predict horizon using reachability gramian."""
        result = self.gramian_calc.compute(
            A_bar, B_bar,
            horizon=max_horizon,
            threshold=threshold,
            track_singular_values=True,
            show_progress=False,
        )
        
        if result.horizon_reached is not None:
            horizon = result.horizon_reached
            confidence = 0.95
        else:
            # Horizon not reached within max_horizon
            horizon = max_horizon
            confidence = 0.5
        
        return HorizonResult(
            horizon=horizon,
            gramian_rank=result.rank,
            min_singular_value=result.min_singular_value,
            spectral_radius=spectral_radius,
            is_reachable=result.is_full_rank,
            confidence=confidence,
            method="gramian",
        )
    
    def _predict_via_spectral(
        self,
        A_bar: torch.Tensor,
        spectral_radius: float,
        threshold: float,
        max_horizon: int,
    ) -> HorizonResult:
        """
        Predict horizon using spectral radius decay.
        
        For ρ(Ā) < 1, memory decays as ρ^k.
        Horizon ≈ log(threshold) / log(ρ)
        """
        if spectral_radius >= 1.0:
            # Marginally stable or unstable - infinite theoretical horizon
            horizon = max_horizon
            confidence = 0.3  # Low confidence for edge case
        else:
            # Decay estimate: ρ^H < threshold
            # H = log(threshold) / log(ρ)
            try:
                horizon = int(np.log(threshold) / np.log(spectral_radius))
                horizon = min(horizon, max_horizon)
                horizon = max(horizon, 1)
                confidence = 0.7
            except (ValueError, ZeroDivisionError):
                horizon = max_horizon
                confidence = 0.2
        
        return HorizonResult(
            horizon=horizon,
            gramian_rank=-1,  # Not computed
            min_singular_value=-1.0,
            spectral_radius=spectral_radius,
            is_reachable=(spectral_radius >= 0.9),
            confidence=confidence,
            method="spectral",
        )
    
    def _predict_hybrid(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        spectral_radius: float,
        threshold: float,
        max_horizon: int,
    ) -> HorizonResult:
        """Combine gramian and spectral methods."""
        # First, quick spectral estimate
        spectral_result = self._predict_via_spectral(
            A_bar, spectral_radius, threshold, max_horizon
        )
        
        # If spectral suggests short horizon, verify with gramian
        if spectral_result.horizon < max_horizon // 2:
            search_horizon = min(spectral_result.horizon * 2, max_horizon)
            gramian_result = self._predict_via_gramian(
                A_bar, B_bar, threshold, search_horizon, spectral_radius
            )
            
            # Use gramian result if confident
            if gramian_result.confidence > spectral_result.confidence:
                gramian_result.method = "hybrid"
                return gramian_result
        
        spectral_result.method = "hybrid"
        return spectral_result
    
    def compute_reachability_gramian(
        self,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        horizon: int = 100,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute the Reachability Gramian.
        
        W_R(T) = Σ_{k=0}^{T-1} Ā^k · B̄ · B̄^T · (Ā^T)^k
        
        Args:
            A_bar: Discretized transition matrix.
            B_bar: Discretized input matrix.
            horizon: Number of steps.
            
        Returns:
            Tuple of (gramian matrix, singular values at each step).
        """
        result = self.gramian_calc.compute(
            A_bar, B_bar, horizon,
            track_singular_values=True,
        )
        return result.gramian, result.singular_values
    
    @torch.no_grad()
    def adversarial_cot_generator(
        self,
        base_prompt: str,
        target_horizon: int = 50,
        n_candidates: int = 10,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate Chain-of-Thought that DEGRADES reasoning capacity.
        
        This demonstrates that CoT can be adversarial by finding
        token sequences that minimize the spectral radius.
        
        The idea:
        - Normal CoT keeps ρ(Ā) high (preserves memory)
        - Adversarial CoT forces ρ(Ā) → 0 (causes forgetting)
        
        Note:
            Full gradient-based optimization is complex.
            This PoC uses heuristic search over candidate tokens.
        
        Args:
            base_prompt: Starting prompt.
            target_horizon: Target (reduced) horizon.
            n_candidates: Number of candidates to try per position.
            layer_idx: Layer to analyze.
            
        Returns:
            Dictionary with:
                - adversarial_prompt: Generated adversarial CoT
                - spectral_trajectory: ρ values over tokens
                - original_radius: ρ of original prompt
                - final_radius: ρ after adversarial tokens
        """
        if self.wrapper.tokenizer is None:
            raise ValueError("Tokenizer required for CoT generation")
        
        # Encode base prompt
        base_ids = self.wrapper.tokenizer.encode(base_prompt, return_tensors="pt")
        base_ids = base_ids.to(self.device)
        
        # Track original spectral radius
        trajectory_orig = self.analyzer.track_evolution(base_ids, layer_idx)
        original_radius = trajectory_orig.spectral_radius[-1]
        
        # Heuristic: Find tokens that tend to reduce spectral radius
        # These are typically: periods, padding, special characters
        adversarial_tokens = []
        
        # Common "forgetting" tokens (heuristic)
        forget_tokens = [
            "...", "   ", "\n\n", "---", "___",
            "[END]", "<pad>", "~", "...", "  ",
        ]
        
        forget_ids = []
        for token in forget_tokens:
            try:
                ids = self.wrapper.tokenizer.encode(token, add_special_tokens=False)
                forget_ids.extend(ids)
            except Exception:
                pass
        
        # Remove duplicates
        forget_ids = list(set(forget_ids))[:n_candidates]
        
        # Build adversarial sequence
        current_ids = base_ids.clone()
        spectral_trajectory = [original_radius]
        
        for step in range(min(target_horizon, 20)):  # Limit steps for PoC
            best_token = None
            lowest_radius = float('inf')
            
            for token_id in forget_ids:
                # Try appending this token
                test_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token_id]], device=self.device)
                ], dim=1)
                
                # Measure spectral radius
                try:
                    A = self.extractor.extract_A_matrix(layer_idx)
                    A_bar = self.extractor.discretize(A, torch.tensor(0.01))
                    eigenvalues = self.analyzer.compute_eigenvalues(A_bar)
                    radius = self.analyzer.spectral_radius(eigenvalues)
                    
                    if radius < lowest_radius:
                        lowest_radius = radius
                        best_token = token_id
                except Exception:
                    continue
            
            if best_token is not None:
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[best_token]], device=self.device)
                ], dim=1)
                adversarial_tokens.append(best_token)
                spectral_trajectory.append(lowest_radius)
        
        # Decode result
        adversarial_prompt = self.wrapper.tokenizer.decode(current_ids[0])
        
        return {
            "adversarial_prompt": adversarial_prompt,
            "spectral_trajectory": spectral_trajectory,
            "original_radius": original_radius,
            "final_radius": spectral_trajectory[-1] if spectral_trajectory else original_radius,
            "tokens_added": len(adversarial_tokens),
        }
    
    def analyze_prompt_horizon(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, HorizonResult]:
        """
        Analyze horizon across multiple layers.
        
        Args:
            prompt: Input text.
            layer_indices: Layers to analyze (None = all).
            
        Returns:
            Dictionary mapping layer index to HorizonResult.
        """
        layers = self.wrapper.get_mamba_layers()
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        
        results = {}
        for idx in layer_indices:
            try:
                result = self.predict_horizon(prompt, layer_idx=idx)
                results[idx] = result
            except Exception as e:
                logger.warning(f"Could not analyze layer {idx}: {e}")
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"HorizonPredictor(\n"
            f"  model='{self.wrapper.model_name}',\n"
            f"  device='{self.device}'\n"
            f")"
        )
