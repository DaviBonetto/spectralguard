"""
SpectralGuard: Defense against Hidden State Poisoning (HiSPA) attacks.

Monitors spectral properties of Mamba SSM during inference to detect
adversarial inputs that cause spectral collapse.

References:
    [LeMercier2026] Le Mercier et al. (2026). HiSPA: Hidden State Poisoning
                    Attacks on Mamba-based Language Models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """
    Result of a security check.
    
    Attributes:
        is_safe: Whether the prompt is considered safe.
        reason: Description of the classification.
        confidence: Confidence score (0-1).
        trajectory: Spectral radius trajectory (if computed).
        attack_location: Token index where attack was detected (if any).
        details: Additional analysis details.
    """
    is_safe: bool
    reason: str
    confidence: float = 1.0
    trajectory: Optional[List[float]] = None
    attack_location: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SafeZoneProfile:
    """
    Profile of "normal" spectral behavior.
    
    Learned from a dataset of safe prompts to establish baseline.
    
    Attributes:
        mean_radius: Mean spectral radius.
        std_radius: Standard deviation.
        min_radius: Minimum observed radius.
        lower_bound: Lower threshold (mean - 2*std).
        num_samples: Number of prompts used for learning.
    """
    mean_radius: float
    std_radius: float
    min_radius: float
    lower_bound: float
    num_samples: int


class SpectralGuard:
    """
    Defense system against Hidden State Poisoning (HiSPA) attacks.
    
    HiSPA attacks force eigenvalues λ → 0, causing spectral collapse
    and "amnesia" in the model. SpectralGuard detects this by:
    
    1. Monitoring spectral radius ρ(Ā) during prompt processing
    2. Detecting sudden drops in ρ (signature of HiSPA)
    3. Blocking prompts that exhibit collapse patterns
    
    Baseline: Paper HiSPA (Jan 2026) showed 71% CHSS degradation.
    Our contribution: First spectral-based defense mechanism.
    
    Example:
        >>> guard = SpectralGuard(model, threshold=0.3)
        >>> is_safe, reason = guard.check_prompt("Hello, how are you?")
        >>> print(f"Safe: {is_safe}, Reason: {reason}")
        
        >>> # With full details
        >>> result = guard.check_prompt(suspicious_prompt, return_details=True)
        >>> if not result.is_safe:
        ...     print(f"Attack detected at token {result.attack_location}")
    """
    
    def __init__(
        self,
        model: Union["MambaWrapper", nn.Module],
        threshold: float = 0.3,
        window_size: int = 10,
        collapse_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """
        Initialize SpectralGuard.
        
        Args:
            model: Mamba model to protect.
            threshold: Minimum acceptable spectral radius.
            window_size: Window for detecting sudden drops.
            collapse_threshold: Delta threshold for "sudden drop".
            device: Computation device.
        """
        from core.mamba_wrapper import MambaWrapper
        from spectral.eigenvalue_analyzer import SpectralAnalyzer
        
        if isinstance(model, MambaWrapper):
            self.wrapper = model
        else:
            self.wrapper = MambaWrapper(model, device=device)
        
        self.analyzer = SpectralAnalyzer(self.wrapper, device=device)
        self.device = device
        
        # Detection parameters
        self.threshold = threshold
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        
        # Learned safe zone (optional)
        self.safe_zone: Optional[SafeZoneProfile] = None
        
        # Statistics
        self._total_checks = 0
        self._blocked_count = 0
        
        logger.info(
            f"SpectralGuard initialized: threshold={threshold}, "
            f"window_size={window_size}"
        )
    
    @torch.no_grad()
    def simulate_spectral_trajectory(
        self,
        prompt: Union[str, torch.Tensor],
        layer_idx: int = 0,
        delta_value: float = 0.01,
    ) -> List[float]:
        """
        Simulate prompt processing and extract spectral trajectory.
        
        This is efficient - we only compute spectral properties,
        not full text generation.
        
        Args:
            prompt: Input text or tokens.
            layer_idx: Which layer to monitor.
            delta_value: Discretization step.
            
        Returns:
            List of spectral radius values, one per token.
        """
        trajectory = self.analyzer.track_evolution(
            prompt,
            layer_idx=layer_idx,
            save_every=1,
            delta_value=delta_value,
        )
        
        return trajectory.spectral_radius
    
    def detect_collapse(
        self,
        trajectory: List[float],
    ) -> Tuple[bool, float, Optional[int]]:
        """
        Detect spectral collapse (signature of HiSPA attack).
        
        Criteria:
        1. Sudden drop: Δρ > collapse_threshold within window
        2. Final radius: ρ < threshold
        3. High variance: Indicates instability
        
        Args:
            trajectory: List of spectral radius values.
            
        Returns:
            Tuple of:
                - is_attack: Whether collapse was detected
                - confidence: Detection confidence (0-1)
                - attack_location: Token index where attack started
        """
        if len(trajectory) < 2:
            return False, 0.0, None
        
        trajectory = np.array(trajectory)
        is_attack = False
        confidence = 0.0
        attack_location = None
        
        # Check 1: Sudden drop within window
        for i in range(len(trajectory) - self.window_size):
            window = trajectory[i:i + self.window_size]
            drop = window[0] - window[-1]
            
            if drop > self.collapse_threshold:
                is_attack = True
                attack_location = i
                confidence = min(1.0, drop / self.collapse_threshold)
                break
        
        # Check 2: Final radius too low
        final_radius = trajectory[-1]
        if final_radius < self.threshold:
            if not is_attack:
                is_attack = True
                attack_location = len(trajectory) - 1
            confidence = max(confidence, 1.0 - final_radius / self.threshold)
        
        # Check 3: High variance (instability)
        variance = np.var(trajectory)
        if variance > 0.1:  # High variance threshold
            confidence = min(1.0, confidence + 0.2)
        
        return is_attack, float(confidence), attack_location
    
    def check_prompt(
        self,
        prompt: Union[str, torch.Tensor],
        return_details: bool = False,
        layer_idx: int = 0,
    ) -> Union[Tuple[bool, str], SecurityCheckResult]:
        """
        Check if a prompt is safe to process.
        
        Main API for SpectralGuard.
        
        Args:
            prompt: Input text or tokens.
            return_details: If True, return full SecurityCheckResult.
            layer_idx: Layer to analyze.
            
        Returns:
            If return_details=False: Tuple of (is_safe, reason)
            If return_details=True: SecurityCheckResult with full analysis
            
        Example:
            >>> is_safe, reason = guard.check_prompt("Hello!")
            >>> if not is_safe:
            ...     print(f"Blocked: {reason}")
        """
        self._total_checks += 1
        
        # Simulate trajectory
        try:
            trajectory = self.simulate_spectral_trajectory(
                prompt, layer_idx=layer_idx
            )
        except Exception as e:
            logger.error(f"Error simulating trajectory: {e}")
            result = SecurityCheckResult(
                is_safe=True,  # Fail open
                reason="analysis_error",
                confidence=0.0,
            )
            return (True, "analysis_error") if not return_details else result
        
        # Detect collapse
        is_attack, confidence, attack_location = self.detect_collapse(trajectory)
        
        # Determine reason
        if not is_attack:
            reason = "safe"
            is_safe = True
        elif attack_location is not None and trajectory[attack_location] < self.threshold:
            reason = "spectral_collapse"
            is_safe = False
            self._blocked_count += 1
        else:
            reason = "spectral_instability"
            is_safe = False
            self._blocked_count += 1
        
        # Check against learned safe zone if available
        if is_safe and self.safe_zone is not None:
            mean_radius = np.mean(trajectory)
            if mean_radius < self.safe_zone.lower_bound:
                is_safe = False
                reason = "below_safe_zone"
                confidence = max(confidence, 0.7)
                self._blocked_count += 1
        
        if return_details:
            return SecurityCheckResult(
                is_safe=is_safe,
                reason=reason,
                confidence=confidence,
                trajectory=trajectory,
                attack_location=attack_location,
                details={
                    "mean_radius": float(np.mean(trajectory)),
                    "min_radius": float(np.min(trajectory)),
                    "max_radius": float(np.max(trajectory)),
                    "variance": float(np.var(trajectory)),
                },
            )
        
        return is_safe, reason
    
    @torch.no_grad()
    def learn_safe_zones(
        self,
        safe_prompts: List[str],
        layer_idx: int = 0,
        show_progress: bool = True,
    ) -> SafeZoneProfile:
        """
        Learn the distribution of "normal" spectral behavior.
        
        Processes a dataset of known-safe prompts to establish
        baseline spectral radius statistics.
        
        Args:
            safe_prompts: List of safe prompts for calibration.
            layer_idx: Layer to analyze.
            show_progress: Show progress bar.
            
        Returns:
            SafeZoneProfile with learned statistics.
            
        Example:
            >>> safe_prompts = load_safe_prompts(n=1000)
            >>> profile = guard.learn_safe_zones(safe_prompts)
            >>> print(f"Safe zone: μ={profile.mean_radius:.3f} ± {profile.std_radius:.3f}")
        """
        from tqdm import tqdm
        
        all_radii = []
        
        iterator = safe_prompts
        if show_progress:
            iterator = tqdm(iterator, desc="Learning safe zones")
        
        for prompt in iterator:
            try:
                trajectory = self.simulate_spectral_trajectory(
                    prompt, layer_idx=layer_idx
                )
                mean_radius = float(np.mean(trajectory))
                all_radii.append(mean_radius)
            except Exception as e:
                logger.warning(f"Failed to process prompt: {e}")
        
        if not all_radii:
            raise ValueError("No prompts could be processed")
        
        radii = np.array(all_radii)
        
        self.safe_zone = SafeZoneProfile(
            mean_radius=float(np.mean(radii)),
            std_radius=float(np.std(radii)),
            min_radius=float(np.min(radii)),
            lower_bound=float(np.mean(radii) - 2 * np.std(radii)),
            num_samples=len(radii),
        )
        
        logger.info(
            f"Learned safe zone from {len(radii)} prompts: "
            f"μ={self.safe_zone.mean_radius:.4f}, σ={self.safe_zone.std_radius:.4f}"
        )
        
        return self.safe_zone
    
    def batch_check(
        self,
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[SecurityCheckResult]:
        """
        Check multiple prompts.
        
        Args:
            prompts: List of prompts to check.
            show_progress: Show progress bar.
            
        Returns:
            List of SecurityCheckResult for each prompt.
        """
        from tqdm import tqdm
        
        results = []
        
        iterator = prompts
        if show_progress:
            iterator = tqdm(iterator, desc="Security check")
        
        for prompt in iterator:
            result = self.check_prompt(prompt, return_details=True)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with check counts and block rates.
        """
        block_rate = (
            self._blocked_count / self._total_checks
            if self._total_checks > 0 else 0.0
        )
        
        return {
            "total_checks": self._total_checks,
            "blocked_count": self._blocked_count,
            "block_rate": block_rate,
            "threshold": self.threshold,
            "window_size": self.window_size,
            "safe_zone_learned": self.safe_zone is not None,
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self._total_checks = 0
        self._blocked_count = 0
    
    def adjust_threshold(self, new_threshold: float) -> None:
        """
        Adjust the detection threshold.
        
        Lower threshold = more sensitive (more false positives)
        Higher threshold = less sensitive (more false negatives)
        
        Args:
            new_threshold: New spectral radius threshold.
        """
        old = self.threshold
        self.threshold = new_threshold
        logger.info(f"Threshold adjusted: {old} -> {new_threshold}")
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"SpectralGuard(\n"
            f"  threshold={self.threshold},\n"
            f"  window_size={self.window_size},\n"
            f"  total_checks={stats['total_checks']},\n"
            f"  block_rate={stats['block_rate']:.2%}\n"
            f")"
        )
