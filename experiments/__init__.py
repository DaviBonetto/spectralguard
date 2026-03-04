"""
Experiments module for validating spectral theory.
"""

from mamba_spectral.experiments.test_horizon import experiment_horizon_validation
from mamba_spectral.experiments.test_grokking import experiment_spectral_grokking
from mamba_spectral.experiments.test_security import experiment_spectral_guard

__all__ = [
    "experiment_horizon_validation",
    "experiment_spectral_grokking",
    "experiment_spectral_guard",
]
