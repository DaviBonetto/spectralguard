"""
Mamba-Spectral: Spectral Foundations of Reasoning in State Space Models

A library for spectral analysis of Mamba SSM architecture, including:
- Eigenvalue extraction and tracking
- Reasoning horizon prediction via reachability gramian
- Security framework (SpectralGuard) against HiSPA attacks

Author: Research Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Core imports
from core.mamba_wrapper import MambaWrapper
from core.state_extractor import StateExtractor

# Spectral analysis
from spectral.eigenvalue_analyzer import SpectralAnalyzer
from spectral.gramian import ReachabilityGramian
from spectral.horizon_predictor import HorizonPredictor

# Security
from security.spectral_guard import SpectralGuard
from security.adversarial_gen import AdversarialGenerator

# Convenience function for validation
from utils.validation import validation_test

__all__ = [
    # Core
    "MambaWrapper",
    "StateExtractor",
    # Spectral
    "SpectralAnalyzer",
    "ReachabilityGramian",
    "HorizonPredictor",
    # Security
    "SpectralGuard",
    "AdversarialGenerator",
    # Utils
    "validation_test",
]
