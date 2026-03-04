"""
Security module for SpectralGuard defense against HiSPA attacks.
"""

from mamba_spectral.security.spectral_guard import SpectralGuard
from mamba_spectral.security.adversarial_gen import AdversarialGenerator

__all__ = ["SpectralGuard", "AdversarialGenerator"]
