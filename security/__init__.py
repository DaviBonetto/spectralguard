"""
Security module for SpectralGuard defense against HiSPA attacks.
"""

from security.spectral_guard import SpectralGuard
from security.adversarial_gen import AdversarialGenerator

__all__ = ["SpectralGuard", "AdversarialGenerator"]
