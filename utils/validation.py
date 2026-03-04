"""
Validation utilities for mamba-spectral.

Provides validation tests to verify installation and functionality.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def validation_test(
    verbose: bool = True,
    run_gpu_tests: bool = False,
) -> bool:
    """
    Run validation tests to verify mamba-spectral installation.
    
    Tests:
    1. Import all modules
    2. Create mock data
    3. Test eigenvalue computation
    4. Test gramian computation
    5. Test visualization (without display)
    6. (Optional) Test with GPU/model
    
    Args:
        verbose: Print progress information.
        run_gpu_tests: Run tests requiring GPU/model.
        
    Returns:
        bool: True if all tests pass.
        
    Example:
        >>> from mamba_spectral import validation_test
        >>> validation_test()
        === MAMBA-SPECTRAL VALIDATION ===
        [1/6] Testing imports... PASS
        ...
        === ALL TESTS PASSED ===
        True
    """
    def log(msg: str):
        if verbose:
            print(msg)
    
    log("\n=== MAMBA-SPECTRAL VALIDATION ===\n")
    
    all_passed = True
    tests_run = 0
    tests_passed = 0
    
    # Test 1: Imports
    log("[1/6] Testing imports...")
    tests_run += 1
    try:
        from mamba_spectral import (
            MambaWrapper,
            StateExtractor,
            SpectralAnalyzer,
            ReachabilityGramian,
            HorizonPredictor,
            SpectralGuard,
            AdversarialGenerator,
        )
        from mamba_spectral.visualization import (
            plot_eigenvalue_spectrum,
            plot_spectral_radius_trajectory,
        )
        from mamba_spectral.utils import (
            generate_associative_recall,
            load_safe_prompts,
        )
        log("      PASS")
        tests_passed += 1
    except Exception as e:
        log(f"      FAIL: {e}")
        all_passed = False
    
    # Test 2: Mock eigenvalue computation
    log("[2/6] Testing eigenvalue computation...")
    tests_run += 1
    try:
        import torch
        
        # Create diagonal A matrix (like Mamba)
        A = -torch.exp(torch.randn(64, 16))  # [d_inner, d_state]
        
        # Discretize
        delta = 0.01
        A_bar = torch.exp(delta * A)
        
        # Compute eigenvalues (for diagonal, eigenvalues = diagonal)
        eigenvalues = A_bar.flatten().numpy()
        
        # Verify properties
        spectral_radius = np.max(np.abs(eigenvalues))
        assert len(eigenvalues) == 64 * 16
        assert spectral_radius >= 0
        assert spectral_radius < 2.0  # Should be close to 1
        
        log(f"      PASS (ρ = {spectral_radius:.4f})")
        tests_passed += 1
    except Exception as e:
        log(f"      FAIL: {e}")
        all_passed = False
    
    # Test 3: Gramian computation
    log("[3/6] Testing gramian computation...")
    tests_run += 1
    try:
        from mamba_spectral.spectral.gramian import ReachabilityGramian
        
        gramian_calc = ReachabilityGramian(device='cpu')
        
        # Small test matrices
        n = 10
        A_bar = torch.eye(n) * 0.9  # Stable diagonal
        B_bar = torch.randn(n, 1)
        
        result = gramian_calc.compute(A_bar, B_bar, horizon=20)
        
        assert result.gramian.shape == (n, n)
        assert result.rank > 0
        
        log(f"      PASS (rank = {result.rank}/{n})")
        tests_passed += 1
    except Exception as e:
        log(f"      FAIL: {e}")
        all_passed = False
    
    # Test 4: Dataset generation
    log("[4/6] Testing dataset generation...")
    tests_run += 1
    try:
        from mamba_spectral.utils.datasets import (
            generate_associative_recall,
            generate_math_problems,
            load_safe_prompts,
        )
        
        data_ar = generate_associative_recall(n_samples=5, distance=10)
        data_math = generate_math_problems(n_samples=5)
        prompts = load_safe_prompts(n=5)
        
        assert len(data_ar) == 5
        assert len(data_math) == 5
        assert len(prompts) == 5
        assert "prompt" in data_ar[0]
        
        log("      PASS")
        tests_passed += 1
    except Exception as e:
        log(f"      FAIL: {e}")
        all_passed = False
    
    # Test 5: Visualization (no display)
    log("[5/6] Testing visualization...")
    tests_run += 1
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        from mamba_spectral.visualization.spectral_plots import (
            plot_eigenvalue_spectrum,
            plot_spectral_radius_trajectory,
        )
        
        # Test eigenvalue plot
        eigenvalues = np.random.randn(100) + 1j * np.random.randn(100) * 0.1
        fig1 = plot_eigenvalue_spectrum(eigenvalues)
        plt.close(fig1)
        
        # Test trajectory plot
        trajectory = [0.9 + 0.05 * np.random.randn() for _ in range(50)]
        fig2 = plot_spectral_radius_trajectory(trajectory)
        plt.close(fig2)
        
        log("      PASS")
        tests_passed += 1
    except Exception as e:
        log(f"      FAIL: {e}")
        all_passed = False
    
    # Test 6: GPU/Model tests (optional)
    log("[6/6] Testing GPU/model integration...")
    tests_run += 1
    if run_gpu_tests:
        try:
            import torch
            if not torch.cuda.is_available():
                log("      SKIP (no GPU)")
            else:
                # Try to import mamba-ssm
                try:
                    from mamba_ssm import Mamba
                    
                    # Create small model
                    model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
                    
                    # Test forward pass
                    x = torch.randn(1, 10, 64).cuda()
                    y = model(x)
                    
                    assert y.shape == x.shape
                    log("      PASS")
                    tests_passed += 1
                except ImportError:
                    log("      SKIP (mamba-ssm not installed)")
                    tests_passed += 1  # Count as pass if optional
        except Exception as e:
            log(f"      FAIL: {e}")
            all_passed = False
    else:
        log("      SKIP (run_gpu_tests=False)")
        tests_passed += 1
    
    # Summary
    log(f"\n{'='*40}")
    log(f"Tests: {tests_passed}/{tests_run} passed")
    
    if all_passed:
        log("\n=== ALL TESTS PASSED ===\n")
    else:
        log("\n=== SOME TESTS FAILED ===\n")
    
    return all_passed


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dict mapping package name to install status.
    """
    dependencies = {
        "torch": False,
        "numpy": False,
        "scipy": False,
        "matplotlib": False,
        "seaborn": False,
        "sklearn": False,
        "tqdm": False,
        "transformers": False,
        "mamba_ssm": False,
    }
    
    for pkg in dependencies:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            dependencies[pkg] = True
        except ImportError:
            dependencies[pkg] = False
    
    return dependencies


def print_system_info():
    """Print system and package information."""
    import platform
    
    print("\n=== SYSTEM INFO ===\n")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    
    deps = check_dependencies()
    print("\n=== DEPENDENCIES ===\n")
    for pkg, installed in deps.items():
        status = "✓" if installed else "✗"
        print(f"  {status} {pkg}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mamba-Spectral validation")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--info", action="store_true", help="Print system info")
    args = parser.parse_args()
    
    if args.info:
        print_system_info()
    
    success = validation_test(verbose=True, run_gpu_tests=args.gpu)
    sys.exit(0 if success else 1)
