"""
Unit tests for mamba-spectral core functionality.
"""

import pytest
import numpy as np
import torch


class TestSpectralAnalyzer:
    """Tests for SpectralAnalyzer class."""
    
    def test_compute_eigenvalues_diagonal(self):
        """Test eigenvalue computation for diagonal matrix."""
        # Diagonal matrix
        A = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        
        # For diagonal, eigenvalues = diagonal
        eigenvalues = A.numpy()
        
        assert len(eigenvalues) == 5
        assert eigenvalues[0] == pytest.approx(0.9)
    
    def test_spectral_radius(self):
        """Test spectral radius computation."""
        eigenvalues = np.array([0.9, -0.8, 0.5 + 0.5j, 0.5 - 0.5j])
        magnitudes = np.abs(eigenvalues)
        spectral_radius = np.max(magnitudes)
        
        assert spectral_radius == pytest.approx(0.9, rel=0.01)
    
    def test_discretization_zoh(self):
        """Test Zero-Order Hold discretization."""
        A = torch.tensor([-1.0, -2.0, -0.5])  # Continuous (stable)
        delta = 0.1
        
        A_bar = torch.exp(delta * A)
        
        # All should be in (0, 1)
        assert torch.all(A_bar > 0)
        assert torch.all(A_bar < 1)
    
    def test_discretization_stability(self):
        """Test that discretization preserves stability."""
        # A = -exp(A_log) ensures negative eigenvalues (stable)
        A_log = torch.randn(10)
        A = -torch.exp(A_log)
        
        # Any positive delta should give stable Ā
        delta = 0.01
        A_bar = torch.exp(delta * A)
        
        spectral_radius = torch.max(torch.abs(A_bar)).item()
        assert spectral_radius < 1.0  # Stable


class TestReachabilityGramian:
    """Tests for ReachabilityGramian class."""
    
    def test_gramian_shape(self):
        """Test gramian has correct shape."""
        from spectral.gramian import ReachabilityGramian
        
        n = 8
        A_bar = torch.eye(n) * 0.9
        B_bar = torch.randn(n, 1)
        
        calc = ReachabilityGramian(device='cpu')
        result = calc.compute(A_bar, B_bar, horizon=10)
        
        assert result.gramian.shape == (n, n)
    
    def test_gramian_symmetric(self):
        """Test gramian is symmetric."""
        from spectral.gramian import ReachabilityGramian
        
        n = 5
        A_bar = torch.eye(n) * 0.9
        B_bar = torch.randn(n, 1)
        
        calc = ReachabilityGramian(device='cpu')
        result = calc.compute(A_bar, B_bar, horizon=10)
        
        # W_R should be symmetric
        assert np.allclose(result.gramian, result.gramian.T, atol=1e-6)
    
    def test_gramian_rank(self):
        """Test gramian rank computation."""
        from spectral.gramian import ReachabilityGramian
        
        n = 4
        A_bar = torch.diag(torch.tensor([0.9, 0.8, 0.7, 0.6]))
        B_bar = torch.tensor([[1.0], [1.0], [0.0], [0.0]])  # Only affects 2 states
        
        calc = ReachabilityGramian(device='cpu')
        result = calc.compute(A_bar, B_bar, horizon=50)
        
        # Should not be full rank
        assert result.rank <= n


class TestDatasets:
    """Tests for dataset generation utilities."""
    
    def test_associative_recall_format(self):
        """Test associative recall dataset format."""
        from utils.datasets import generate_associative_recall
        
        data = generate_associative_recall(n_samples=5, distance=10)
        
        assert len(data) == 5
        assert "prompt" in data[0]
        assert "answer" in data[0]
        assert len(data[0]["answer"]) == 3  # Default key_length
    
    def test_math_problems_format(self):
        """Test math problems dataset format."""
        from utils.datasets import generate_math_problems
        
        data = generate_math_problems(n_samples=5, difficulty="easy")
        
        assert len(data) == 5
        assert "prompt" in data[0]
        assert "answer" in data[0]
    
    def test_safe_prompts_count(self):
        """Test safe prompts loading."""
        from utils.datasets import load_safe_prompts
        
        prompts = load_safe_prompts(n=20)
        
        assert len(prompts) == 20
        assert all(isinstance(p, str) for p in prompts)


class TestVisualization:
    """Tests for visualization functions (no display)."""
    
    def test_eigenvalue_spectrum_plot(self):
        """Test eigenvalue spectrum plot creation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.spectral_plots import plot_eigenvalue_spectrum
        
        eigenvalues = np.random.randn(50) + 1j * np.random.randn(50) * 0.1
        fig = plot_eigenvalue_spectrum(eigenvalues)
        
        assert fig is not None
        plt.close(fig)
    
    def test_trajectory_plot(self):
        """Test trajectory plot creation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.spectral_plots import plot_spectral_radius_trajectory
        
        trajectory = [0.9 + 0.01 * i for i in range(20)]
        fig = plot_spectral_radius_trajectory(trajectory)
        
        assert fig is not None
        plt.close(fig)


class TestValidation:
    """Tests for validation utilities."""
    
    def test_validation_runs(self):
        """Test that validation test runs without errors."""
        from utils.validation import validation_test
        
        # Should not raise
        result = validation_test(verbose=False, run_gpu_tests=False)
        
        assert isinstance(result, bool)
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        from utils.validation import check_dependencies
        
        deps = check_dependencies()
        
        assert "torch" in deps
        assert "numpy" in deps
        assert deps["torch"] == True  # Should be installed
        assert deps["numpy"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
