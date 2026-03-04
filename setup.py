from setuptools import find_packages, setup


setup(
    name="spectralguard",
    version="0.2.0",
    description="SpectralGuard: spectral safety monitor for recurrent and hybrid foundation models",
    long_description=(
        "SpectralGuard provides a stable monitoring API for spectral hazard detection: "
        "monitor(prompt, hidden_states) -> (is_safe, spectral_hazard_score)."
    ),
    long_description_content_type="text/plain",
    packages=find_packages(include=["spectralguard", "spectralguard.*"]),
    install_requires=[
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
        ]
    },
    python_requires=">=3.9",
)
