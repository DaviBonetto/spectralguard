import subprocess
import sys

import pytest


def test_adaptive_v4_stealth_cli_help() -> None:
    pytest.importorskip("transformers")
    cmd = [
        sys.executable,
        "-m",
        "mamba_spectral.scripts.run_adaptive_v4_stealth",
        "--help",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "dual lexical controls" in proc.stdout
