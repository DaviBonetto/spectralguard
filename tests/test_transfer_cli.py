import subprocess
import sys

import pytest


def test_transfer_script_cli_help() -> None:
    pytest.importorskip("transformers")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_stealthy_transfer_zamba2",
        "--help",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "AdaptiveHiSPA v4 -> Zamba2" in proc.stdout
