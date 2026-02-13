"""
Integration tests for mimosa based on examples/run.sh scenarios.

These tests cover various command-line scenarios from the example script,
excluding any tests with .fa and .fasta files as specified.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run CLI through current Python env to avoid global PATH contamination."""
    args = cmd[1:] if cmd and cmd[0] == "mimosa" else cmd
    return subprocess.run([sys.executable, "-m", "mimosa.cli", *args], capture_output=True, text=True)


@pytest.fixture
def examples_dir():
    """Path to examples directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


def test_motali_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Test motali comparison: pwm vs sitega"""
    cmd = [
        "mimosa",
        "motali",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "sitega",
        "-v",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_bamm_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: bamm vs pwm"""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "myog.ihbcp"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "pwm",
        "--permutations",
        "100",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Verify output contains expected keys
    import json

    output = json.loads(result.stdout)
    expected_keys = ["query", "target", "score", "offset", "orientation", "metric"]
    for key in expected_keys:
        assert key in output, f"Missing key '{key}' in output"


def test_motif_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Test motif comparison: bamm vs bamm"""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "gata2.ihbcp"),
        str(examples_dir / "gata4.ihbcp"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
        "--permutations",
        "100",
        "--metric",
        "cj",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Verify output contains expected keys
    import json

    output = json.loads(result.stdout)
    expected_keys = ["query", "target", "score", "offset", "orientation", "metric"]
    for key in expected_keys:
        assert key in output, f"Missing key '{key}' in output"


def test_tomtom_like_comparison_pwm_vs_pwm(examples_dir, temp_dir):
    """Test tomtom-like comparison: pwm vs pwm"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "pif4.meme"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--metric",
        "ed",
        "--permutations",
        "1000",
        "--permute-rows",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Verify output contains expected keys
    import json

    output = json.loads(result.stdout)
    expected_keys = ["query", "target", "score", "offset", "orientation", "metric"]
    for key in expected_keys:
        assert key in output, f"Missing key '{key}' in output"


def test_tomtom_like_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Test tomtom-like comparison: bamm vs bamm"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "gata2.ihbcp"),
        str(examples_dir / "gata4.ihbcp"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
        "--permutations",
        "1000",
        "--metric",
        "ed",
        "-v",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Verify output contains expected keys
    import json

    output = json.loads(result.stdout)
    expected_keys = ["query", "target", "score", "offset", "orientation", "metric"]
    for key in expected_keys:
        assert key in output, f"Missing key '{key}' in output"


def test_motif_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: sitega vs pwm"""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--permutations",
        "100",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs pwm"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "sitega_gata2.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "ed",
        "--permutations",
        "100",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_pwm_pcc(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs pwm with PCC metric"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "pcc",
        "--permutations",
        "100",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_sequence_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: sitega vs pwm"""
    # Note: This test uses scores_1.fasta and scores_2.fasta which are not excluded
    # according to the requirements since they don't have the .fa/.fasta extension
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "cj",
        "--permutations",
        "100",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_1(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (first scenario)"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "pcc",
        "--permutations",
        "1000",
        "--permute-rows",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_2(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (second scenario)"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--permutations",
        "1000",
        "--permute-rows",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_3(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (third scenario)"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_stat6.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--permutations",
        "1000",
        "--permute-rows",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Test tomtom-like comparison: pwm vs sitega"""
    cmd = [
        "mimosa",
        "tomtom-like",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--permutations",
        "1000",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_profile_comparison_basic(examples_dir, temp_dir):
    """Test profile comparison between two score profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--metric",
        "cj",
        "--permutations",
        "100",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Verify output contains expected keys
    import json

    output = json.loads(result.stdout)
    expected_keys = ["query", "target", "score", "offset", "orientation", "metric"]
    for key in expected_keys:
        assert key in output, f"Missing key '{key}' in output"


def test_pipeline_with_missing_files():
    """Test pipeline behavior with missing input files."""
    cmd = [
        "mimosa",
        "motif",
        "nonexistent1.ihbcp",
        "nonexistent2.ihbcp",
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with missing files"
    assert "file not found" in result.stderr.lower(), "Should mention missing file"


def test_pipeline_with_invalid_mode():
    """Test pipeline behavior with invalid mode."""
    cmd = [
        "mimosa",
        "invalid_mode",
        "file1",
        "file2",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with invalid mode"


if __name__ == "__main__":
    pytest.main([__file__])
