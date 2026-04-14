"""Integration tests for the public CLI modes of mimosa."""

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


def test_profile_comparison_bamm_vs_pwm(examples_dir, temp_dir):
    """Profile mode should compare bamm vs pwm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
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


def test_profile_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Profile mode should compare bamm vs bamm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.ihbcp"),
        str(examples_dir / "gata4.ihbcp"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
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


def test_motif_comparison_pwm_vs_pwm(examples_dir, temp_dir):
    """Motif mode should compare pwm vs pwm directly."""
    cmd = [
        "mimosa",
        "motif",
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


def test_motif_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Motif mode should compare bamm vs bamm directly."""
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


def test_profile_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Profile mode should compare sitega vs pwm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
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


def test_motif_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Motif mode should compare sitega vs pwm directly."""
    cmd = [
        "mimosa",
        "motif",
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


def test_motif_comparison_sitega_vs_pwm_pcc(examples_dir, temp_dir):
    """Motif mode should support PFM-based sitega vs pwm comparison."""
    cmd = [
        "mimosa",
        "motif",
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


def test_profile_comparison_sitega_vs_pwm_second_case(examples_dir, temp_dir):
    """Profile mode should handle a second sitega vs pwm example."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "sitega.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "co",
        "--permutations",
        "100",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_sitega_1(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the first scenario."""
    cmd = [
        "mimosa",
        "motif",
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


def test_motif_comparison_sitega_vs_sitega_2(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the second scenario."""
    cmd = [
        "mimosa",
        "motif",
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


def test_motif_comparison_sitega_vs_sitega_3(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the third scenario."""
    cmd = [
        "mimosa",
        "motif",
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


def test_motif_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Motif mode should compare pwm vs sitega directly."""
    cmd = [
        "mimosa",
        "motif",
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
    """Profile mode should compare two precomputed score profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "co",
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


def test_profile_comparison_accepts_dice_metric(examples_dir, temp_dir):
    """Profile mode should expose the Dice metric through CLI."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "dice",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    import json

    output = json.loads(result.stdout)
    assert output["metric"] == "dice"
    assert "score" in output


def test_profile_comparison_rejects_removed_l1sim_metric(examples_dir, temp_dir):
    """Profile CLI should reject removed metrics before running a comparison."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "l1sim",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_profile_comparison_with_empirical_logfpr_thresholding(examples_dir, temp_dir):
    """Profile mode should apply hard thresholding on empirically normalized profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "gata4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--fasta",
        str(examples_dir / "foreground.fa"),
        "--metric",
        "co",
        "--min-logfpr",
        "2",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    import json

    output = json.loads(result.stdout)
    assert output["metric"] == "co"
    assert "score" in output


def test_profile_comparison_rejects_promoter_argument(examples_dir, temp_dir):
    """Profile CLI should reject the removed promoter-calibration argument."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "gata4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--promoters",
        str(examples_dir / "background.fa"),
    ]

    result = run_cli(cmd)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_profile_comparison_invalid_kernel_range(examples_dir, temp_dir):
    """Profile mode should fail fast when kernel range contains no odd size."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--min-kernel-size",
        "4",
        "--max-kernel-size",
        "4",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with invalid kernel-size range"
    assert "odd value" in result.stderr.lower(), "Should mention odd kernel-size requirement"


def test_pipeline_with_missing_files():
    """Test pipeline behavior with missing input files."""
    cmd = [
        "mimosa",
        "profile",
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


def test_profile_comparison_rejects_corr_metric(examples_dir, temp_dir):
    """Profile CLI should reject the removed Pearson metric."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "corr",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with unsupported profile metric"
    assert "invalid choice" in result.stderr.lower()


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
