"""
Integration tests for yamcot based on examples/run.sh scenarios.

These tests cover various command-line scenarios from the example script,
excluding any tests with .fa and .fasta files as specified.
"""
import subprocess
from pathlib import Path

import pytest


def test_motif_comparison_bamm_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: bamm vs pwm"""
    cmd = [
        "yamcot", "motif",
        str(examples_dir / "myog"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "bamm",
        "--model2-type", "pwm",
        "--perm", "500",
        "--metric", "co"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Test motif comparison: bamm vs bamm"""
    cmd = [
        "yamcot", "motif",
        str(examples_dir / "gata2"),
        str(examples_dir / "gata4"),
        "--model1-type", "bamm",
        "--model2-type", "bamm",
        "--perm", "1000",
        "--metric", "cj"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: sitega vs pwm"""
    cmd = [
        "yamcot", "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "sitega",
        "--model2-type", "pwm",
        "--perm", "500",
        "--metric", "co"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs pwm"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "sitega_gata2.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "sitega",
        "--model2-type", "pwm",
        "--metric", "ed",
        "--permutations", "500"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_pwm_vs_pwm(examples_dir, temp_dir):
    """Test tomtom-like comparison: pwm vs pwm"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "pif4.meme"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "pwm",
        "--model2-type", "pwm",
        "--metric", "ed",
        "--permutations", "10000",
        "--permute-rows"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_pwm_pcc(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs pwm with PCC metric"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "sitega",
        "--model2-type", "pwm",
        "--metric", "pcc",
        "--permutations", "1000",
        "--pfm-mode"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_sequence_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Test motif comparison: sitega vs pwm"""
    # Note: This test uses scores_1.fasta and scores_2.fasta which are not excluded
    # according to the requirements since they don't have the .fa/.fasta extension
    cmd = [
        "yamcot", "motif",
        str(examples_dir / "sitega.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type", "sitega",
        "--model2-type", "pwm",
        "--metric", "cj",
        "--perm", "1000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_1(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (first scenario)"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type", "sitega",
        "--model2-type", "sitega",
        "--metric", "pcc",
        "--permutations", "10000",
        "--permute-rows",
        "--pfm-mode"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_2(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (second scenario)"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type", "sitega",
        "--model2-type", "sitega",
        "--metric", "ed",
        "--permutations", "10000",
        "--permute-rows",
        "--pfm-mode"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_sitega_vs_sitega_3(examples_dir, temp_dir):
    """Test tomtom-like comparison: sitega vs sitega (third scenario)"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_stat6.mat"),
        "--model1-type", "sitega",
        "--model2-type", "sitega",
        "--metric", "ed",
        "--permutations", "10000",
        "--permute-rows",
        "--pfm-mode"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Test tomtom-like comparison: pwm vs sitega"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type", "pwm",
        "--model2-type", "sitega",
        "--metric", "ed",
        "--permutations", "1000",
        "--pfm-mode"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_tomtom_like_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Test tomtom-like comparison: bamm vs bamm"""
    cmd = [
        "yamcot", "tomtom-like",
        str(examples_dir / "gata2"),
        str(examples_dir / "gata4"),
        "--model1-type", "bamm",
        "--model2-type", "bamm",
        "--permutations", "1000",
        "--metric", "ed",
        "-v"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motali_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Test motali comparison: pwm vs sitega"""
    cmd = [
        "yamcot", "motali",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type", "pwm",
        "--model2-type", "sitega",
        "-v"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__])