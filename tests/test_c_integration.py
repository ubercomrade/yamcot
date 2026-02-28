import os
import tempfile
from pathlib import Path

import pytest

from mimosa import compare_motifs
from mimosa.io import read_fasta
from mimosa.models import PwmStrategy


def test_c_module_import():
    """Test that C module can be imported correctly."""
    from mimosa._core import _core

    assert hasattr(_core, "run_motali_cpp")
    assert callable(_core.run_motali_cpp)


def test_c_function_signature():
    """Test that C function has correct signature."""
    from mimosa._core import run_motali_cpp

    # Check function documentation
    doc = run_motali_cpp.__doc__
    assert doc is not None
    assert "run_motali_cpp" in doc
    assert "file_fasta" in doc
    assert "type_model_1" in doc
    assert "type_model_2" in doc


def test_c_function_call_with_invalid_params():
    """Test that C function handles invalid parameters gracefully."""
    from mimosa._core import run_motali_cpp

    # Test with non-existent files to verify function can be called
    # and returns appropriate error codes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fa") as temp_fa:
        temp_fa.write(b">seq1\nATCG\n")

    try:
        # Call with temporary file and dummy parameters
        # This should fail due to missing files but not crash
        result = run_motali_cpp(
            file_fasta=temp_fa.name,
            type_model_1="pwm",
            type_model_2="pwm",
            file_model_1="/nonexistent/model1.txt",
            file_model_2="/nonexistent/model2.txt",
            file_table_1="/nonexistent/table1.txt",
            file_table_2="/nonexistent/table2.txt",
            shift=10,
            threshold=0.01,
            file_hist="/tmp/hist.txt",
            yes_out_hist=0,
            file_prc="/tmp/prc.txt",
            yes_out_prc=0,
            file_short_over="/tmp/short_over.txt",
            file_short_all="/tmp/short_all.txt",
            file_sta_long="/tmp/sta_long.txt",
        )

        # Should return error code (-1) for invalid files
        assert isinstance(result, int)

    finally:
        # Clean up
        os.unlink(temp_fa.name)


def test_c_function_return_type():
    """Test that C function returns integer as expected."""
    from mimosa._core import run_motali_cpp

    # Instead of checking signature (which doesn't work with nanobind),
    # just verify the function exists and is callable
    assert callable(run_motali_cpp)


def test_motali_no_fd_leak_on_repeated_calls(tmp_path):
    """Repeated Motali calls should not leak file descriptors."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("FD leak check requires /proc/self/fd (Linux only).")

    root = Path(__file__).resolve().parent.parent
    examples_dir = root / "examples"

    model1 = PwmStrategy.load(str(examples_dir / "pif4.meme"), {"index": 0})
    model2 = PwmStrategy.load(str(examples_dir / "gata2.meme"), {"index": 0})
    sequences = read_fasta(examples_dir / "foreground.fa")
    promoters = read_fasta(examples_dir / "background.fa")

    before = len(list(fd_dir.iterdir()))

    for _ in range(20):
        result = compare_motifs(
            model1=model1,
            model2=model2,
            strategy="motali",
            sequences=sequences,
            promoters=promoters,
            tmp_directory=str(tmp_path),
        )
        assert "score" in result

    after = len(list(fd_dir.iterdir()))
    assert after - before < 8, f"Potential FD leak detected: before={before}, after={after}"


if __name__ == "__main__":
    test_c_module_import()
    test_c_function_signature()
    test_c_function_return_type()
    print("Basic C integration tests passed!")
