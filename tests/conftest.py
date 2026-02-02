"""
Pytest configuration and common fixtures for unimotifcomparator tests.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Force testing the installed package, not the local source
# This is necessary for cibuildwheel to test the actual built wheel
project_root = str(Path(__file__).parent.parent.absolute())
if project_root in sys.path:
    sys.path.remove(project_root)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def examples_dir():
    """Return path to examples directory."""
    return Path(__file__).parent.parent / "examples"