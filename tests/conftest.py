"""
Pytest configuration and common fixtures for unimotifcomparator tests.
"""
import os
import tempfile
from pathlib import Path

import pytest


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