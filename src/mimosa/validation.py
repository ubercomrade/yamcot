"""Shared validation helpers for CLI and library entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

SITE_SELECTION_MODES = {"best", "threshold"}
CACHE_MODES = {"off", "on"}


def validate_file_exists(path: str | Path, label: str) -> Path:
    """Return an existing path or raise a descriptive error."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return resolved


def validate_positive_int(name: str, value: int) -> int:
    """Validate one positive integer parameter."""
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return number


def validate_non_negative(name: str, value: Optional[float]) -> Optional[float]:
    """Validate one optional non-negative numeric parameter."""
    if value is None:
        return None

    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def validate_kernel_size_range(min_kernel_size: int, max_kernel_size: int) -> tuple[int, int]:
    """Validate surrogate-kernel bounds and require at least one odd width."""
    minimum = validate_positive_int("min_kernel_size", min_kernel_size)
    maximum = validate_positive_int("max_kernel_size", max_kernel_size)
    if minimum > maximum:
        raise ValueError(f"min_kernel_size ({minimum}) must be less than or equal to max_kernel_size ({maximum}).")

    first_odd = minimum if minimum % 2 == 1 else minimum + 1
    if first_odd > maximum:
        raise ValueError(
            "Kernel size range must include at least one odd value, "
            "because surrogate convolution uses centered kernels."
        )

    return minimum, maximum


def validate_pfm_top_fraction(value: Optional[float]) -> Optional[float]:
    """Validate the optional fraction of hits used for PFM reconstruction."""
    if value is None:
        return None

    fraction = float(value)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("pfm_top_fraction must be in the (0, 1] range.")
    return fraction


def validate_cache_mode(value: str) -> str:
    """Normalize and validate cache mode."""
    normalized = str(value).lower()
    if normalized not in CACHE_MODES:
        raise ValueError("cache_mode must be either 'off' or 'on'.")
    return normalized


def validate_profile_normalization(name: str, available_names: Iterable[str]) -> str:
    """Normalize and validate the selected profile normalization strategy."""
    normalized = str(name).lower()
    available = tuple(sorted(available_names))
    if normalized not in available:
        raise ValueError(f"profile_normalization must be one of: {', '.join(available)}.")
    return normalized


def validate_site_mode(mode: str, fpr_threshold: Optional[float]) -> str:
    """Validate the site/PFM selection mode and dependent parameters."""
    normalized = str(mode).lower()
    if normalized not in SITE_SELECTION_MODES:
        raise ValueError(f"mode must be 'best' or 'threshold', got {mode}")
    if normalized == "threshold" and fpr_threshold is None:
        raise ValueError("fpr_threshold is required for mode='threshold'")
    return normalized
