"""Utilities for lazy on-disk caching of derived profile signals."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TypedDict

import numpy as np

from mimosa.batches import SCORE_PADDING, ProfileBundle, pack_profile_bundle
from mimosa.models import GenericModel

CACHE_VERSION = "v8"


class ProfileCacheSpec(TypedDict):
    model: GenericModel
    sequences: dict | None
    promoters: dict | None
    profile_kind: str
    cache_dir: str


def _hash_array(array: np.ndarray) -> bytes:
    """Hash one array shape, dtype, and raw bytes deterministically."""
    contiguous = np.ascontiguousarray(array)
    shape = np.asarray(contiguous.shape, dtype=np.int64)

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(shape.tobytes())
    hasher.update(contiguous.view(np.uint8))
    return hasher.digest()


def fingerprint_batch(batch) -> str | None:
    """Return a stable content fingerprint for one padded or masked batch."""
    if batch is None:
        return None

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(_hash_array(np.asarray(batch["values"])))
    mask = batch.get("mask")
    if mask is not None:
        hasher.update(_hash_array(np.asarray(mask, dtype=np.uint8)))
    hasher.update(_hash_array(np.asarray(batch["lengths"], dtype=np.int64)))
    return hasher.hexdigest()


def fingerprint_profile_bundle(bundle) -> str | None:
    """Return a stable content fingerprint for one 3D profile bundle."""
    if bundle is None:
        return None

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(_hash_array(np.asarray(bundle["values"])))
    hasher.update(_hash_array(np.asarray(bundle["lengths"], dtype=np.int64)))
    return hasher.hexdigest()


def fingerprint_model(model: GenericModel) -> str:
    """Return a stable fingerprint for the effective model representation."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(model.type_key.encode("ascii"))
    hasher.update(model.name.encode("utf-8"))
    hasher.update(str(model.length).encode("ascii"))
    hasher.update(str(model.config.get("kmer")).encode("ascii"))

    if model.type_key == "scores":
        batch_fp = fingerprint_batch(model.config["scores_data"])
        if batch_fp is not None:
            hasher.update(batch_fp.encode("ascii"))
        return hasher.hexdigest()

    hasher.update(_hash_array(np.asarray(model.representation)))
    return hasher.hexdigest()


def _profile_cache_path(spec: ProfileCacheSpec) -> Path:
    """Build the file path for one cached profile artifact."""
    model_fp = fingerprint_model(spec["model"])
    sequence_fp = fingerprint_batch(spec.get("sequences")) or "no-sequences"
    base = Path(spec["cache_dir"]) / CACHE_VERSION / "profiles" / spec["profile_kind"] / sequence_fp
    promoter_fp = fingerprint_batch(spec.get("promoters"))
    if promoter_fp is not None:
        base = base / promoter_fp
    return base / f"{model_fp}.npz"


def load_profile_cache(spec: ProfileCacheSpec) -> ProfileBundle | None:
    """Load one cached normalized profile bundle if it is present and readable."""
    path = _profile_cache_path(spec)
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["version"])
            if version != CACHE_VERSION:
                return None
            values = payload["values"].astype(np.float32, copy=False)
            lengths = payload["lengths"].astype(np.int64, copy=False)
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None

    return pack_profile_bundle(values, lengths, SCORE_PADDING)


def store_profile_cache(spec: ProfileCacheSpec, profile: ProfileBundle) -> Path:
    """Store one derived profile bundle atomically on disk."""
    path = _profile_cache_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="profile-", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(
            tmp_path,
            version=np.array(CACHE_VERSION),
            values=np.asarray(profile["values"], dtype=np.float32),
            lengths=np.asarray(profile["lengths"], dtype=np.int64),
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return path


def clear_cache(cache_dir: str) -> int:
    """Remove all cached artifacts from the configured cache directory."""
    root = Path(cache_dir)
    if not root.exists():
        return 0

    removed = sum(1 for _ in root.rglob("*"))
    shutil.rmtree(root)
    return removed
