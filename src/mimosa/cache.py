"""Utilities for lazy on-disk caching of derived profile signals."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from mimosa.models import GenericModel
from mimosa.ragged import RaggedData

CACHE_VERSION = "v2"


def _hash_array(array: np.ndarray) -> bytes:
    """Hash array shape, dtype, and raw bytes deterministically."""
    contiguous = np.ascontiguousarray(array)
    shape = np.asarray(contiguous.shape, dtype=np.int64)

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(shape.tobytes())
    hasher.update(contiguous.view(np.uint8))
    return hasher.digest()


def fingerprint_ragged(ragged: Optional[RaggedData]) -> Optional[str]:
    """Return a stable content fingerprint for a RaggedData object."""
    if ragged is None:
        return None

    meta = (
        id(ragged.data),
        id(ragged.offsets),
        ragged.data.shape,
        ragged.offsets.shape,
        ragged.data.dtype.str,
        ragged.offsets.dtype.str,
    )
    cached_meta = getattr(ragged, "_cache_fingerprint_meta", None)
    cached_value = getattr(ragged, "_cache_fingerprint", None)
    if cached_meta == meta and cached_value is not None:
        return cached_value

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(_hash_array(ragged.data))
    hasher.update(_hash_array(ragged.offsets))
    fingerprint = hasher.hexdigest()

    ragged._cache_fingerprint_meta = meta
    ragged._cache_fingerprint = fingerprint
    return fingerprint


def fingerprint_model(model: GenericModel) -> str:
    """Return a stable fingerprint for the effective model representation."""
    if model.type_key == "scores":
        scores_data = model.config["scores_data"]
        meta = ("scores", id(scores_data.data), id(scores_data.offsets), model.name, model.length)
        cached_meta = model.config.get("_cache_model_fingerprint_meta")
        cached_value = model.config.get("_cache_model_fingerprint")
        if cached_meta == meta and cached_value is not None:
            return cached_value

        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(model.type_key.encode("ascii"))
        hasher.update(model.name.encode("utf-8"))
        hasher.update(str(model.length).encode("ascii"))
        ragged_fp = fingerprint_ragged(scores_data)
        if ragged_fp is not None:
            hasher.update(ragged_fp.encode("ascii"))
        fingerprint = hasher.hexdigest()

        model.config["_cache_model_fingerprint_meta"] = meta
        model.config["_cache_model_fingerprint"] = fingerprint
        return fingerprint

    representation = np.asarray(model.representation)
    meta = (
        id(representation),
        representation.shape,
        representation.dtype.str,
        model.type_key,
        model.name,
        model.length,
        model.config.get("kmer"),
    )
    cached_meta = model.config.get("_cache_model_fingerprint_meta")
    cached_value = model.config.get("_cache_model_fingerprint")
    if cached_meta == meta and cached_value is not None:
        return cached_value

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(model.type_key.encode("ascii"))
    hasher.update(model.name.encode("utf-8"))
    hasher.update(str(model.length).encode("ascii"))
    hasher.update(str(model.config.get("kmer")).encode("ascii"))
    hasher.update(_hash_array(representation))
    fingerprint = hasher.hexdigest()

    model.config["_cache_model_fingerprint_meta"] = meta
    model.config["_cache_model_fingerprint"] = fingerprint
    return fingerprint


def _profile_cache_path(
    model: GenericModel,
    sequences: Optional[RaggedData],
    promoters: Optional[RaggedData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
) -> Path:
    """Build the file path for a cached profile artifact."""
    model_fp = fingerprint_model(model)
    seq_fp = fingerprint_ragged(sequences) or "no-sequences"
    base = Path(cache_dir) / CACHE_VERSION / "profiles" / profile_kind / seq_fp
    prom_fp = fingerprint_ragged(promoters)
    if prom_fp is not None:
        base = base / prom_fp
    return base / f"{model_fp}.{strand}.npz"


def load_profile_cache(
    model: GenericModel,
    sequences: Optional[RaggedData],
    promoters: Optional[RaggedData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
) -> Optional[RaggedData]:
    """Load a cached profile if it is present and readable."""
    path = _profile_cache_path(model, sequences, promoters, strand, profile_kind, cache_dir)
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["version"])
            if version != CACHE_VERSION:
                return None
            data = payload["data"].astype(np.float32, copy=False)
            offsets = payload["offsets"].astype(np.int64, copy=False)
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None

    return RaggedData(data=data, offsets=offsets)


def store_profile_cache(
    model: GenericModel,
    sequences: Optional[RaggedData],
    promoters: Optional[RaggedData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
    profile: RaggedData,
) -> Path:
    """Store a derived profile atomically on disk."""
    path = _profile_cache_path(model, sequences, promoters, strand, profile_kind, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="profile-", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(
                handle,
                version=np.array(CACHE_VERSION),
                data=profile.data.astype(np.float32, copy=False),
                offsets=profile.offsets.astype(np.int64, copy=False),
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
