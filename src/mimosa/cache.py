"""Utilities for lazy on-disk caching of derived profile signals."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from mimosa.matrix import MatrixData
from mimosa.models import GenericModel

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


def fingerprint_matrix_data(matrix_data: Optional[MatrixData]) -> Optional[str]:
    """Return a stable content fingerprint for matrix-backed variable-length data."""
    if matrix_data is None:
        return None

    meta = (
        id(matrix_data.matrix),
        id(matrix_data.lengths),
        matrix_data.matrix.shape,
        matrix_data.lengths.shape,
        matrix_data.matrix.dtype.str,
        matrix_data.lengths.dtype.str,
    )
    cached_meta = getattr(matrix_data, "_cache_fingerprint_meta", None)
    cached_value = getattr(matrix_data, "_cache_fingerprint", None)
    if cached_meta == meta and cached_value is not None:
        return cached_value

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(_hash_array(matrix_data.matrix))
    hasher.update(_hash_array(matrix_data.lengths))
    fingerprint = hasher.hexdigest()

    matrix_data._cache_fingerprint_meta = meta
    matrix_data._cache_fingerprint = fingerprint
    return fingerprint


def fingerprint_model(model: GenericModel) -> str:
    """Return a stable fingerprint for the effective model representation."""
    if model.type_key == "scores":
        scores_data = model.config["scores_data"]
        meta = ("scores", id(scores_data.matrix), id(scores_data.lengths), model.name, model.length)
        cached_meta = model.config.get("_cache_model_fingerprint_meta")
        cached_value = model.config.get("_cache_model_fingerprint")
        if cached_meta == meta and cached_value is not None:
            return cached_value

        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(model.type_key.encode("ascii"))
        hasher.update(model.name.encode("utf-8"))
        hasher.update(str(model.length).encode("ascii"))
        matrix_fp = fingerprint_matrix_data(scores_data)
        if matrix_fp is not None:
            hasher.update(matrix_fp.encode("ascii"))
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
    sequences: Optional[MatrixData],
    promoters: Optional[MatrixData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
) -> Path:
    """Build the file path for a cached profile artifact."""
    model_fp = fingerprint_model(model)
    seq_fp = fingerprint_matrix_data(sequences) or "no-sequences"
    base = Path(cache_dir) / CACHE_VERSION / "profiles" / profile_kind / seq_fp
    prom_fp = fingerprint_matrix_data(promoters)
    if prom_fp is not None:
        base = base / prom_fp
    return base / f"{model_fp}.{strand}.npz"


def load_profile_cache(
    model: GenericModel,
    sequences: Optional[MatrixData],
    promoters: Optional[MatrixData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
) -> Optional[MatrixData]:
    """Load a cached profile if it is present and readable."""
    path = _profile_cache_path(model, sequences, promoters, strand, profile_kind, cache_dir)
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["version"])
            if version != CACHE_VERSION:
                return None
            matrix = payload["matrix"].astype(np.float32, copy=False)
            lengths = payload["lengths"].astype(np.int64, copy=False)
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None

    return MatrixData(matrix=matrix, lengths=lengths, pad_value=np.float32(0.0))


def store_profile_cache(
    model: GenericModel,
    sequences: Optional[MatrixData],
    promoters: Optional[MatrixData],
    strand: str,
    profile_kind: str,
    cache_dir: str,
    profile: MatrixData,
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
                matrix=profile.matrix.astype(np.float32, copy=False),
                lengths=profile.lengths.astype(np.int64, copy=False),
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
