"""Dense masked batch helpers for sequences and score profiles."""

from __future__ import annotations

from typing import Iterable, TypedDict

import numpy as np

NUCLEOTIDE_PADDING = np.int8(4)
SCORE_PADDING = np.float32(0.0)
BATCH_NDIM = 2
PROFILE_BUNDLE_NDIM = 3
PLUS_STRAND = 0
MINUS_STRAND = 1
STRAND_COUNT = 2


class DenseBatch(TypedDict):
    values: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    padding_value: int | float


class ProfileBundle(TypedDict):
    values: np.ndarray
    lengths: np.ndarray
    padding_value: float


def empty_batch(dtype, padding_value) -> DenseBatch:
    """Return an empty dense batch."""
    values = np.full((0, 0), padding_value, dtype=dtype)
    mask = np.zeros((0, 0), dtype=bool)
    lengths = np.zeros(0, dtype=np.int64)
    return {
        "values": values,
        "mask": mask,
        "lengths": lengths,
        "padding_value": np.asarray(padding_value, dtype=dtype).item(),
    }


def empty_profile_bundle(dtype, padding_value, n_profiles: int = STRAND_COUNT) -> ProfileBundle:
    """Return an empty 3D profile bundle."""
    values = np.full((n_profiles, 0, 0), padding_value, dtype=dtype)
    lengths = np.zeros(0, dtype=np.int64)
    return {
        "values": values,
        "lengths": lengths,
        "padding_value": np.asarray(padding_value, dtype=dtype).item(),
    }


def pack_batch(values, mask, lengths, padding_value) -> DenseBatch:
    """Normalize one dense batch payload."""
    values_array = np.asarray(values)
    mask_array = np.asarray(mask, dtype=bool)
    lengths_array = np.asarray(lengths, dtype=np.int64)

    if values_array.ndim != BATCH_NDIM:
        raise ValueError("batch values must be a 2D array")
    if mask_array.shape != values_array.shape:
        raise ValueError("batch mask must have the same shape as values")
    if lengths_array.shape != (values_array.shape[0],):
        raise ValueError("batch lengths must have shape (n_rows,)")

    packed = np.full(values_array.shape, padding_value, dtype=values_array.dtype)
    packed[mask_array] = values_array[mask_array]

    return {
        "values": packed,
        "mask": mask_array,
        "lengths": lengths_array,
        "padding_value": np.asarray(padding_value, dtype=values_array.dtype).item(),
    }


def pack_profile_bundle(values, lengths, padding_value) -> ProfileBundle:
    """Normalize one 3D profile bundle payload."""
    values_array = np.asarray(values)
    lengths_array = np.asarray(lengths, dtype=np.int64)

    if values_array.ndim != PROFILE_BUNDLE_NDIM:
        raise ValueError("profile bundle values must be a 3D array")
    if lengths_array.shape != (values_array.shape[1],):
        raise ValueError("profile bundle lengths must have shape (n_rows,)")

    packed = np.full(values_array.shape, padding_value, dtype=values_array.dtype)
    width = values_array.shape[2]
    for row_index, row_length in enumerate(lengths_array):
        current_length = int(row_length)
        if current_length < 0 or current_length > width:
            raise ValueError("profile bundle row lengths must fit within the padded width")
        if current_length == 0:
            continue
        packed[:, row_index, :current_length] = values_array[:, row_index, :current_length]

    return {
        "values": packed,
        "lengths": lengths_array,
        "padding_value": np.asarray(padding_value, dtype=values_array.dtype).item(),
    }


def make_batch(rows: Iterable[np.ndarray], dtype=None, padding_value=0) -> DenseBatch:
    """Build a dense masked batch from one iterable of 1D arrays."""
    row_list = [np.asarray(row, dtype=dtype).ravel() if dtype is not None else np.asarray(row).ravel() for row in rows]
    if not row_list:
        resolved_dtype = dtype if dtype is not None else np.float32
        return empty_batch(resolved_dtype, padding_value)

    resolved_dtype = dtype if dtype is not None else row_list[0].dtype
    normalized_rows = [np.asarray(row, dtype=resolved_dtype) for row in row_list]
    lengths = np.asarray([row.size for row in normalized_rows], dtype=np.int64)
    width = int(lengths.max(initial=0))
    values = np.full((len(normalized_rows), width), padding_value, dtype=resolved_dtype)
    mask = np.zeros((len(normalized_rows), width), dtype=bool)

    for row_index, row in enumerate(normalized_rows):
        length = int(lengths[row_index])
        if length == 0:
            continue
        values[row_index, :length] = row
        mask[row_index, :length] = True

    return pack_batch(values, mask, lengths, padding_value)


def make_sequence_batch(rows: Iterable[np.ndarray]) -> DenseBatch:
    """Build a dense masked batch for integer-encoded sequences."""
    return make_batch(rows, dtype=np.int8, padding_value=NUCLEOTIDE_PADDING)


def make_score_batch(rows: Iterable[np.ndarray]) -> DenseBatch:
    """Build a dense masked batch for score profiles."""
    return make_batch(rows, dtype=np.float32, padding_value=SCORE_PADDING)


def make_strand_bundle(plus_batch: DenseBatch, minus_batch: DenseBatch) -> ProfileBundle:
    """Build one two-strand profile bundle from plus/minus dense batches."""
    plus_values = np.asarray(plus_batch["values"])
    minus_values = np.asarray(minus_batch["values"])
    plus_lengths = np.asarray(plus_batch["lengths"], dtype=np.int64)
    minus_lengths = np.asarray(minus_batch["lengths"], dtype=np.int64)

    if plus_values.shape != minus_values.shape:
        raise ValueError("plus and minus batches must have identical shapes")
    if not np.array_equal(plus_lengths, minus_lengths):
        raise ValueError("plus and minus batches must have identical row lengths")

    return pack_profile_bundle(
        np.stack((plus_values, minus_values), axis=0),
        plus_lengths,
        plus_batch["padding_value"],
    )


def num_rows(batch) -> int:
    """Return the number of rows in one batch."""
    return int(batch["values"].shape[0])


def batch_width(batch) -> int:
    """Return the padded row width."""
    return int(batch["values"].shape[1])


def row_values(batch, row_index: int) -> np.ndarray:
    """Return the valid values of one row."""
    length = int(batch["lengths"][row_index])
    return batch["values"][row_index, :length]


def profile_values(bundle, profile_index: int) -> np.ndarray:
    """Return one 2D profile matrix view from a 3D bundle."""
    return bundle["values"][profile_index]


def profile_view(bundle: ProfileBundle, profile_index: int) -> DenseBatch:
    """Return one lightweight 2D profile view from a 3D bundle."""
    return {
        "values": bundle["values"][profile_index],
        "lengths": bundle["lengths"],
        "padding_value": bundle["padding_value"],
    }


def profile_row_values(bundle, profile_index: int, row_index: int) -> np.ndarray:
    """Return the valid values of one row for one profile inside a bundle."""
    length = int(bundle["lengths"][row_index])
    return bundle["values"][profile_index, row_index, :length]


def flatten_valid(batch) -> np.ndarray:
    """Return all valid elements as one flat 1D array."""
    if batch["values"].size == 0:
        return np.empty(0, dtype=batch["values"].dtype)
    return batch["values"][batch["mask"]]


def flatten_profile_bundle(bundle: ProfileBundle, profile_index: int | None = None) -> np.ndarray:
    """Return valid bundle elements as one flat 1D array."""
    values = np.asarray(bundle["values"])
    lengths = np.asarray(bundle["lengths"], dtype=np.int64)

    if values.size == 0:
        return np.empty(0, dtype=values.dtype)

    width = values.shape[2]
    row_mask = np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]
    if not np.any(row_mask):
        return np.empty(0, dtype=values.dtype)

    if profile_index is not None:
        return values[profile_index][row_mask]
    return values[:, row_mask].reshape(-1)


def batch_with_values(batch: DenseBatch, values, padding_value=None) -> DenseBatch:
    """Return a copy of one batch with the same mask/lengths and different values."""
    resolved_padding = batch["padding_value"] if padding_value is None else padding_value
    return pack_batch(values, batch["mask"], batch["lengths"], resolved_padding)
