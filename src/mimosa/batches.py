"""Dense masked batch helpers for sequences and score profiles."""

from __future__ import annotations

from typing import Iterable

import numpy as np

NUCLEOTIDE_PADDING = np.int8(4)
SCORE_PADDING = np.float32(4.0)
BATCH_NDIM = 2


def empty_batch(dtype, padding_value):
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


def pack_batch(values, mask, lengths, padding_value):
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


def make_batch(rows: Iterable[np.ndarray], dtype=None, padding_value=0):
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


def make_sequence_batch(rows: Iterable[np.ndarray]):
    """Build a dense masked batch for integer-encoded sequences."""
    return make_batch(rows, dtype=np.int8, padding_value=NUCLEOTIDE_PADDING)


def make_score_batch(rows: Iterable[np.ndarray]):
    """Build a dense masked batch for score profiles."""
    return make_batch(rows, dtype=np.float32, padding_value=SCORE_PADDING)


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


def flatten_valid(batch) -> np.ndarray:
    """Return all valid elements as one flat 1D array."""
    if batch["values"].size == 0:
        return np.empty(0, dtype=batch["values"].dtype)
    return batch["values"][batch["mask"]]


def batch_with_values(batch, values, padding_value=None):
    """Return a copy of one batch with the same mask/lengths and different values."""
    resolved_padding = batch["padding_value"] if padding_value is None else padding_value
    return pack_batch(values, batch["mask"], batch["lengths"], resolved_padding)
