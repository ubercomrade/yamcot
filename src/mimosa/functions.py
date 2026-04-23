"""Numerical helpers and Numba-backed scoring kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

from mimosa.batches import (
    SCORE_PADDING,
    batch_with_values,
    flatten_profile_bundle,
    flatten_valid,
    pack_batch,
    pack_profile_bundle,
)

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
PROFILE_EPS = np.float32(1e-6)
SCAN_BUCKET_STEP = 32
WINDOW_MATRIX_NDIM = 2


def pfm_to_pwm(pfm):
    """Convert Position Frequency Matrix to Position Weight Matrix."""
    return np.log((pfm + 0.0001) / 0.25)


def pcm_to_pfm(pcm, pseudocount: float = 0.25):
    """Convert Position Count Matrix to Position Frequency Matrix."""
    number_of_sites = pcm.sum(axis=0)
    nuc_pseudo = float(pseudocount)
    return (pcm + nuc_pseudo) / (number_of_sites + 4.0 * nuc_pseudo)


def build_score_log_tail_table(scores: np.ndarray) -> np.ndarray:
    """Build a score-to-log-tail lookup table from one score sample."""
    flat = np.asarray(scores, dtype=np.float32).ravel()
    if flat.size == 0:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    scores_sorted = np.sort(flat)[::-1]
    unique_scores, counts = np.unique(scores_sorted, return_counts=True)
    unique_scores = unique_scores[::-1]
    counts = counts[::-1]

    cum_counts = np.cumsum(counts)
    tail_probabilities = cum_counts / flat.size
    log_tail = -np.log10(tail_probabilities)
    return np.column_stack([unique_scores, log_tail]).astype(np.float32, copy=False)


@njit(cache=True, nogil=False)
def _lower_bound_desc(values, target):
    """Find the first descending-table index whose score is not greater than target."""
    size = values.shape[0]
    if size <= 1 or target >= values[0]:
        return 0
    if target <= values[size - 1]:
        return size - 1

    lo = 0
    hi = size
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if values[mid] > target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=False)
def _apply_score_log_tail_table_numba(values, mask, scores_col, log_tail_col, padding_value: float):
    """Map one dense masked score matrix to empirical log-tail values."""
    rows, cols = values.shape
    mapped = np.empty((rows, cols), dtype=np.float32)

    for row_index in range(rows):
        for col_index in range(cols):
            if mask[row_index, col_index]:
                idx = _lower_bound_desc(scores_col, values[row_index, col_index])
                mapped[row_index, col_index] = log_tail_col[idx]
            else:
                mapped[row_index, col_index] = padding_value

    return mapped


def apply_score_log_tail_table(score_batch, table: np.ndarray):
    """Map one score batch to empirical log-tail values using a lookup table."""
    table_arr = np.asarray(table, dtype=np.float32)
    if table_arr.size == 0:
        empty_values = np.full_like(score_batch["values"], SCORE_PADDING)
        return batch_with_values(score_batch, empty_values, padding_value=SCORE_PADDING)

    mapped = _apply_score_log_tail_table_numba(
        np.ascontiguousarray(score_batch["values"], dtype=np.float32),
        np.ascontiguousarray(score_batch["mask"], dtype=np.bool_),
        np.ascontiguousarray(table_arr[:, 0], dtype=np.float32),
        np.ascontiguousarray(table_arr[:, 1], dtype=np.float32),
        np.float32(SCORE_PADDING),
    )
    return batch_with_values(score_batch, mapped, padding_value=SCORE_PADDING)


def _build_length_mask(lengths: np.ndarray, width: int) -> np.ndarray:
    """Build one dense prefix mask from row lengths."""
    return np.arange(width, dtype=np.int64)[None, :] < np.asarray(lengths, dtype=np.int64)[:, None]


def apply_score_log_tail_table_to_profile_bundle(profile_bundle, table: np.ndarray):
    """Map one 3D profile bundle to empirical log-tail values using one lookup table."""
    table_arr = np.asarray(table, dtype=np.float32)
    values = np.ascontiguousarray(profile_bundle["values"], dtype=np.float32)
    lengths = np.asarray(profile_bundle["lengths"], dtype=np.int64)

    if table_arr.size == 0:
        empty_values = np.full_like(values, SCORE_PADDING)
        return pack_profile_bundle(empty_values, lengths, SCORE_PADDING)

    mask = np.ascontiguousarray(_build_length_mask(lengths, values.shape[2]), dtype=np.bool_)
    mapped = np.empty_like(values)
    scores_col = np.ascontiguousarray(table_arr[:, 0], dtype=np.float32)
    log_tail_col = np.ascontiguousarray(table_arr[:, 1], dtype=np.float32)

    for profile_index in range(values.shape[0]):
        mapped[profile_index] = _apply_score_log_tail_table_numba(
            values[profile_index],
            mask,
            scores_col,
            log_tail_col,
            np.float32(SCORE_PADDING),
        )

    return pack_profile_bundle(mapped, lengths, SCORE_PADDING)


def scores_to_empirical_log_tail_bundle(profile_bundle):
    """Convert one 3D profile bundle to empirical log-tail values within the current sample."""
    table = build_score_log_tail_table(flatten_profile_bundle(profile_bundle))
    return apply_score_log_tail_table_to_profile_bundle(profile_bundle, table)


def normalize_empirical_log_tail_pair(score_batch_plus, score_batch_minus):
    """Normalize two strand score batches using one shared empirical log-tail mapping."""
    profile_bundle = pack_profile_bundle(
        np.stack(
            (
                np.asarray(score_batch_plus["values"], dtype=np.float32),
                np.asarray(score_batch_minus["values"], dtype=np.float32),
            ),
            axis=0,
        ),
        np.asarray(score_batch_plus["lengths"], dtype=np.int64),
        SCORE_PADDING,
    )
    normalized = scores_to_empirical_log_tail_bundle(profile_bundle)
    values_plus = normalized["values"][0]
    values_minus = normalized["values"][1]
    return (
        pack_batch(values_plus, score_batch_plus["mask"], score_batch_plus["lengths"], SCORE_PADDING),
        pack_batch(values_minus, score_batch_minus["mask"], score_batch_minus["lengths"], SCORE_PADDING),
    )


def lookup_score_for_tail_probability(table: np.ndarray, tail_probability: float) -> float:
    """Convert a tail probability threshold to the corresponding score cutoff."""
    if tail_probability <= 0:
        return float(table[0, 0])

    target_log_tail = -np.log10(tail_probability)
    scores_col = table[:, 0]
    log_tail_col = table[:, 1]
    mask = log_tail_col >= target_log_tail

    if not np.any(mask):
        return float(scores_col[-1])

    last_valid = np.where(mask)[0][-1]
    return float(scores_col[last_valid])


def score_seq(num_site, kmer, model):
    """Compute the score for one encoded site."""
    site = np.asarray(num_site, dtype=np.int64)
    matrix = np.asarray(model, dtype=np.float32).reshape(-1, np.asarray(model).shape[-1])
    kmer = int(kmer)
    score = 0.0

    for position in range(site.shape[0] - kmer + 1):
        code = 0
        for offset in range(kmer):
            code = code * 5 + int(site[position + offset])
        score += float(matrix[code, position])

    return score


def _prepare_model_rows(matrix: np.ndarray) -> np.ndarray:
    """Return one motif tensor as a flat 5-ary row table."""
    arr = np.asarray(matrix, dtype=np.float32)
    return np.ascontiguousarray(arr.reshape((-1, arr.shape[-1])), dtype=np.float32)


def _resolve_scan_layout(kmer: int, motif_len: int, with_context: bool) -> tuple[int, int, int]:
    """Resolve the geometry used by the sequence-scanning kernels."""
    context_len = kmer - 1 if with_context else 0
    window_size = motif_len + context_len
    n_terms = window_size - kmer + 1
    return context_len, window_size, n_terms


def _prepare_scan_inputs(sequences, matrix: np.ndarray):
    """Normalize scan inputs to contiguous arrays and derived geometry."""
    values = np.ascontiguousarray(sequences["values"], dtype=np.int8)
    lengths = np.ascontiguousarray(sequences["lengths"], dtype=np.int64)
    model_rows = _prepare_model_rows(matrix)
    motif_len = int(model_rows.shape[-1])
    out_lengths = np.maximum(lengths - motif_len + 1, 0)
    max_scores = int(out_lengths.max(initial=0))
    return values, lengths, model_rows, motif_len, max_scores, out_lengths


def _iter_scan_buckets(lengths: np.ndarray, motif_len: int, bucket_step: int = SCAN_BUCKET_STEP):
    """Yield row-index buckets with similar output lengths."""
    out_lengths = np.maximum(lengths - motif_len + 1, 0)
    positive_indices = np.flatnonzero(out_lengths > 0)
    if positive_indices.size == 0:
        return

    bucket_ids = (out_lengths[positive_indices] - 1) // max(int(bucket_step), 1)
    order = np.argsort(bucket_ids, kind="mergesort")
    sorted_indices = positive_indices[order]
    sorted_bucket_ids = bucket_ids[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bucket_ids)) + 1]
    stops = np.r_[starts[1:], sorted_indices.size]
    for start, stop in zip(starts, stops, strict=False):
        yield sorted_indices[start:stop]


@njit(cache=True, fastmath=True, nogil=False)
def _score_window_forward(seq_row, length: int, model_rows, pos: int, kmer: int, context_len: int, n_terms: int):
    """Score one forward-aligned window."""
    total = np.float32(0.0)
    for term in range(n_terms):
        code = 0
        src_start = pos - context_len + term
        for offset in range(kmer):
            src = src_start + offset
            encoded = 4
            if 0 <= src < length:
                encoded = int(seq_row[src])
            code = code * 5 + encoded
        total += model_rows[code, term]
    return total


@njit(cache=True, fastmath=True, nogil=False)
def _score_window_reverse(seq_row, length: int, model_rows, pos: int, kmer: int, window_size: int, n_terms: int):
    """Score one reverse-complement-aligned window."""
    total = np.float32(0.0)
    for term in range(n_terms):
        code = 0
        for offset in range(kmer):
            src = pos + (window_size - 1 - (term + offset))
            encoded = 4
            if 0 <= src < length:
                encoded = int(RC_TABLE[int(seq_row[src])])
            code = code * 5 + encoded
        total += model_rows[code, term]
    return total


@njit(cache=True, fastmath=True, nogil=False)
def _scan_dense_kernel_numba(values, lengths, model_rows, kmer: int, context_len: int, n_terms: int):
    """Score one dense encoded sequence batch for one strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_forward(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                context_len,
                n_terms,
            )
            mask[row_index, pos] = True

    return scores, mask


@njit(cache=True, fastmath=True, nogil=False)
def _scan_dense_reverse_kernel_numba(values, lengths, model_rows, kmer: int, window_size: int, n_terms: int):
    """Score one dense encoded sequence batch on the reverse-complement strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_reverse(seq_row, length, model_rows, pos, kmer, window_size, n_terms)
            mask[row_index, pos] = True

    return scores, mask


@njit(cache=True, fastmath=True, nogil=False)
def _scan_dense_strands_kernel_numba(
    values, lengths, model_rows, kmer: int, context_len: int, window_size: int, n_terms: int
):
    """Score one dense encoded batch on both strands in one call."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, 2, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, 0, pos] = _score_window_forward(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                context_len,
                n_terms,
            )
            scores[row_index, 1, pos] = _score_window_reverse(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                window_size,
                n_terms,
            )
            mask[row_index, 0, pos] = True
            mask[row_index, 1, pos] = True

    return scores, mask


def _empty_score_scan_batch(n_rows: int, max_scores: int, out_lengths: np.ndarray):
    """Return one empty score batch with the requested output geometry."""
    empty_values = np.full((n_rows, max_scores), SCORE_PADDING, dtype=np.float32)
    empty_mask = np.zeros((n_rows, max_scores), dtype=bool)
    return pack_batch(empty_values, empty_mask, out_lengths, SCORE_PADDING)


def batch_all_scores(
    sequences, matrix: np.ndarray, kmer: int = 1, is_revcomp: bool = False, with_context: bool = False
):
    """Compute scores for all sequences in one dense masked batch."""
    values, lengths, model_rows, motif_len, max_scores, out_lengths = _prepare_scan_inputs(sequences, matrix)
    n_rows = int(values.shape[0])

    if n_rows == 0 or max_scores == 0:
        return _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    context_len, window_size, n_terms = _resolve_scan_layout(int(kmer), motif_len, bool(with_context))
    if int(lengths.max(initial=0)) == int(lengths.min(initial=0)):
        if is_revcomp:
            scored_values, scored_mask = _scan_dense_reverse_kernel_numba(
                values,
                lengths,
                model_rows,
                int(kmer),
                window_size,
                n_terms,
            )
        else:
            scored_values, scored_mask = _scan_dense_kernel_numba(
                values,
                lengths,
                model_rows,
                int(kmer),
                context_len,
                n_terms,
            )
    else:
        scored_values = np.full((n_rows, max_scores), SCORE_PADDING, dtype=np.float32)
        scored_mask = np.zeros((n_rows, max_scores), dtype=np.bool_)
        for bucket_indices in _iter_scan_buckets(lengths, motif_len):
            bucket_lengths = np.ascontiguousarray(lengths[bucket_indices], dtype=np.int64)
            bucket_width = int(bucket_lengths.max(initial=0))
            bucket_values = np.ascontiguousarray(values[bucket_indices, :bucket_width], dtype=np.int8)
            if is_revcomp:
                bucket_scores, bucket_mask = _scan_dense_reverse_kernel_numba(
                    bucket_values,
                    bucket_lengths,
                    model_rows,
                    int(kmer),
                    window_size,
                    n_terms,
                )
            else:
                bucket_scores, bucket_mask = _scan_dense_kernel_numba(
                    bucket_values,
                    bucket_lengths,
                    model_rows,
                    int(kmer),
                    context_len,
                    n_terms,
                )
            bucket_score_width = bucket_scores.shape[1]
            scored_values[bucket_indices, :bucket_score_width] = bucket_scores
            scored_mask[bucket_indices, :bucket_score_width] = bucket_mask

    return pack_batch(scored_values, scored_mask, out_lengths, SCORE_PADDING)


def batch_all_scores_strands(sequences, matrix: np.ndarray, kmer: int = 1, with_context: bool = False):
    """Compute scores for both strands in one dense masked batch call."""
    values, lengths, model_rows, motif_len, max_scores, out_lengths = _prepare_scan_inputs(sequences, matrix)
    n_rows = int(values.shape[0])

    if n_rows == 0 or max_scores == 0:
        empty_batch = _empty_score_scan_batch(n_rows, max_scores, out_lengths)
        return empty_batch, _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    context_len, window_size, n_terms = _resolve_scan_layout(int(kmer), motif_len, bool(with_context))
    if int(lengths.max(initial=0)) == int(lengths.min(initial=0)):
        scored_values, scored_mask = _scan_dense_strands_kernel_numba(
            values,
            lengths,
            model_rows,
            int(kmer),
            context_len,
            window_size,
            n_terms,
        )
    else:
        scored_values = np.full((n_rows, 2, max_scores), SCORE_PADDING, dtype=np.float32)
        scored_mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)
        for bucket_indices in _iter_scan_buckets(lengths, motif_len):
            bucket_lengths = np.ascontiguousarray(lengths[bucket_indices], dtype=np.int64)
            bucket_width = int(bucket_lengths.max(initial=0))
            bucket_values = np.ascontiguousarray(values[bucket_indices, :bucket_width], dtype=np.int8)
            bucket_scores, bucket_mask = _scan_dense_strands_kernel_numba(
                bucket_values,
                bucket_lengths,
                model_rows,
                int(kmer),
                context_len,
                window_size,
                n_terms,
            )
            bucket_score_width = bucket_scores.shape[2]
            scored_values[bucket_indices, :, :bucket_score_width] = bucket_scores
            scored_mask[bucket_indices, :, :bucket_score_width] = bucket_mask

    plus_batch = pack_batch(scored_values[:, 0, :], scored_mask[:, 0, :], out_lengths, SCORE_PADDING)
    minus_batch = pack_batch(scored_values[:, 1, :], scored_mask[:, 1, :], out_lengths, SCORE_PADDING)
    return plus_batch, minus_batch


def precision_recall_curve(classification, scores):
    """Compute the precision-recall curve."""
    classification = np.asarray(classification)
    scores = np.asarray(scores)

    if len(scores) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    precision = np.zeros(max_size)
    recall = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)
    precision[0] = 1.0
    recall[0] = 0.0
    uniq_scores[0] = np.inf

    true_positive = 0
    false_positive = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    true_false_ratio = 1.0 if number_of_false == 0 else number_of_true / number_of_false

    position = 1
    current_score = sorted_scores[0]

    for index, score in enumerate(sorted_scores):
        flag = sorted_classification[index]
        if flag == 1:
            true_positive += 1
        else:
            false_positive += 1

        if index == len(scores) - 1 or current_score != sorted_scores[index + 1]:
            uniq_scores[position] = score
            denominator = true_positive + true_false_ratio * false_positive
            precision[position] = true_positive / denominator if denominator > 0 else 1.0
            recall[position] = true_positive / number_of_true if number_of_true > 0 else 0.0
            position += 1
            if index < len(scores) - 1:
                current_score = sorted_scores[index + 1]

    return precision[:position], recall[:position], uniq_scores[:position]


def roc_curve(classification, scores):
    """Compute the ROC curve."""
    classification = np.asarray(classification)
    scores = np.asarray(scores)

    if len(scores) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    tpr = np.zeros(max_size)
    fpr = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)
    uniq_scores[0] = np.inf

    true_positive = 0
    false_positive = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    position = 1
    current_score = sorted_scores[0]

    for index, score in enumerate(sorted_scores):
        flag = sorted_classification[index]
        if flag == 1:
            true_positive += 1
        else:
            false_positive += 1

        if index == len(scores) - 1 or current_score != sorted_scores[index + 1]:
            uniq_scores[position] = score
            tpr[position] = true_positive / number_of_true if number_of_true > 0 else 0.0
            fpr[position] = false_positive / number_of_false if number_of_false > 0 else 0.0
            position += 1
            if index < len(scores) - 1:
                current_score = sorted_scores[index + 1]

    return tpr[:position], fpr[:position], uniq_scores[:position]


def cut_roc(tpr: np.ndarray, fpr: np.ndarray, thr: np.ndarray, score_cutoff: float):
    """Truncate ROC curve at a specific score threshold."""
    if score_cutoff == -np.inf:
        return tpr, fpr, thr

    mask = thr >= score_cutoff
    if not np.any(mask):
        return (
            np.array([tpr[0]], dtype=tpr.dtype),
            np.array([0.0], dtype=fpr.dtype),
            np.array([score_cutoff], dtype=thr.dtype),
        )

    last = int(np.where(mask)[0][-1])
    if thr[last] == score_cutoff or last == len(thr) - 1:
        return tpr[: last + 1], fpr[: last + 1], thr[: last + 1]

    s0, s1 = float(thr[last]), float(thr[last + 1])
    t0, t1 = float(tpr[last]), float(tpr[last + 1])
    f0, f1 = float(fpr[last]), float(fpr[last + 1])
    alpha = 0.0 if s0 == s1 else (score_cutoff - s0) / (s1 - s0)
    t_cut = t0 + alpha * (t1 - t0)
    f_cut = f0 + alpha * (f1 - f0)

    tpr_cut = np.concatenate([tpr[: last + 1], np.array([t_cut], dtype=tpr.dtype)])
    fpr_cut = np.concatenate([fpr[: last + 1], np.array([f_cut], dtype=fpr.dtype)])
    thr_cut = np.concatenate([thr[: last + 1], np.array([score_cutoff], dtype=thr.dtype)])
    return tpr_cut, fpr_cut, thr_cut


def cut_prc(rec: np.ndarray, prec: np.ndarray, thr: np.ndarray, score_cutoff: float):
    """Truncate Precision-Recall curve at a specific score threshold."""
    if score_cutoff == -np.inf:
        return rec, prec, thr

    mask = thr >= score_cutoff
    if not np.any(mask):
        return (
            np.array([0.0], dtype=rec.dtype),
            np.array([prec[0]], dtype=prec.dtype),
            np.array([score_cutoff], dtype=thr.dtype),
        )

    last = int(np.where(mask)[0][-1])
    if thr[last] == score_cutoff or last == len(thr) - 1:
        return rec[: last + 1], prec[: last + 1], thr[: last + 1]

    s0, s1 = float(thr[last]), float(thr[last + 1])
    r0, r1 = float(rec[last]), float(rec[last + 1])
    p0, p1 = float(prec[last]), float(prec[last + 1])
    alpha = 0.0 if s0 == s1 else (score_cutoff - s0) / (s1 - s0)
    r_cut = r0 + alpha * (r1 - r0)
    p_cut = p0 + alpha * (p1 - p0)

    rec_cut = np.concatenate([rec[: last + 1], np.array([r_cut], dtype=rec.dtype)])
    prec_cut = np.concatenate([prec[: last + 1], np.array([p_cut], dtype=prec.dtype)])
    thr_cut = np.concatenate([thr[: last + 1], np.array([score_cutoff], dtype=thr.dtype)])
    return rec_cut, prec_cut, thr_cut


def standardized_pauc(pauc_raw: float, pauc_min: float, pauc_max: float) -> float:
    """Standardize a partial AUC value to the range [0.5, 1.0]."""
    denom = pauc_max - pauc_min
    if denom <= 0:
        return 0.5
    return 0.5 * (1.0 + (pauc_raw - pauc_min) / denom)


def scores_to_empirical_log_tail(score_batch):
    """Convert one score batch to empirical log-tail values within the current sample."""
    table = build_score_log_tail_table(flatten_valid(score_batch))
    return apply_score_log_tail_table(score_batch, table)


def prepare_profile_bundle(bundle: dict) -> dict:
    """Return one contiguous profile bundle with explicit float32 values."""
    values = np.ascontiguousarray(bundle["values"], dtype=np.float32)
    lengths = np.ascontiguousarray(bundle["lengths"], dtype=np.int64)
    return pack_profile_bundle(values, lengths, SCORE_PADDING)


def calc_co(scores1: np.ndarray, scores2: np.ndarray, eps: float = float(PROFILE_EPS)) -> float:
    """Compute the CO score over one selected window collection."""
    values1 = np.asarray(scores1, dtype=np.float32).ravel()
    values2 = np.asarray(scores2, dtype=np.float32).ravel()
    denom = float(min(values1.sum(), values2.sum()))
    if denom <= eps:
        return 0.0
    return float(np.minimum(values1, values2).sum() / denom)


def calc_dice(scores1: np.ndarray, scores2: np.ndarray, eps: float = float(PROFILE_EPS)) -> float:
    """Compute the Dice score over one selected window collection."""
    values1 = np.asarray(scores1, dtype=np.float32).ravel()
    values2 = np.asarray(scores2, dtype=np.float32).ravel()
    denom = float(values1.sum() + values2.sum())
    if denom <= eps:
        return 0.0
    return float((2.0 * np.minimum(values1, values2).sum()) / denom)


def rowwise_cosine(x: np.ndarray, y: np.ndarray, eps: float = float(PROFILE_EPS)) -> np.ndarray:
    """Compute one cosine value per selected window."""
    values_x = np.asarray(x, dtype=np.float32)
    values_y = np.asarray(y, dtype=np.float32)
    if values_x.shape != values_y.shape:
        raise ValueError("x and y must have the same shape.")
    if values_x.ndim != WINDOW_MATRIX_NDIM:
        raise ValueError("x and y must be 2D arrays.")

    dots = np.sum(values_x * values_y, axis=1)
    norms = np.linalg.norm(values_x, axis=1) * np.linalg.norm(values_y, axis=1)
    out = np.full(values_x.shape[0], np.nan, dtype=np.float32)
    valid = norms > eps
    out[valid] = dots[valid] / norms[valid]
    return out


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{key}-{params[key]}" for key in keys)
