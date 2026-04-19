"""Numerical helpers and Numba-backed scoring kernels."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mimosa.batches import SCORE_PADDING, batch_with_values, flatten_valid, pack_batch

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
PROFILE_EPS = np.float32(1e-6)
_PROFILE_METRIC_CO = 0
_PROFILE_METRIC_DICE = 1


def build_profile_score_options(search_range: int, min_value: float = 0.0, metric: str = "co") -> dict:
    """Build one validated options dictionary for profile scoring."""
    return {
        "search_range": int(search_range),
        "min_value": float(min_value),
        "metric": str(metric),
    }


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


def apply_score_log_tail_table(score_batch, table: np.ndarray):
    """Map one score batch to empirical log-tail values using a lookup table."""
    flat = flatten_valid(score_batch)
    if flat.size == 0:
        empty_values = np.full_like(score_batch["values"], SCORE_PADDING)
        return batch_with_values(score_batch, empty_values, padding_value=SCORE_PADDING)

    scores_col = np.asarray(table[:, 0], dtype=np.float32)
    log_tail_col = np.asarray(table[:, 1], dtype=np.float32)
    idx = np.searchsorted(-scores_col, -flat, side="left")
    idx = np.clip(idx, 0, len(log_tail_col) - 1)

    mapped = np.full(score_batch["values"].shape, SCORE_PADDING, dtype=np.float32)
    mapped[score_batch["mask"]] = log_tail_col[idx]
    return batch_with_values(score_batch, mapped, padding_value=SCORE_PADDING)


def normalize_empirical_log_tail_pair(score_batch_plus, score_batch_minus):
    """Normalize two strand score batches using one shared empirical log-tail mapping."""
    flat_plus = flatten_valid(score_batch_plus).astype(np.float32, copy=False)
    flat_minus = flatten_valid(score_batch_minus).astype(np.float32, copy=False)
    total_size = int(flat_plus.size + flat_minus.size)

    if total_size == 0:
        empty_plus = np.full(score_batch_plus["values"].shape, SCORE_PADDING, dtype=np.float32)
        empty_minus = np.full(score_batch_minus["values"].shape, SCORE_PADDING, dtype=np.float32)
        return (
            batch_with_values(score_batch_plus, empty_plus, padding_value=SCORE_PADDING),
            batch_with_values(score_batch_minus, empty_minus, padding_value=SCORE_PADDING),
        )

    combined = np.concatenate((flat_plus, flat_minus)).astype(np.float32, copy=False)
    _, inverse_idx, counts = np.unique(combined, return_inverse=True, return_counts=True)
    tail_counts = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    log_tail = (-np.log10(tail_counts / float(total_size))).astype(np.float32, copy=False)
    mapped = log_tail[inverse_idx]

    values_plus = np.full(score_batch_plus["values"].shape, SCORE_PADDING, dtype=np.float32)
    values_minus = np.full(score_batch_minus["values"].shape, SCORE_PADDING, dtype=np.float32)
    values_plus[score_batch_plus["mask"]] = mapped[: flat_plus.size]
    values_minus[score_batch_minus["mask"]] = mapped[flat_plus.size :]

    return (
        batch_with_values(score_batch_plus, values_plus, padding_value=SCORE_PADDING),
        batch_with_values(score_batch_minus, values_minus, padding_value=SCORE_PADDING),
    )


def lookup_log_tail(table: np.ndarray, score: float) -> float:
    """Convert a score to the corresponding log-tail value."""
    scores_col = table[:, 0]
    log_tail_col = table[:, 1]

    if score >= scores_col[0]:
        return float(log_tail_col[0])
    if score <= scores_col[-1]:
        return float(log_tail_col[-1])

    idx = np.searchsorted(-scores_col, -score, side="left")
    if idx >= len(log_tail_col):
        return float(log_tail_col[-1])
    return float(log_tail_col[idx])


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
    max_scores = max(int(values.shape[1]) - motif_len + 1, 0)
    out_lengths = np.maximum(lengths - motif_len + 1, 0)
    return values, lengths, model_rows, motif_len, max_scores, out_lengths


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def _scan_dense_kernel_numba(values, lengths, model_rows, kmer: int, context_len: int, n_terms: int):
    """Score one dense encoded sequence batch for one strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in prange(n_rows):
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


@njit(cache=True, parallel=True)
def _scan_dense_reverse_kernel_numba(values, lengths, model_rows, kmer: int, window_size: int, n_terms: int):
    """Score one dense encoded sequence batch on the reverse-complement strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in prange(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_reverse(seq_row, length, model_rows, pos, kmer, window_size, n_terms)
            mask[row_index, pos] = True

    return scores, mask


@njit(cache=True, parallel=True)
def _scan_dense_strands_kernel_numba(
    values, lengths, model_rows, kmer: int, context_len: int, window_size: int, n_terms: int
):
    """Score one dense encoded batch on both strands in one call."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, 2, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)

    for row_index in prange(n_rows):
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

    return pack_batch(scored_values, scored_mask, out_lengths, SCORE_PADDING)


def batch_all_scores_strands(sequences, matrix: np.ndarray, kmer: int = 1, with_context: bool = False):
    """Compute scores for both strands in one dense masked batch call."""
    values, lengths, model_rows, motif_len, max_scores, out_lengths = _prepare_scan_inputs(sequences, matrix)
    n_rows = int(values.shape[0])

    if n_rows == 0 or max_scores == 0:
        empty_batch = _empty_score_scan_batch(n_rows, max_scores, out_lengths)
        return empty_batch, _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    context_len, window_size, n_terms = _resolve_scan_layout(int(kmer), motif_len, bool(with_context))
    scored_values, scored_mask = _scan_dense_strands_kernel_numba(
        values,
        lengths,
        model_rows,
        int(kmer),
        context_len,
        window_size,
        n_terms,
    )

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


def _build_profile_threshold_mask(values, mask, min_value: float):
    """Build the effective mask used by thresholded profile scoring."""
    if min_value <= 0.0:
        return mask
    return mask & (values >= min_value)


def _prepare_profile_for_scoring(profile, min_value: float):
    """Normalize one raw profile batch for compiled scoring kernels."""
    if "active_mask" in profile and "mask" not in profile:
        return profile

    values = np.ascontiguousarray(profile["values"], dtype=np.float32)
    mask = np.ascontiguousarray(profile["mask"], dtype=bool)
    return {
        "values": values,
        "active_mask": np.ascontiguousarray(_build_profile_threshold_mask(values, mask, float(min_value)), dtype=bool),
    }


def _prepare_profile_bundle_for_scoring(bundle: dict, min_value: float) -> dict:
    """Prepare one strand bundle for repeated profile comparisons."""
    return {
        "plus": _prepare_profile_for_scoring(bundle["plus"], min_value),
        "minus": _prepare_profile_for_scoring(bundle["minus"], min_value),
    }


def _resolve_profile_metric_code(metric: str) -> int:
    """Resolve one public profile metric name to the internal kernel code."""
    if metric == "co":
        return _PROFILE_METRIC_CO
    if metric == "dice":
        return _PROFILE_METRIC_DICE
    raise ValueError("metric must be one of: 'co', 'dice'")


@njit(cache=True)
def _profile_score_numba(values1, active1, values2, active2, search_range: int, metric_code: int):
    """Compute the best profile similarity score for all offsets."""
    n_rows = values1.shape[0]
    width1 = values1.shape[1]
    width2 = values2.shape[1]
    best_score = -1.0
    best_offset = 0

    for offset in range(-search_range, search_range + 1):
        start1 = offset if offset > 0 else 0
        start2 = -offset if offset < 0 else 0
        remaining = width2 - start2
        overlap = min(width1 - start1, remaining)

        inter = 0.0
        sum1 = 0.0
        sum2 = 0.0

        if overlap > 0:
            for row_index in range(n_rows):
                for delta in range(overlap):
                    col1 = start1 + delta
                    col2 = start2 + delta
                    if active1[row_index, col1] and active2[row_index, col2]:
                        value1 = float(values1[row_index, col1])
                        value2 = float(values2[row_index, col2])
                        inter += value1 if value1 < value2 else value2
                        sum1 += value1
                        sum2 += value2

        if metric_code == _PROFILE_METRIC_CO:
            denom = sum1 if sum1 < sum2 else sum2
            score = inter / denom if denom > PROFILE_EPS else -1.0
        else:
            denom = sum1 + sum2
            score = (2.0 * inter) / denom if denom > PROFILE_EPS else -1.0

        if score > best_score:
            best_score = score
            best_offset = offset

    if best_score < 0.0:
        return np.float32(0.0), np.int32(0)
    return np.float32(best_score), np.int32(best_offset)


@njit(cache=True, parallel=True)
def _profile_score_orientations_numba(
    values1_unique,
    active1_unique,
    left_indices,
    values2_unique,
    active2_unique,
    right_indices,
    search_range: int,
    metric_code: int,
):
    """Compute profile similarity for multiple orientation pairs in one call."""
    n_pairs = left_indices.shape[0]
    scores = np.zeros(n_pairs, dtype=np.float32)
    offsets = np.zeros(n_pairs, dtype=np.int32)

    for pair_index in prange(n_pairs):
        left_index = left_indices[pair_index]
        right_index = right_indices[pair_index]
        score, offset = _profile_score_numba(
            values1_unique[left_index],
            active1_unique[left_index],
            values2_unique[right_index],
            active2_unique[right_index],
            search_range,
            metric_code,
        )
        scores[pair_index] = score
        offsets[pair_index] = offset

    return scores, offsets


def fast_profile_score(profile1, profile2, options: dict):
    """Compute the best overlap-based similarity score for two dense score batches."""
    min_value = float(options.get("min_value", 0.0))
    prepared1 = _prepare_profile_for_scoring(profile1, min_value)
    prepared2 = _prepare_profile_for_scoring(profile2, min_value)
    values1 = prepared1["values"]
    values2 = prepared2["values"]

    if values1.shape[0] != values2.shape[0]:
        raise ValueError("profile batches must have the same number of rows")

    if values1.shape[1] == 0 or values2.shape[1] == 0:
        return 0.0, 0

    score, offset = _profile_score_numba(
        values1,
        prepared1["active_mask"],
        values2,
        prepared2["active_mask"],
        int(options["search_range"]),
        _resolve_profile_metric_code(str(options.get("metric", "co"))),
    )
    return float(score), int(offset)


def _prepare_profile_pairs(profile_pairs: list[tuple[dict, dict]], min_value: float) -> list[tuple[dict, dict]]:
    """Prepare all profile pairs while reusing repeated inputs."""
    prepared_cache = {}

    def prepare_cached(profile):
        key = id(profile)
        cached = prepared_cache.get(key)
        if cached is None:
            cached = _prepare_profile_for_scoring(profile, min_value)
            prepared_cache[key] = cached
        return cached

    return [
        (prepare_cached(left_profile), prepare_cached(right_profile))
        for left_profile, right_profile in profile_pairs
    ]


def _validate_profile_pair_shapes(prepared_pairs: list[tuple[dict, dict]]) -> tuple[int, int, int]:
    """Validate that all prepared profile pairs share the same geometry."""
    first_left, first_right = prepared_pairs[0]
    expected_rows = int(first_left["values"].shape[0])
    expected_left_width = int(first_left["values"].shape[1])
    expected_right_width = int(first_right["values"].shape[1])

    for left_profile, right_profile in prepared_pairs:
        if left_profile["values"].shape[0] != expected_rows or right_profile["values"].shape[0] != expected_rows:
            raise ValueError("all profile pairs must have the same number of rows")
        if left_profile["values"].shape[1] != expected_left_width:
            raise ValueError("left profile widths must match across orientation pairs")
        if right_profile["values"].shape[1] != expected_right_width:
            raise ValueError("right profile widths must match across orientation pairs")

    return expected_rows, expected_left_width, expected_right_width


def _index_unique_profiles(prepared_pairs: list[tuple[dict, dict]]):
    """Deduplicate prepared profiles and keep pair indices for the kernel call."""
    left_unique = []
    left_index = {}
    right_unique = []
    right_index = {}
    left_indices = []
    right_indices = []

    for left_profile, right_profile in prepared_pairs:
        left_key = id(left_profile)
        if left_key not in left_index:
            left_index[left_key] = len(left_unique)
            left_unique.append(left_profile)
        left_indices.append(left_index[left_key])

        right_key = id(right_profile)
        if right_key not in right_index:
            right_index[right_key] = len(right_unique)
            right_unique.append(right_profile)
        right_indices.append(right_index[right_key])

    return left_unique, left_indices, right_unique, right_indices


def fast_profile_score_orientations(profile_pairs: list[tuple[dict, dict]], options: dict):
    """Compute profile similarity for all orientation pairs in one call."""
    if not profile_pairs:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    min_value = float(options.get("min_value", 0.0))
    prepared_pairs = _prepare_profile_pairs(profile_pairs, min_value)
    _, expected_left_width, expected_right_width = _validate_profile_pair_shapes(prepared_pairs)

    if expected_left_width == 0 or expected_right_width == 0:
        n_pairs = len(profile_pairs)
        return np.zeros(n_pairs, dtype=np.float32), np.zeros(n_pairs, dtype=np.int32)

    left_unique, left_indices, right_unique, right_indices = _index_unique_profiles(prepared_pairs)

    values1_unique = np.ascontiguousarray(
        np.stack([profile["values"] for profile in left_unique], axis=0),
        dtype=np.float32,
    )
    active1_unique = np.ascontiguousarray(
        np.stack([profile["active_mask"] for profile in left_unique], axis=0),
        dtype=bool,
    )
    values2_unique = np.ascontiguousarray(
        np.stack([profile["values"] for profile in right_unique], axis=0),
        dtype=np.float32,
    )
    active2_unique = np.ascontiguousarray(
        np.stack([profile["active_mask"] for profile in right_unique], axis=0),
        dtype=bool,
    )

    scores, offsets = _profile_score_orientations_numba(
        values1_unique,
        active1_unique,
        np.asarray(left_indices, dtype=np.int32),
        values2_unique,
        active2_unique,
        np.asarray(right_indices, dtype=np.int32),
        int(options["search_range"]),
        _resolve_profile_metric_code(str(options.get("metric", "co"))),
    )

    return np.asarray(scores, dtype=np.float32), np.asarray(offsets, dtype=np.int32)


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{key}-{params[key]}" for key in keys)
