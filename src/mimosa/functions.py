"""Numerical helpers and JAX-backed scoring kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mimosa.batches import SCORE_PADDING, batch_with_values, flatten_valid, pack_batch

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
PROFILE_EPS = np.float32(1e-6)
_JAX_RC_TABLE = jnp.asarray(RC_TABLE, dtype=jnp.int32)


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
    return arr.reshape((-1, arr.shape[-1]))


def _score_dense_sources(seq_row, length, src, *, max_len, powers, model_rows, term_idx, apply_revcomp: bool):
    """Score one source-index tensor against the shared flattened motif rows."""
    safe_src = jnp.clip(src, 0, max_len - 1)
    picked = seq_row[safe_src]
    valid_src = (src >= 0) & (src < length)
    encoded = jnp.where(valid_src, picked, 4)

    if apply_revcomp:
        encoded = _JAX_RC_TABLE[encoded]

    codes = jnp.sum(encoded * powers, axis=-1)
    contributions = model_rows[codes, term_idx[None, :]]
    return jnp.sum(contributions, axis=-1)


@partial(jax.jit, static_argnames=("kmer", "with_context", "is_revcomp"))
def _scan_dense_kernel(values, lengths, model_rows, *, kmer: int, with_context: bool, is_revcomp: bool):
    """Score one dense encoded sequence batch with a shared JAX kernel."""
    max_len = values.shape[1]
    motif_len = model_rows.shape[-1]
    context_len = kmer - 1 if with_context else 0
    window_size = motif_len + context_len
    n_terms = window_size - kmer + 1
    max_scores = max(max_len - motif_len + 1, 0)

    positions = jnp.arange(max_scores, dtype=jnp.int32)
    term_idx = jnp.arange(n_terms, dtype=jnp.int32)
    kmer_idx = jnp.arange(kmer, dtype=jnp.int32)
    powers = (5 ** jnp.arange(kmer - 1, -1, -1, dtype=jnp.int32)).astype(jnp.int32)

    def scan_row(seq_row, length):
        n_scores = jnp.maximum(length - motif_len + 1, 0)
        valid_positions = positions < n_scores

        if is_revcomp:
            src = positions[:, None, None] + (window_size - 1 - (term_idx[None, :, None] + kmer_idx[None, None, :]))
        else:
            src = positions[:, None, None] - context_len + term_idx[None, :, None] + kmer_idx[None, None, :]

        scores = _score_dense_sources(
            seq_row,
            length,
            src,
            max_len=max_len,
            powers=powers,
            model_rows=model_rows,
            term_idx=term_idx,
            apply_revcomp=is_revcomp,
        )
        padded_scores = jnp.where(valid_positions, scores, jnp.float32(SCORE_PADDING))
        return padded_scores, valid_positions

    return jax.vmap(scan_row)(values, lengths)


@partial(jax.jit, static_argnames=("kmer", "with_context"))
def _scan_dense_strands_kernel(values, lengths, model_rows, *, kmer: int, with_context: bool):
    """Score one dense encoded batch on both strands in a single JAX call."""
    max_len = values.shape[1]
    motif_len = model_rows.shape[-1]
    context_len = kmer - 1 if with_context else 0
    window_size = motif_len + context_len
    n_terms = window_size - kmer + 1
    max_scores = max(max_len - motif_len + 1, 0)

    positions = jnp.arange(max_scores, dtype=jnp.int32)
    term_idx = jnp.arange(n_terms, dtype=jnp.int32)
    kmer_idx = jnp.arange(kmer, dtype=jnp.int32)
    powers = (5 ** jnp.arange(kmer - 1, -1, -1, dtype=jnp.int32)).astype(jnp.int32)

    def scan_row(seq_row, length):
        n_scores = jnp.maximum(length - motif_len + 1, 0)
        valid_positions = positions < n_scores
        forward_src = positions[:, None, None] - context_len + term_idx[None, :, None] + kmer_idx[None, None, :]
        reverse_src = positions[:, None, None] + (
            window_size - 1 - (term_idx[None, :, None] + kmer_idx[None, None, :])
        )

        forward_scores = _score_dense_sources(
            seq_row,
            length,
            forward_src,
            max_len=max_len,
            powers=powers,
            model_rows=model_rows,
            term_idx=term_idx,
            apply_revcomp=False,
        )
        reverse_scores = _score_dense_sources(
            seq_row,
            length,
            reverse_src,
            max_len=max_len,
            powers=powers,
            model_rows=model_rows,
            term_idx=term_idx,
            apply_revcomp=True,
        )

        padded_forward = jnp.where(valid_positions, forward_scores, jnp.float32(SCORE_PADDING))
        padded_reverse = jnp.where(valid_positions, reverse_scores, jnp.float32(SCORE_PADDING))
        strand_scores = jnp.stack((padded_forward, padded_reverse), axis=0)
        strand_mask = jnp.stack((valid_positions, valid_positions), axis=0)
        return strand_scores, strand_mask

    return jax.vmap(scan_row)(values, lengths)


def _empty_score_scan_batch(n_rows: int, max_scores: int, out_lengths: np.ndarray):
    """Return one empty score batch with the requested output geometry."""
    empty_values = np.full((n_rows, max_scores), SCORE_PADDING, dtype=np.float32)
    empty_mask = np.zeros((n_rows, max_scores), dtype=bool)
    return pack_batch(empty_values, empty_mask, out_lengths, SCORE_PADDING)


def batch_all_scores(
    sequences, matrix: np.ndarray, kmer: int = 1, is_revcomp: bool = False, with_context: bool = False
):
    """Compute scores for all sequences in one dense masked batch."""
    values_raw = sequences["values"]
    lengths_raw = sequences["lengths"]
    n_rows = int(values_raw.shape[0])

    model_rows_np = _prepare_model_rows(matrix)
    motif_len = int(model_rows_np.shape[-1])
    max_scores = max(int(values_raw.shape[1]) - motif_len + 1, 0)
    lengths_np = np.asarray(lengths_raw, dtype=np.int64)
    out_lengths = np.maximum(lengths_np - motif_len + 1, 0)

    if n_rows == 0 or max_scores == 0:
        return _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    values = jnp.asarray(values_raw, dtype=jnp.int32)
    lengths = jnp.asarray(lengths_raw, dtype=jnp.int32)
    model_rows = jnp.asarray(model_rows_np, dtype=jnp.float32)

    scored_values, scored_mask = _scan_dense_kernel(
        values,
        lengths,
        model_rows,
        kmer=int(kmer),
        with_context=bool(with_context),
        is_revcomp=bool(is_revcomp),
    )

    return pack_batch(scored_values, scored_mask, out_lengths, SCORE_PADDING)


def batch_all_scores_strands(sequences, matrix: np.ndarray, kmer: int = 1, with_context: bool = False):
    """Compute scores for both strands in one dense masked batch call."""
    values_raw = sequences["values"]
    lengths_raw = sequences["lengths"]
    n_rows = int(values_raw.shape[0])

    model_rows_np = _prepare_model_rows(matrix)
    motif_len = int(model_rows_np.shape[-1])
    max_scores = max(int(values_raw.shape[1]) - motif_len + 1, 0)
    lengths_np = np.asarray(lengths_raw, dtype=np.int64)
    out_lengths = np.maximum(lengths_np - motif_len + 1, 0)

    if n_rows == 0 or max_scores == 0:
        empty_batch = _empty_score_scan_batch(n_rows, max_scores, out_lengths)
        return empty_batch, _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    values = jnp.asarray(values_raw, dtype=jnp.int32)
    lengths = jnp.asarray(lengths_raw, dtype=jnp.int32)
    model_rows = jnp.asarray(model_rows_np, dtype=jnp.float32)

    scored_values, scored_mask = _scan_dense_strands_kernel(
        values,
        lengths,
        model_rows,
        kmer=int(kmer),
        with_context=bool(with_context),
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


def _profile_score_kernel_impl(values1, mask1, values2, mask2, *, search_range: int, min_value: float, metric: str):
    """Compute the best profile similarity score for all offsets."""
    width1 = values1.shape[1]
    width2 = values2.shape[1]
    positions1 = jnp.arange(width1, dtype=jnp.int32)
    offsets = jnp.arange(-search_range, search_range + 1, dtype=jnp.int32)

    def offset_stats(offset):
        pos2 = positions1 - offset
        safe_pos2 = jnp.clip(pos2, 0, width2 - 1)
        gathered_values2 = values2[:, safe_pos2]
        gathered_mask2 = mask2[:, safe_pos2]

        pair_mask = mask1 & gathered_mask2 & ((pos2 >= 0) & (pos2 < width2))[None, :]
        threshold_mask = (values1 >= min_value) | (gathered_values2 >= min_value)
        pair_mask = pair_mask & jnp.where(min_value > 0.0, threshold_mask, jnp.ones_like(pair_mask))

        inter = jnp.sum(jnp.where(pair_mask, jnp.minimum(values1, gathered_values2), 0.0))
        sum1 = jnp.sum(jnp.where(pair_mask, values1, 0.0))
        sum2 = jnp.sum(jnp.where(pair_mask, gathered_values2, 0.0))
        return inter, sum1, sum2

    inters, sum1s, sum2s = jax.vmap(offset_stats)(offsets)

    if metric == "co":
        denom = jnp.minimum(sum1s, sum2s)
        scores = jnp.where(denom > PROFILE_EPS, inters / denom, -1.0)
    elif metric == "dice":
        denom = sum1s + sum2s
        scores = jnp.where(denom > PROFILE_EPS, (2.0 * inters) / denom, -1.0)
    else:
        raise ValueError("metric must be one of: 'co', 'dice'")

    best_index = jnp.argmax(scores)
    best_score = scores[best_index]
    best_offset = offsets[best_index]
    valid = best_score >= 0.0
    return jnp.where(valid, best_score, 0.0), jnp.where(valid, best_offset, 0)


@partial(jax.jit, static_argnames=("search_range", "metric"))
def _profile_score_kernel(values1, mask1, values2, mask2, *, search_range: int, min_value: float, metric: str):
    """JIT wrapper for one profile-pair score."""
    return _profile_score_kernel_impl(
        values1,
        mask1,
        values2,
        mask2,
        search_range=search_range,
        min_value=min_value,
        metric=metric,
    )


@partial(jax.jit, static_argnames=("search_range", "metric"))
def _profile_score_orientations_kernel(
    values1_stack, mask1_stack, values2_stack, mask2_stack, *, search_range: int, min_value: float, metric: str
):
    """Compute profile similarity for multiple orientation pairs in one JAX call."""

    def score_one(values1, mask1, values2, mask2):
        return _profile_score_kernel_impl(
            values1,
            mask1,
            values2,
            mask2,
            search_range=search_range,
            min_value=min_value,
            metric=metric,
        )

    return jax.vmap(score_one)(values1_stack, mask1_stack, values2_stack, mask2_stack)


def fast_profile_score(profile1, profile2, options: dict):
    """Compute the best overlap-based similarity score for two dense score batches."""
    if profile1["values"].shape[0] != profile2["values"].shape[0]:
        raise ValueError("profile batches must have the same number of rows")

    values1 = jnp.asarray(profile1["values"], dtype=jnp.float32)
    mask1 = jnp.asarray(profile1["mask"], dtype=bool)
    values2 = jnp.asarray(profile2["values"], dtype=jnp.float32)
    mask2 = jnp.asarray(profile2["mask"], dtype=bool)

    if values1.shape[1] == 0 or values2.shape[1] == 0:
        return 0.0, 0

    score, offset = _profile_score_kernel(
        values1,
        mask1,
        values2,
        mask2,
        search_range=int(options["search_range"]),
        min_value=float(options.get("min_value", 0.0)),
        metric=str(options.get("metric", "co")),
    )
    return float(score), int(offset)


def fast_profile_score_orientations(profile_pairs: list[tuple[dict, dict]], options: dict):
    """Compute profile similarity for all orientation pairs in one JAX call."""
    if not profile_pairs:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    first_left, first_right = profile_pairs[0]
    expected_rows = int(first_left["values"].shape[0])
    expected_left_width = int(first_left["values"].shape[1])
    expected_right_width = int(first_right["values"].shape[1])

    for left_profile, right_profile in profile_pairs:
        if left_profile["values"].shape[0] != expected_rows or right_profile["values"].shape[0] != expected_rows:
            raise ValueError("all profile pairs must have the same number of rows")
        if left_profile["values"].shape[1] != expected_left_width:
            raise ValueError("left profile widths must match across orientation pairs")
        if right_profile["values"].shape[1] != expected_right_width:
            raise ValueError("right profile widths must match across orientation pairs")

    if expected_left_width == 0 or expected_right_width == 0:
        n_pairs = len(profile_pairs)
        return np.zeros(n_pairs, dtype=np.float32), np.zeros(n_pairs, dtype=np.int32)

    values1_stack = jnp.stack([jnp.asarray(pair[0]["values"], dtype=jnp.float32) for pair in profile_pairs], axis=0)
    mask1_stack = jnp.stack([jnp.asarray(pair[0]["mask"], dtype=bool) for pair in profile_pairs], axis=0)
    values2_stack = jnp.stack([jnp.asarray(pair[1]["values"], dtype=jnp.float32) for pair in profile_pairs], axis=0)
    mask2_stack = jnp.stack([jnp.asarray(pair[1]["mask"], dtype=bool) for pair in profile_pairs], axis=0)

    scores, offsets = _profile_score_orientations_kernel(
        values1_stack,
        mask1_stack,
        values2_stack,
        mask2_stack,
        search_range=int(options["search_range"]),
        min_value=float(options.get("min_value", 0.0)),
        metric=str(options.get("metric", "co")),
    )

    return np.asarray(scores, dtype=np.float32), np.asarray(offsets, dtype=np.int32)


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{key}-{params[key]}" for key in keys)
