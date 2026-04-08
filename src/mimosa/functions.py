from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mimosa.matrix import MatrixData

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
PROFILE_DATA_DTYPE = np.float32
PROFILE_LENGTHS_DTYPE = np.int64
PROFILE_EPS = np.float32(1e-6)


def pfm_to_pwm(pfm):
    """Convert Position Frequency Matrix to Position Weight Matrix."""
    pwm = np.log((pfm + 0.0001) / 0.25)
    return pwm


def pcm_to_pfm(pcm):
    """Convert Position Count Matrix to Position Frequency Matrix."""
    number_of_sites = pcm.sum(axis=0)
    nuc_pseudo = 0.25
    pfm = (pcm + nuc_pseudo) / (number_of_sites + 1)
    return pfm


def _flatten_scan_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize any motif tensor to a 2D scan matrix of shape [5**k, motif_len]."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    return np.ascontiguousarray(arr.reshape(-1, arr.shape[-1]), dtype=np.float32)


def score_seq(num_site, kmer, model):
    """Compute score for a sequence site using a k-mer model."""
    matrix = _flatten_scan_matrix(model)
    score = 0.0
    seq_len = num_site.shape[0]
    for i in range(seq_len - kmer + 1):
        score_idx = 0
        for j in range(kmer):
            score_idx = score_idx * 5 + int(num_site[i + j])
        score += float(matrix[score_idx, i])
    return score


@functools.lru_cache(maxsize=None)
def _build_batch_score_kernel(max_width: int, motif_len: int, kmer: int, with_context: bool, is_revcomp: bool):
    """Create a shape-specialized JAX batch-scoring kernel."""
    max_scores = max(max_width - motif_len + 1, 0)
    context_len = max(kmer - 1, 0)
    window_size = motif_len + context_len
    rc_table = jnp.asarray(RC_TABLE, dtype=jnp.int32)

    def _score_window(window: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
        total = jnp.float32(0.0)
        for pos in range(motif_len):
            score_idx = jnp.int32(0)
            for step in range(kmer):
                score_idx = score_idx * 5 + window[pos + step].astype(jnp.int32)
            total = total + matrix[score_idx, pos]
        return total

    if with_context:

        @jax.jit
        def kernel(sequences: jnp.ndarray, lengths: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
            padded = jnp.pad(sequences, ((0, 0), (context_len, context_len)), constant_values=4)

            def score_row(seq_len: jnp.ndarray, padded_row: jnp.ndarray) -> jnp.ndarray:
                row_out = jnp.zeros((max_scores,), dtype=jnp.float32)
                n_scores = jnp.maximum(seq_len - motif_len + 1, 0)

                def body(pos: int, acc: jnp.ndarray) -> jnp.ndarray:
                    start = pos if not is_revcomp else pos + context_len
                    window = lax.dynamic_slice(padded_row, (start,), (window_size,))
                    if is_revcomp:
                        window = rc_table[jnp.flip(window)]
                    return acc.at[pos].set(_score_window(window, matrix))

                return lax.fori_loop(0, n_scores, body, row_out)

            return jax.vmap(score_row)(lengths, padded)

    else:

        @jax.jit
        def kernel(sequences: jnp.ndarray, lengths: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
            def score_row(seq_row: jnp.ndarray, seq_len: jnp.ndarray) -> jnp.ndarray:
                row_out = jnp.zeros((max_scores,), dtype=jnp.float32)
                n_scores = jnp.maximum(seq_len - motif_len + 1, 0)

                def body(pos: int, acc: jnp.ndarray) -> jnp.ndarray:
                    window = lax.dynamic_slice(seq_row, (pos,), (motif_len,))
                    if is_revcomp:
                        window = rc_table[jnp.flip(window)]
                    return acc.at[pos].set(_score_window(window, matrix))

                return lax.fori_loop(0, n_scores, body, row_out)

            return jax.vmap(score_row)(sequences, lengths)

    return kernel


def batch_all_scores(
    sequences: MatrixData,
    matrix: np.ndarray,
    kmer: int = 1,
    is_revcomp: bool = False,
    with_context: bool = False,
) -> MatrixData:
    """Compute scores for all sequence rows in MatrixData."""
    seq_matrix = np.ascontiguousarray(sequences.matrix, dtype=np.int8)
    seq_lengths = np.ascontiguousarray(sequences.lengths, dtype=np.int64)
    scan_matrix = _flatten_scan_matrix(matrix)
    motif_len = int(scan_matrix.shape[1])
    out_lengths = np.maximum(seq_lengths - motif_len + 1, 0).astype(np.int64, copy=False)

    if seq_matrix.shape[0] == 0:
        empty = np.empty((0, int(out_lengths.max(initial=0))), dtype=np.float32)
        return MatrixData(empty, out_lengths, pad_value=np.float32(0.0))

    kernel = _build_batch_score_kernel(seq_matrix.shape[1], motif_len, int(kmer), bool(with_context), bool(is_revcomp))
    scores = kernel(
        jnp.asarray(seq_matrix, dtype=jnp.int32),
        jnp.asarray(seq_lengths, dtype=jnp.int32),
        jnp.asarray(scan_matrix, dtype=jnp.float32),
    )
    return MatrixData(np.asarray(scores, dtype=np.float32), out_lengths, pad_value=np.float32(0.0))


def precision_recall_curve(classification, scores):
    """Compute precision-recall curve."""
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

    tp = 0
    fp = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    true_false_ratio = 1.0 if number_of_false == 0 else number_of_true / number_of_false

    position = 1
    score = sorted_scores[0]

    for i in range(len(scores)):
        current_score = sorted_scores[i]
        current_flag = sorted_classification[i]

        if current_flag == 1:
            tp += 1
        else:
            fp += 1

        if i == len(scores) - 1 or score != sorted_scores[i + 1]:
            uniq_scores[position] = current_score
            precision[position] = tp / (tp + true_false_ratio * fp) if tp + fp > 0 else 1.0
            recall[position] = tp / number_of_true if number_of_true > 0 else 0.0
            position += 1
            if i < len(scores) - 1:
                score = sorted_scores[i + 1]

    return precision[:position], recall[:position], uniq_scores[:position]


def roc_curve(classification, scores):
    """Compute ROC curve."""
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

    tpr[0] = 0.0
    fpr[0] = 0.0
    uniq_scores[0] = np.inf

    tp = 0
    fp = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    position = 1
    score = sorted_scores[0]

    for i in range(len(scores)):
        current_score = sorted_scores[i]
        current_flag = sorted_classification[i]

        if current_flag == 1:
            tp += 1
        else:
            fp += 1

        if i == len(scores) - 1 or score != sorted_scores[i + 1]:
            uniq_scores[position] = current_score
            tpr[position] = tp / number_of_true if number_of_true > 0 else 0.0
            fpr[position] = fp / number_of_false if number_of_false > 0 else 0.0
            position += 1
            if i < len(scores) - 1:
                score = sorted_scores[i + 1]

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
    """Standardize partial AUC value to range [0.5, 1]."""
    denom = pauc_max - pauc_min
    if denom <= 0:
        return 0.5
    return 0.5 * (1.0 + (pauc_raw - pauc_min) / denom)


def scores_to_frequencies(scores: MatrixData) -> MatrixData:
    """Convert scores to empirical log-tail frequencies within the current sample."""
    flat = scores.flatten_valid()
    if flat.size == 0:
        empty = np.zeros_like(scores.matrix, dtype=np.float32)
        return MatrixData(empty, scores.lengths.copy(), pad_value=np.float32(0.0))

    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    surv = np.cumsum(cnt[::-1])[::-1]

    eps = 1e-12
    log_p = np.log10(flat.size + eps) - np.log10(surv + eps)
    valid_freq = log_p[inv].astype(np.float32)

    matrix = np.zeros_like(scores.matrix, dtype=np.float32)
    matrix[scores.valid_mask()] = valid_freq
    return MatrixData(matrix, scores.lengths.copy(), pad_value=np.float32(0.0))


def _normalize_profile_score_inputs(data, lengths):
    """Return profile-score inputs as contiguous matrix and lengths arrays."""
    matrix_data = data if isinstance(data, MatrixData) and lengths is None else MatrixData(data, lengths)
    return (
        np.ascontiguousarray(matrix_data.matrix, dtype=PROFILE_DATA_DTYPE),
        np.ascontiguousarray(matrix_data.lengths, dtype=PROFILE_LENGTHS_DTYPE),
    )


def _legacy_support_to_mask(active_idx, active_ptr, lengths, width):
    """Convert the former sparse support tuple to a matrix mask."""
    support = np.zeros((lengths.size, width), dtype=bool)
    active_idx = np.asarray(active_idx, dtype=np.int64)
    active_ptr = np.asarray(active_ptr, dtype=np.int64)

    for i in range(lengths.size):
        begin = int(active_ptr[i])
        end = int(active_ptr[i + 1])
        if end <= begin:
            continue
        positions = active_idx[begin:end]
        positions = positions[(positions >= 0) & (positions < lengths[i])]
        support[i, positions] = True

    return support


def _normalize_profile_support_inputs(active_idx, active_ptr, lengths, width):
    """Return support as a valid boolean matrix or None."""
    if active_idx is None or active_ptr is None:
        return None

    support = np.asarray(active_idx)
    if support.ndim == 2:
        mask = np.ascontiguousarray(support, dtype=bool)
    elif support.ndim == 1:
        mask = _legacy_support_to_mask(active_idx, active_ptr, lengths, width)
    else:
        raise ValueError("Unsupported support representation")

    valid = np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]
    return np.logical_and(mask, valid)


def build_profile_support(data, lengths, min_value):
    """Build threshold support as a boolean matrix with explicit row lengths."""
    matrix, norm_lengths = _normalize_profile_score_inputs(data, lengths)
    valid = np.arange(matrix.shape[1], dtype=np.int64)[None, :] < norm_lengths[:, None]
    support = np.logical_and(matrix >= np.float32(min_value), valid)
    return support, norm_lengths.copy()


@functools.lru_cache(maxsize=None)
def _build_profile_stats_kernel(width1: int, width2: int, search_range: int, mode: str):
    """Create a shape-specialized JAX kernel for profile overlap statistics."""
    n_offsets = 2 * search_range + 1

    @jax.jit
    def kernel(matrix1, lengths1, matrix2, lengths2, support1, support2, min_value):
        def seq_body(i, state):
            inters, sum1s, sum2s = state
            row1 = matrix1[i]
            row2 = matrix2[i]
            len1 = lengths1[i]
            len2 = lengths2[i]
            sup1 = support1[i]
            sup2 = support2[i]

            def offset_body(k, current):
                current_inters, current_sum1s, current_sum2s = current
                offset = k - search_range
                idx1 = jnp.maximum(offset, 0)
                idx2 = jnp.maximum(-offset, 0)
                overlap = jnp.minimum(len1 - idx1, len2 - idx2)
                overlap = jnp.maximum(overlap, 0)

                def pos_body(j, accum):
                    inter, sum1, sum2 = accum
                    v1 = row1[idx1 + j]
                    v2 = row2[idx2 + j]

                    if mode == "all":
                        active = jnp.bool_(True)
                    elif mode == "threshold":
                        active = jnp.logical_or(v1 >= min_value, v2 >= min_value)
                    else:
                        active = jnp.logical_or(sup1[idx1 + j], sup2[idx2 + j])

                    inter = inter + jnp.where(active, jnp.minimum(v1, v2), jnp.float32(0.0))
                    sum1 = sum1 + jnp.where(active, v1, jnp.float32(0.0))
                    sum2 = sum2 + jnp.where(active, v2, jnp.float32(0.0))
                    return inter, sum1, sum2

                inter, sum1, sum2 = lax.fori_loop(
                    0,
                    overlap,
                    pos_body,
                    (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                )
                current_inters = current_inters.at[k].add(inter)
                current_sum1s = current_sum1s.at[k].add(sum1)
                current_sum2s = current_sum2s.at[k].add(sum2)
                return current_inters, current_sum1s, current_sum2s

            return lax.fori_loop(0, n_offsets, offset_body, (inters, sum1s, sum2s))

        initial = (
            jnp.zeros((n_offsets,), dtype=jnp.float32),
            jnp.zeros((n_offsets,), dtype=jnp.float32),
            jnp.zeros((n_offsets,), dtype=jnp.float32),
        )
        return lax.fori_loop(0, matrix1.shape[0], seq_body, initial)

    return kernel


def _reduce_profile_score(inters, sum1s, sum2s, search_range, metric):
    """Reduce accumulated overlap statistics to the best score and offset."""
    if metric == "co":
        denom = np.minimum(sum1s, sum2s)
        scores = np.full_like(inters, -1.0, dtype=np.float32)
        np.divide(inters, denom, out=scores, where=denom > PROFILE_EPS)
    elif metric == "dice":
        denom = sum1s + sum2s
        scores = np.full_like(inters, -1.0, dtype=np.float32)
        np.divide(2.0 * inters, denom, out=scores, where=denom > PROFILE_EPS)
    else:
        raise ValueError("metric must be one of: 'co', 'dice'")

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < 0.0:
        return np.float32(0.0), np.int64(0)
    return np.float32(best_score), np.int64(best_idx - search_range)


def fast_profile_score(
    data1,
    offsets1,
    data2,
    offsets2,
    search_range,
    min_value=0.0,
    metric="co",
    active_idx1=None,
    active_ptr1=None,
    active_idx2=None,
    active_ptr2=None,
):
    """Dispatch profile similarity scoring to matrix-based JAX overlap kernels."""
    matrix1, lengths1 = _normalize_profile_score_inputs(data1, offsets1)
    matrix2, lengths2 = _normalize_profile_score_inputs(data2, offsets2)
    search_range = int(search_range)
    min_value = np.float32(min_value)

    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Both profiles must contain the same number of sequences")

    support1 = _normalize_profile_support_inputs(active_idx1, active_ptr1, lengths1, matrix1.shape[1])
    support2 = _normalize_profile_support_inputs(active_idx2, active_ptr2, lengths2, matrix2.shape[1])

    if support1 is not None and support2 is not None:
        mode = "support"
    elif min_value > np.float32(0.0):
        mode = "threshold"
    else:
        mode = "all"

    if support1 is None:
        support1 = np.zeros(matrix1.shape, dtype=bool)
    if support2 is None:
        support2 = np.zeros(matrix2.shape, dtype=bool)

    kernel = _build_profile_stats_kernel(matrix1.shape[1], matrix2.shape[1], search_range, mode)
    inters, sum1s, sum2s = kernel(
        jnp.asarray(matrix1, dtype=jnp.float32),
        jnp.asarray(lengths1, dtype=jnp.int32),
        jnp.asarray(matrix2, dtype=jnp.float32),
        jnp.asarray(lengths2, dtype=jnp.int32),
        jnp.asarray(support1, dtype=jnp.bool_),
        jnp.asarray(support2, dtype=jnp.bool_),
        jnp.asarray(min_value, dtype=jnp.float32),
    )

    return _reduce_profile_score(
        np.asarray(inters, dtype=np.float32),
        np.asarray(sum1s, dtype=np.float32),
        np.asarray(sum2s, dtype=np.float32),
        search_range,
        metric,
    )


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
