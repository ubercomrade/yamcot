from dataclasses import dataclass

import numpy as np
from numba import float32, int64, njit, prange, types

from mimosa.ragged import RaggedData

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
PROFILE_DATA_DTYPE = np.float32
PROFILE_OFFSETS_DTYPE = np.int64
PROFILE_EPS = np.float32(1e-6)

_PROFILE_STATS_SIGNATURE = types.Tuple((float32[::1], float32[::1], float32[::1]))(
    float32[::1], int64[::1], float32[::1], int64[::1], int64
)
_PROFILE_STATS_THRESHOLD_SIGNATURE = types.Tuple((float32[::1], float32[::1], float32[::1]))(
    float32[::1], int64[::1], float32[::1], int64[::1], int64, float32
)
_PROFILE_SUPPORT_SIGNATURE = types.Tuple((int64[::1], int64[::1]))(float32[::1], int64[::1], float32)
_PROFILE_SPARSE_STATS_SIGNATURE = types.Tuple((float32[::1], float32[::1], float32[::1]))(
    float32[::1], int64[::1], int64[::1], int64[::1], float32[::1], int64[::1], int64[::1], int64[::1], int64
)
_PROFILE_REDUCER_SIGNATURE = types.Tuple((float32, int64))(float32[::1], float32[::1], float32[::1], int64)


@dataclass(frozen=True)
class ProfileScoreOptions:
    """Options for overlap-based profile scoring."""

    search_range: int
    min_value: float = 0.0
    metric: str = "co"


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
    flat = np.asarray(scores, dtype=np.float64).ravel()
    if flat.size == 0:
        return np.array([[0.0, 0.0]], dtype=np.float64)

    scores_sorted = np.sort(flat)[::-1]
    unique_scores, counts = np.unique(scores_sorted, return_counts=True)
    unique_scores = unique_scores[::-1]
    counts = counts[::-1]

    cum_counts = np.cumsum(counts)
    tail_probabilities = cum_counts / flat.size
    log_tail = -np.log10(tail_probabilities)
    return np.column_stack([unique_scores, log_tail]).astype(np.float64)


def apply_score_log_tail_table(ragged_scores: RaggedData, table: np.ndarray) -> RaggedData:
    """Map raw scores to log-tail values using a lookup table."""
    flat = ragged_scores.data
    if flat.size == 0:
        return RaggedData(np.zeros(0, dtype=np.float32), ragged_scores.offsets)

    scores_col = table[:, 0]
    log_tail_col = table[:, 1]

    idx = np.searchsorted(-scores_col, -flat, side="left")
    idx = np.clip(idx, 0, len(log_tail_col) - 1)
    return RaggedData(log_tail_col[idx].astype(np.float32), ragged_scores.offsets)


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


@njit
def score_seq(num_site, kmer, model):
    """Compute score for a sequence site using a k-mer model."""
    score = 0.0
    seq_len = num_site.shape[0]
    for i in range(seq_len - kmer + 1):
        score_idx = 0
        for j in range(kmer):
            score_idx = score_idx * 5 + num_site[i + j]
        score += model.flat[score_idx * model.shape[-1] + i]

    return score


@njit(inline="always", cache=False)
def _fill_rc_buffer(data, start, length, buffer):
    """Fill a buffer with reverse-complement values without allocations."""
    for j in range(length):
        val = data[start + length - 1 - j]
        buffer[j] = RC_TABLE[val]


@njit(inline="always", cache=False)
def _fill_forward_context_window(seq, seq_len, site_start, motif_len, context_len, buffer):
    """Fill one forward-scanning context window with N-padding."""
    buffer[:] = 4

    window_start = site_start - context_len
    window_end = site_start + motif_len
    actual_start = 0 if window_start < 0 else window_start
    actual_end = seq_len if window_end > seq_len else window_end
    dest_start = 0 if window_start >= 0 else -window_start
    copy_len = actual_end - actual_start

    if copy_len > 0:
        buffer[dest_start : dest_start + copy_len] = seq[actual_start:actual_end]


@njit(parallel=True, fastmath=True, cache=False)
def _batch_all_scores_jit(data, offsets, matrix, kmer, is_revcomp):
    """Compute PWM scores in batch with a JIT-compiled kernel."""
    n_seq = len(offsets) - 1
    m = matrix.shape[-1]

    new_offsets = np.zeros(n_seq + 1, dtype=np.int64)
    for i in range(n_seq):
        seq_len = offsets[i + 1] - offsets[i]
        if seq_len >= m:
            new_offsets[i + 1] = seq_len - m + 1

    for i in range(n_seq):
        new_offsets[i + 1] += new_offsets[i]

    total_scores = new_offsets[n_seq]
    results = np.zeros(total_scores, dtype=np.float32)

    for i in prange(n_seq):
        start = offsets[i]
        seq_len = offsets[i + 1] - start
        out_start = new_offsets[i]
        n_scores = new_offsets[i + 1] - out_start

        if n_scores > 0:
            if not is_revcomp:
                for k in range(n_scores):
                    num_site = data[start + k : start + k + m]
                    results[out_start + k] = score_seq(num_site, kmer, matrix)
            else:
                rc_seq = np.empty(seq_len, dtype=data.dtype)
                _fill_rc_buffer(data, start, seq_len, rc_seq)

                for k in range(n_scores):
                    num_site = rc_seq[k : k + m]
                    results[out_start + (n_scores - 1 - k)] = score_seq(num_site, kmer, matrix)

    return results, new_offsets


@njit(parallel=True, fastmath=True, cache=False)
def _batch_all_scores_with_context_jit(data, offsets, matrix, kmer, is_revcomp):
    """Compute BaMM scores in batch with context-aware padding."""
    n_seq = len(offsets) - 1
    m = matrix.shape[-1]
    context_len = kmer - 1
    window_size = m + context_len

    new_offsets = np.zeros(n_seq + 1, dtype=np.int64)
    for i in range(n_seq):
        seq_len = offsets[i + 1] - offsets[i]
        if seq_len >= m:
            new_offsets[i + 1] = seq_len - m + 1

    for i in range(n_seq):
        new_offsets[i + 1] += new_offsets[i]

    total_scores = new_offsets[n_seq]
    results = np.zeros(total_scores, dtype=np.float32)

    for i in prange(n_seq):
        start = offsets[i]
        seq_len = offsets[i + 1] - start
        out_start = new_offsets[i]
        n_scores = new_offsets[i + 1] - out_start

        if n_scores > 0:
            seq = data[start : start + seq_len]
            if is_revcomp:
                seq = np.empty(seq_len, dtype=data.dtype)
                _fill_rc_buffer(data, start, seq_len, seq)

            site_buffer = np.full(window_size, 4, dtype=data.dtype)
            for k in range(n_scores):
                _fill_forward_context_window(seq, seq_len, k, m, context_len, site_buffer)
                result_idx = out_start + k if not is_revcomp else out_start + (n_scores - 1 - k)
                results[result_idx] = score_seq(site_buffer, kmer, matrix)

    return results, new_offsets


def batch_all_scores(
    sequences: RaggedData,
    matrix: np.ndarray,
    kmer: int = 1,
    is_revcomp: bool = False,
    with_context: bool = False,
) -> RaggedData:
    """Compute scores for all sequences in RaggedData."""
    if with_context:
        data, offsets = _batch_all_scores_with_context_jit(sequences.data, sequences.offsets, matrix, kmer, is_revcomp)
    else:
        data, offsets = _batch_all_scores_jit(sequences.data, sequences.offsets, matrix, kmer, is_revcomp)
    return RaggedData(data, offsets)


@njit
def precision_recall_curve(classification, scores):
    """Compute precision-recall curve (JIT-compiled)."""
    if len(scores) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])

    # Get indices for sorting scores in descending order
    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]

    # Initialize arrays (with +1 buffer for initial point)
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    precision = np.zeros(max_size)
    recall = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)

    # Initial point: (recall=0, precision=1, threshold=inf)
    precision[0] = 1.0
    recall[0] = 0.0
    uniq_scores[0] = np.inf

    TP, FP = 0, 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)

    true_false_ratio = 1.0 if number_of_false == 0 else number_of_true / number_of_false

    position = 1
    score = sorted_scores[0]

    for i in range(len(scores)):
        _score = sorted_scores[i]
        _flag = sorted_classification[i]

        # Update TP and FP
        if _flag == 1:
            TP += 1
        else:
            FP += 1

        # Check if score changed
        if i == len(scores) - 1 or score != sorted_scores[i + 1]:
            uniq_scores[position] = _score

            if TP + FP > 0:
                precision[position] = TP / (TP + true_false_ratio * FP)
            else:
                precision[position] = 1.0

            if number_of_true > 0:
                recall[position] = TP / number_of_true
            else:
                recall[position] = 0.0

            position += 1
            if i < len(scores) - 1:
                score = sorted_scores[i + 1]

    return precision[:position], recall[:position], uniq_scores[:position]


@njit
def roc_curve(classification, scores):
    """Compute ROC curve (JIT-compiled)."""
    if len(scores) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])

    # Get indices for sorting scores in descending order
    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]

    # Initialize arrays
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    tpr = np.zeros(max_size)
    fpr = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)

    # Initial point: (fpr=0, tpr=0, threshold=inf)
    tpr[0] = 0.0
    fpr[0] = 0.0
    uniq_scores[0] = np.inf

    TP, FP = 0, 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    position = 1
    score = sorted_scores[0]

    for i in range(len(scores)):
        _score = sorted_scores[i]
        _flag = sorted_classification[i]

        # Update TP and FP
        if _flag == 1:
            TP += 1
        else:
            FP += 1

        # Check if score changed
        if i == len(scores) - 1 or score != sorted_scores[i + 1]:
            uniq_scores[position] = _score

            if number_of_true > 0:
                tpr[position] = TP / number_of_true
            else:
                tpr[position] = 0.0

            if number_of_false > 0:
                fpr[position] = FP / number_of_false
            else:
                fpr[position] = 0.0

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


def scores_to_empirical_log_tail(ragged_scores: RaggedData) -> RaggedData:
    """Convert scores to empirical log-tail values within the current sample."""
    table = build_score_log_tail_table(ragged_scores.data)
    return apply_score_log_tail_table(ragged_scores, table)


@njit(_PROFILE_SUPPORT_SIGNATURE, cache=True)
def _build_profile_support_numba(data, offsets, min_value):
    """Build sparse local-position support for one profile at the requested threshold."""
    n_seq = len(offsets) - 1
    active_ptr = np.zeros(n_seq + 1, dtype=np.int64)

    total_active = np.int64(0)
    for i in range(n_seq):
        start = offsets[i]
        end = offsets[i + 1]
        count = np.int64(0)
        for j in range(end - start):
            if data[start + j] >= min_value:
                count += 1
        total_active += count
        active_ptr[i + 1] = total_active

    active_idx = np.empty(total_active, dtype=np.int64)
    for i in range(n_seq):
        start = offsets[i]
        end = offsets[i + 1]
        write_pos = active_ptr[i]
        for j in range(end - start):
            if data[start + j] >= min_value:
                active_idx[write_pos] = j
                write_pos += 1

    return active_idx, active_ptr


@njit(inline="always", cache=True)
def _lower_bound_int64(arr, left, right, value):
    """Return the first index in arr[left:right] with arr[idx] >= value."""
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@njit(inline="always", cache=True)
def _accumulate_overlap_value(inters, sum1s, sum2s, offset_idx, v1, v2):
    """Accumulate one aligned value pair into overlap statistics."""
    sum1s[offset_idx] += v1
    sum2s[offset_idx] += v2
    inters[offset_idx] += v1 if v1 < v2 else v2


@njit(inline="always", cache=True)
def _accumulate_dense_shift_numba(data1, base1, data2, base2, overlap, min_value, offset_idx, inters, sum1s, sum2s):
    """Accumulate dense overlap statistics for one shift."""
    thresholded = min_value > np.float32(0.0)
    for pos in range(overlap):
        v1 = data1[base1 + pos]
        v2 = data2[base2 + pos]
        if thresholded and v1 < min_value and v2 < min_value:
            continue
        _accumulate_overlap_value(inters, sum1s, sum2s, offset_idx, v1, v2)


@njit(inline="always", cache=True)
def _accumulate_sparse_shift_numba(
    data1,
    start1,
    active_idx1,
    a_lo,
    a_hi,
    data2,
    start2,
    active_idx2,
    b_lo,
    b_hi,
    shift,
    offset_idx,
    inters,
    sum1s,
    sum2s,
    sentinel,
    negative_shift,
):
    """Accumulate sparse overlap statistics for one shift direction."""
    ia = a_lo
    ib = b_lo
    while ia < a_hi or ib < b_hi:
        pos_a = active_idx1[ia] if ia < a_hi else sentinel
        if ib < b_hi:
            raw_b = active_idx2[ib]
            pos_b = raw_b - shift if negative_shift else raw_b + shift
        else:
            pos_b = sentinel

        if pos_a < pos_b:
            pos = pos_a
            ia += 1
        elif pos_b < pos_a:
            pos = pos_b
            ib += 1
        else:
            pos = pos_a
            ia += 1
            ib += 1

        paired_pos = pos + shift if negative_shift else pos - shift
        _accumulate_overlap_value(
            inters,
            sum1s,
            sum2s,
            offset_idx,
            data1[start1 + pos],
            data2[start2 + paired_pos],
        )


@njit(_PROFILE_SPARSE_STATS_SIGNATURE, fastmath=True, cache=True)
def _accumulate_profile_stats_sparse_numba(
    data1,
    offsets1,
    active_idx1,
    active_ptr1,
    data2,
    offsets2,
    active_idx2,
    active_ptr2,
    search_range,
):
    """Accumulate exact overlap statistics using sparse threshold support."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    center = search_range
    inters = np.zeros(n_offsets, dtype=np.float32)
    sum1s = np.zeros(n_offsets, dtype=np.float32)
    sum2s = np.zeros(n_offsets, dtype=np.float32)
    sentinel = np.int64(1 << 60)

    for i in range(n_seq):
        start1 = offsets1[i]
        end1 = offsets1[i + 1]
        start2 = offsets2[i]
        end2 = offsets2[i + 1]
        vlen1 = end1 - start1
        vlen2 = end2 - start2

        a_begin = active_ptr1[i]
        a_end = active_ptr1[i + 1]
        b_begin = active_ptr2[i]
        b_end = active_ptr2[i + 1]

        for neg_shift in range(1, search_range + 1):
            offset_idx = center - neg_shift
            overlap = min(vlen1, vlen2 - neg_shift)
            if overlap <= 0:
                continue

            a_lo = a_begin
            a_hi = _lower_bound_int64(active_idx1, a_lo, a_end, overlap)
            b_lo = _lower_bound_int64(active_idx2, b_begin, b_end, neg_shift)
            b_hi = _lower_bound_int64(active_idx2, b_lo, b_end, neg_shift + overlap)
            _accumulate_sparse_shift_numba(
                data1,
                start1,
                active_idx1,
                a_lo,
                a_hi,
                data2,
                start2,
                active_idx2,
                b_lo,
                b_hi,
                neg_shift,
                offset_idx,
                inters,
                sum1s,
                sum2s,
                sentinel,
                True,
            )

        for pos_shift in range(search_range + 1):
            offset_idx = center + pos_shift
            overlap = min(vlen1 - pos_shift, vlen2)
            if overlap <= 0:
                continue

            a_lo = _lower_bound_int64(active_idx1, a_begin, a_end, pos_shift)
            a_hi = _lower_bound_int64(active_idx1, a_lo, a_end, pos_shift + overlap)
            b_lo = b_begin
            b_hi = _lower_bound_int64(active_idx2, b_lo, b_end, overlap)
            _accumulate_sparse_shift_numba(
                data1,
                start1,
                active_idx1,
                a_lo,
                a_hi,
                data2,
                start2,
                active_idx2,
                b_lo,
                b_hi,
                pos_shift,
                offset_idx,
                inters,
                sum1s,
                sum2s,
                sentinel,
                False,
            )

    return inters, sum1s, sum2s


@njit(_PROFILE_STATS_THRESHOLD_SIGNATURE, fastmath=True, cache=True)
def _accumulate_profile_stats_dense_numba(data1, offsets1, data2, offsets2, search_range, min_value):
    """Accumulate overlap statistics for all offsets with optional threshold masking."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    center = search_range
    inters = np.zeros(n_offsets, dtype=np.float32)
    sum1s = np.zeros(n_offsets, dtype=np.float32)
    sum2s = np.zeros(n_offsets, dtype=np.float32)

    for i in range(n_seq):
        start1 = offsets1[i]
        end1 = offsets1[i + 1]
        start2 = offsets2[i]
        end2 = offsets2[i + 1]
        vlen1 = end1 - start1
        vlen2 = end2 - start2

        for neg_shift in range(1, search_range + 1):
            offset_idx = center - neg_shift
            overlap = min(vlen1, vlen2 - neg_shift)
            if overlap <= 0:
                continue
            base1 = start1
            base2 = start2 + neg_shift
            _accumulate_dense_shift_numba(
                data1,
                base1,
                data2,
                base2,
                overlap,
                min_value,
                offset_idx,
                inters,
                sum1s,
                sum2s,
            )

        for pos_shift in range(search_range + 1):
            offset_idx = center + pos_shift
            overlap = min(vlen1 - pos_shift, vlen2)
            if overlap <= 0:
                continue
            base1 = start1 + pos_shift
            base2 = start2
            _accumulate_dense_shift_numba(
                data1,
                base1,
                data2,
                base2,
                overlap,
                min_value,
                offset_idx,
                inters,
                sum1s,
                sum2s,
            )

    return inters, sum1s, sum2s


@njit(_PROFILE_REDUCER_SIGNATURE, fastmath=True, cache=True)
def _best_profile_score_co_numba(inters, sum1s, sum2s, search_range):
    """Reduce accumulated statistics to the best overlap coefficient score."""
    best_score = np.float32(-1.0)
    best_offset = np.int64(0)

    for k in range(inters.size):
        denom = sum1s[k] if sum1s[k] < sum2s[k] else sum2s[k]
        if denom <= PROFILE_EPS:
            continue
        score = inters[k] / denom
        if score > best_score:
            best_score = score
            best_offset = k - search_range

    if best_score < 0.0:
        return np.float32(0.0), np.int64(0)

    return best_score, best_offset


@njit(_PROFILE_REDUCER_SIGNATURE, fastmath=True, cache=True)
def _best_profile_score_dice_numba(inters, sum1s, sum2s, search_range):
    """Reduce accumulated statistics to the best Dice score."""
    best_score = np.float32(-1.0)
    best_offset = np.int64(0)

    for k in range(inters.size):
        denom = sum1s[k] + sum2s[k]
        if denom <= PROFILE_EPS:
            continue
        score = (np.float32(2.0) * inters[k]) / denom
        if score > best_score:
            best_score = score
            best_offset = k - search_range

    if best_score < 0.0:
        return np.float32(0.0), np.int64(0)

    return best_score, best_offset


def _normalize_profile_score_inputs(data, offsets):
    """Return profile-score inputs as contiguous float32 and int64 arrays."""
    return (
        np.ascontiguousarray(data, dtype=PROFILE_DATA_DTYPE),
        np.ascontiguousarray(offsets, dtype=PROFILE_OFFSETS_DTYPE),
    )


def _normalize_profile_support_inputs(active_idx, active_ptr):
    """Return sparse profile support as contiguous int64 arrays."""
    return (
        np.ascontiguousarray(active_idx, dtype=PROFILE_OFFSETS_DTYPE),
        np.ascontiguousarray(active_ptr, dtype=PROFILE_OFFSETS_DTYPE),
    )


def _unpack_profile_input(profile):
    """Return profile arrays from RaggedData or a raw ``(data, offsets)`` pair."""
    if isinstance(profile, RaggedData):
        return profile.data, profile.offsets
    return profile


def _normalize_support_pair(support):
    """Return one optional support pair as contiguous arrays."""
    if support is None:
        return None
    return _normalize_profile_support_inputs(*support)


def build_profile_support(data, offsets, min_value):
    """Build sparse support for profile positions that meet the threshold."""
    data, offsets = _normalize_profile_score_inputs(data, offsets)
    min_value = np.float32(min_value)
    return _build_profile_support_numba(data, offsets, min_value)


def fast_profile_score(
    profile1,
    profile2,
    options: ProfileScoreOptions,
    supports=None,
):
    """Dispatch profile similarity scoring to specialized overlap-based kernels."""
    data1, offsets1 = _normalize_profile_score_inputs(*_unpack_profile_input(profile1))
    data2, offsets2 = _normalize_profile_score_inputs(*_unpack_profile_input(profile2))
    search_range = np.int64(options.search_range)
    min_value = np.float32(options.min_value)
    thresholded = min_value > np.float32(0.0)
    support1, support2 = (None, None) if supports is None else supports
    normalized_support1 = _normalize_support_pair(support1)
    normalized_support2 = _normalize_support_pair(support2)

    use_sparse = normalized_support1 is not None and normalized_support2 is not None
    if thresholded and not use_sparse:
        normalized_support1 = build_profile_support(data1, offsets1, min_value)
        normalized_support2 = build_profile_support(data2, offsets2, min_value)
        use_sparse = True

    if use_sparse:
        inters, sum1s, sum2s = _accumulate_profile_stats_sparse_numba(
            data1,
            offsets1,
            normalized_support1[0],
            normalized_support1[1],
            data2,
            offsets2,
            normalized_support2[0],
            normalized_support2[1],
            search_range,
        )
    else:
        inters, sum1s, sum2s = _accumulate_profile_stats_dense_numba(
            data1,
            offsets1,
            data2,
            offsets2,
            search_range,
            min_value,
        )

    if options.metric == "co":
        return _best_profile_score_co_numba(inters, sum1s, sum2s, search_range)
    if options.metric == "dice":
        return _best_profile_score_dice_numba(inters, sum1s, sum2s, search_range)

    raise ValueError("metric must be one of: 'co', 'dice'")


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
