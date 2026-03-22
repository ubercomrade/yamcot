import numpy as np
from numba import njit, prange

from mimosa.ragged import RaggedData

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)


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


@njit(inline="always", cache=True)
def _fill_rc_buffer(data, start, length, buffer):
    """Fill a buffer with reverse-complement values without allocations."""
    for j in range(length):
        val = data[start + length - 1 - j]
        buffer[j] = RC_TABLE[val]


@njit(inline="always", cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
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
    sequences: RaggedData, matrix: np.ndarray, kmer: int = 1, is_revcomp: bool = False, with_context: bool = False
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
    n = len(scores)
    if n == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]

    max_size = n

    precision = np.zeros(max_size)
    recall = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)

    precision[0] = 1.0
    recall[0] = 0.0
    uniq_scores[0] = np.inf

    TP, FP = 0, 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)

    if number_of_false == 0:
        true_false_ratio = 1.0
    else:
        true_false_ratio = number_of_true / number_of_false

    position = 1
    score = sorted_scores[0]

    for i in range(len(scores)):
        _score = sorted_scores[i]
        _flag = sorted_classification[i]

        if _flag == 1:
            TP += 1
        else:
            FP += 1

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
    n = len(scores)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]

    max_size = n + 1

    tpr = np.zeros(max_size)
    fpr = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)

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

        if _flag == 1:
            TP += 1
        else:
            FP += 1

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


def scores_to_frequencies(ragged_scores: RaggedData) -> RaggedData:
    """Convert scores to empirical log-tail frequencies within the current sample."""
    flat = ragged_scores.data
    n = flat.size

    if n == 0:
        return RaggedData(np.zeros(0, dtype=np.float32), ragged_scores.offsets)

    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    surv = np.cumsum(cnt[::-1])[::-1]

    eps = 1e-12
    log_p = np.log10(n + eps) - np.log10(surv + eps)

    new_data = log_p[inv].astype(np.float32)
    return RaggedData(new_data, ragged_scores.offsets)


@njit(inline="always", cache=True)
def _profile_score_from_stats_numba(metric_id, inter, sum1, sum2, valid_count, eps):
    """Compute a profile similarity score from accumulated overlap statistics."""
    if metric_id == 0:
        denom = sum1 if sum1 < sum2 else sum2
        if denom > eps:
            return inter / denom
        return np.float32(-1.0)

    if metric_id == 1:
        denom = sum1 + sum2 - inter
        if denom > eps:
            return inter / denom
        return np.float32(-1.0)

    if metric_id == 2:
        denom = sum1 + sum2
        if denom > eps:
            return (np.float32(2.0) * inter) / denom
        return np.float32(-1.0)

    if valid_count > 0:
        denom = sum1 + sum2
        l1_sum = denom - (np.float32(2.0) * inter)
        mean_l1 = l1_sum / np.float32(valid_count)
        return np.float32(1.0) / (np.float32(1.0) + mean_l1)
    return np.float32(-1.0)


@njit(fastmath=True, cache=True)
def _fast_profile_score_kernel_numba(data1, offsets1, data2, offsets2, search_range, min_value, metric_id):
    """Compute overlap-derived profile similarity scores over RaggedData with JIT."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1

    inters = np.zeros(n_offsets, dtype=np.float32)
    sum1s = np.zeros(n_offsets, dtype=np.float32)
    sum2s = np.zeros(n_offsets, dtype=np.float32)
    valid_counts = np.zeros(n_offsets, dtype=np.int32)

    for i in range(n_seq):
        s1 = data1[offsets1[i] : offsets1[i + 1]]
        s2 = data2[offsets2[i] : offsets2[i + 1]]
        vlen1 = s1.size
        vlen2 = s2.size

        for k in range(n_offsets):
            offset = k - search_range
            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0

            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            local_inter = np.float32(0.0)
            local_sum1 = np.float32(0.0)
            local_sum2 = np.float32(0.0)
            local_count = 0

            for j in range(overlap):
                v1 = s1[idx1_start + j]
                v2 = s2[idx2_start + j]
                if min_value > 0.0 and v1 < min_value and v2 < min_value:
                    continue
                local_sum1 += v1
                local_sum2 += v2
                local_inter += np.float32(0.5) * (v1 + v2 - abs(v1 - v2))
                local_count += 1

            inters[k] += local_inter
            sum1s[k] += local_sum1
            sum2s[k] += local_sum2
            valid_counts[k] += local_count

    best_score = np.float32(-1.0)
    best_offset = 0
    eps = np.float32(1e-6)
    found_valid = False

    for k in range(n_offsets):
        score = _profile_score_from_stats_numba(metric_id, inters[k], sum1s[k], sum2s[k], valid_counts[k], eps)
        if score >= 0.0:
            found_valid = True
            if score > best_score:
                best_score = score
                best_offset = k - search_range

    if not found_valid:
        return np.float32(0.0), 0

    return best_score, best_offset


def _profile_metric_to_id(metric: str) -> int:
    """Map profile metric names to numba-friendly integer identifiers."""
    if metric == "co":
        return 0
    if metric == "cj":
        return 1
    if metric == "dice":
        return 2
    if metric == "l1sim":
        return 3
    raise ValueError("metric must be one of: 'cj', 'co', 'dice', 'l1sim'")


def fast_profile_score(data1, offsets1, data2, offsets2, search_range, min_value=0.0, metric="cj"):
    """Dispatch profile similarity scoring to the shared overlap-based kernel."""
    metric_id = _profile_metric_to_id(metric)
    return _fast_profile_score_kernel_numba(data1, offsets1, data2, offsets2, search_range, min_value, metric_id)


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
