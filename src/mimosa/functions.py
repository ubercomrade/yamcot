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

    if number_of_false == 0:
        true_false_ratio = 1.0
    else:
        true_false_ratio = number_of_true / number_of_false

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


@njit(parallel=True, fastmath=True, cache=True)
def _fast_profile_score_co_no_threshold_numba(data1, offsets1, data2, offsets2, search_range):
    """Compute overlap coefficient profile similarity scores without threshold masking."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    scores = np.empty(n_offsets, dtype=np.float32)
    scores[:] = np.float32(-1.0)
    eps = np.float32(1e-6)

    for k in prange(n_offsets):
        offset = k - search_range
        inter = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)

        for i in range(n_seq):
            start1 = offsets1[i]
            end1 = offsets1[i + 1]
            start2 = offsets2[i]
            end2 = offsets2[i + 1]
            vlen1 = end1 - start1
            vlen2 = end2 - start2

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0
            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            base1 = start1 + idx1_start
            base2 = start2 + idx2_start
            for j in range(overlap):
                v1 = data1[base1 + j]
                v2 = data2[base2 + j]
                sum1 += v1
                sum2 += v2
                inter += v1 if v1 < v2 else v2

        denom = sum1 if sum1 < sum2 else sum2
        if denom > eps:
            scores[k] = inter / denom

    return scores


@njit(parallel=True, fastmath=True, cache=True)
def _fast_profile_score_co_threshold_numba(data1, offsets1, data2, offsets2, search_range, min_value):
    """Compute overlap coefficient profile similarity scores with threshold masking."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    scores = np.empty(n_offsets, dtype=np.float32)
    scores[:] = np.float32(-1.0)
    eps = np.float32(1e-6)

    for k in prange(n_offsets):
        offset = k - search_range
        inter = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)

        for i in range(n_seq):
            start1 = offsets1[i]
            end1 = offsets1[i + 1]
            start2 = offsets2[i]
            end2 = offsets2[i + 1]
            vlen1 = end1 - start1
            vlen2 = end2 - start2

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0
            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            base1 = start1 + idx1_start
            base2 = start2 + idx2_start
            for j in range(overlap):
                v1 = data1[base1 + j]
                v2 = data2[base2 + j]
                if v1 < min_value and v2 < min_value:
                    continue
                sum1 += v1
                sum2 += v2
                inter += v1 if v1 < v2 else v2

        denom = sum1 if sum1 < sum2 else sum2
        if denom > eps:
            scores[k] = inter / denom

    return scores


@njit(parallel=True, fastmath=True, cache=True)
def _fast_profile_score_dice_no_threshold_numba(data1, offsets1, data2, offsets2, search_range):
    """Compute Dice profile similarity scores without threshold masking."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    scores = np.empty(n_offsets, dtype=np.float32)
    scores[:] = np.float32(-1.0)
    eps = np.float32(1e-6)

    for k in prange(n_offsets):
        offset = k - search_range
        inter = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)

        for i in range(n_seq):
            start1 = offsets1[i]
            end1 = offsets1[i + 1]
            start2 = offsets2[i]
            end2 = offsets2[i + 1]
            vlen1 = end1 - start1
            vlen2 = end2 - start2

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0
            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            base1 = start1 + idx1_start
            base2 = start2 + idx2_start
            for j in range(overlap):
                v1 = data1[base1 + j]
                v2 = data2[base2 + j]
                sum1 += v1
                sum2 += v2
                inter += v1 if v1 < v2 else v2

        denom = sum1 + sum2
        if denom > eps:
            scores[k] = (np.float32(2.0) * inter) / denom

    return scores


@njit(parallel=True, fastmath=True, cache=True)
def _fast_profile_score_dice_threshold_numba(data1, offsets1, data2, offsets2, search_range, min_value):
    """Compute Dice profile similarity scores with threshold masking."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    scores = np.empty(n_offsets, dtype=np.float32)
    scores[:] = np.float32(-1.0)
    eps = np.float32(1e-6)

    for k in prange(n_offsets):
        offset = k - search_range
        inter = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)

        for i in range(n_seq):
            start1 = offsets1[i]
            end1 = offsets1[i + 1]
            start2 = offsets2[i]
            end2 = offsets2[i + 1]
            vlen1 = end1 - start1
            vlen2 = end2 - start2

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0
            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            base1 = start1 + idx1_start
            base2 = start2 + idx2_start
            for j in range(overlap):
                v1 = data1[base1 + j]
                v2 = data2[base2 + j]
                if v1 < min_value and v2 < min_value:
                    continue
                sum1 += v1
                sum2 += v2
                inter += v1 if v1 < v2 else v2

        denom = sum1 + sum2
        if denom > eps:
            scores[k] = (np.float32(2.0) * inter) / denom

    return scores


@njit(cache=True)
def _pick_best_profile_score(scores, search_range):
    """Select the best non-negative score and convert it back to an offset."""
    best_score = np.float32(-1.0)
    best_offset = 0

    for k in range(scores.size):
        score = scores[k]
        if score >= 0.0 and score > best_score:
            best_score = score
            best_offset = k - search_range

    if best_score < 0.0:
        return np.float32(0.0), 0

    return best_score, best_offset


def fast_profile_score(data1, offsets1, data2, offsets2, search_range, min_value=0.0, metric="co"):
    """Dispatch profile similarity scoring to specialized overlap-based kernels."""
    thresholded = min_value > 0.0

    if metric == "co":
        scores = (
            _fast_profile_score_co_threshold_numba(data1, offsets1, data2, offsets2, search_range, min_value)
            if thresholded
            else _fast_profile_score_co_no_threshold_numba(data1, offsets1, data2, offsets2, search_range)
        )
        return _pick_best_profile_score(scores, search_range)

    if metric == "dice":
        scores = (
            _fast_profile_score_dice_threshold_numba(data1, offsets1, data2, offsets2, search_range, min_value)
            if thresholded
            else _fast_profile_score_dice_no_threshold_numba(data1, offsets1, data2, offsets2, search_range)
        )
        return _pick_best_profile_score(scores, search_range)

    raise ValueError("metric must be one of: 'co', 'dice'")


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
