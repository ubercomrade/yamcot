import numpy as np
from numba import njit, prange

from mimosa.ragged import RaggedData


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


@njit(inline="always")
def _fill_rc_buffer(data, start, length, buffer):
    """Fill a buffer with reverse-complement values without allocations."""
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    for j in range(length):
        val = data[start + length - 1 - j]
        buffer[j] = rc_table[val]


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
        out_start = new_offsets[i]
        n_scores = new_offsets[i + 1] - out_start

        if n_scores > 0:
            site_buffer = np.empty(m, dtype=data.dtype)

            for k in range(n_scores):
                if not is_revcomp:
                    num_site = data[start + k : start + k + m]
                    results[out_start + k] = score_seq(num_site, kmer, matrix)
                else:
                    _fill_rc_buffer(data, start + k, m, site_buffer)
                    results[out_start + k] = score_seq(site_buffer, kmer, matrix)

    return results, new_offsets


@njit(parallel=True, fastmath=True, cache=True)
def _batch_all_scores_with_context_jit(data, offsets, matrix, kmer, is_revcomp):
    """Compute BaMM scores in batch with context-aware padding."""
    n_seq = len(offsets) - 1
    m = matrix.shape[-1]
    context_len = kmer - 1
    window_size = m + context_len
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)

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

        site_buffer = np.full(window_size, 4, dtype=data.dtype)

        if n_scores > 0:
            for k in range(n_scores):
                site_buffer[:] = 4

                if not is_revcomp:
                    s_idx = k - context_len
                    e_idx = k + m

                    actual_start = max(0, s_idx)
                    actual_end = min(seq_len, e_idx)
                    dest_start = max(0, -s_idx)

                    copy_len = actual_end - actual_start
                    if copy_len > 0:
                        site_buffer[dest_start : dest_start + copy_len] = data[
                            start + actual_start : start + actual_end
                        ]
                else:
                    r_start = k

                    for t in range(window_size):
                        data_idx = start + r_start + (window_size - 1 - t)
                        if start <= data_idx < start + seq_len:
                            site_buffer[t] = rc_table[data[data_idx]]

                results[out_start + k] = score_seq(site_buffer, kmer, matrix)

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
    """Convert RaggedData containing scores to frequency representation."""
    flat = ragged_scores.data
    n = flat.size

    if n == 0:
        return RaggedData(np.zeros(0, dtype=np.float32), ragged_scores.offsets.copy())

    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    surv = np.cumsum(cnt[::-1])[::-1]

    eps = 1e-12
    log_p = np.log10(n + eps) - np.log10(surv + eps)

    new_data = log_p[inv].astype(np.float32)
    return RaggedData(new_data, ragged_scores.offsets.copy())


@njit(fastmath=True, cache=True)
def _fast_overlap_kernel_numba(data1, offsets1, data2, offsets2, search_range):
    """Fast overlap coefficient kernel for RaggedData using JIT compilation."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1

    inters = np.zeros(n_offsets, dtype=np.float32)
    sum1s = np.zeros(n_offsets, dtype=np.float32)
    sum2s = np.zeros(n_offsets, dtype=np.float32)

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
            local_s1 = np.float32(0.0)
            local_s2 = np.float32(0.0)

            for j in range(overlap):
                v1 = s1[idx1_start + j]
                v2 = s2[idx2_start + j]
                local_s1 += v1
                local_s2 += v2

                local_inter += np.float32(0.5) * (v1 + v2 - abs(v1 - v2))

            inters[k] += local_inter
            sum1s[k] += local_s1
            sum2s[k] += local_s2

    best = -1.0
    best_offset = 0
    eps = 1e-6

    for k in range(n_offsets):
        denom = min(sum1s[k], sum2s[k])
        if denom > eps:
            val = inters[k] / denom
            if val > best:
                best = val
                best_offset = k - search_range

    return best, best_offset


@njit(fastmath=True, cache=True)
def _fast_cj_kernel_numba(data1, offsets1, data2, offsets2, search_range):
    """Compute Continuous Jaccard scores over RaggedData with JIT."""
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1

    sums = np.zeros(n_offsets, dtype=np.float32)
    diffs = np.zeros(n_offsets, dtype=np.float32)

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

            local_sum = np.float32(0.0)
            local_diff = np.float32(0.0)

            for j in range(overlap):
                v1 = s1[idx1_start + j]
                v2 = s2[idx2_start + j]
                local_sum += v1 + v2
                local_diff += abs(v1 - v2)

            sums[k] += local_sum
            diffs[k] += local_diff

    best_cj = -1.0
    best_offset = 0
    eps = 1e-6

    for k in range(n_offsets):
        S = sums[k]
        D = diffs[k]
        denom = S + D
        if denom > eps:
            cj = (S - D) / denom
            if cj > best_cj:
                best_cj = cj
                best_offset = k - search_range

    return best_cj, best_offset


@njit(fastmath=True, cache=True)
def _fast_pearson_kernel(data1, offsets1, data2, offsets2, search_range):
    """Pearson correlation kernel for RaggedData with numba-compatible math."""

    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1

    best_corr = -2.0
    best_offset = 0
    found_valid = False

    for k in range(n_offsets):
        offset = k - search_range

        n = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0

        for i in range(n_seq):
            s1 = data1[offsets1[i] : offsets1[i + 1]]
            s2 = data2[offsets2[i] : offsets2[i + 1]]
            vlen1 = s1.size
            vlen2 = s2.size

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0

            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            for j in range(overlap):
                x = s1[idx1_start + j]
                y = s2[idx2_start + j]
                n += 1
                sum_x += x
                sum_y += y
                sum_xx += x * x
                sum_yy += y * y
                sum_xy += x * y

        if n > 1:
            mean_x = sum_x / n
            mean_y = sum_y / n
            var_x = sum_xx / n - mean_x * mean_x
            var_y = sum_yy / n - mean_y * mean_y

            if var_x > 1e-10 and var_y > 1e-10:
                cov = sum_xy / n - mean_x * mean_y
                corr_val = cov / np.sqrt(var_x * var_y)

                if corr_val > best_corr:
                    best_corr = corr_val
                    best_offset = offset
                found_valid = True

    if not found_valid:
        best_corr = 0.0

    return best_corr, best_offset


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
