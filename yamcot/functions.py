import numpy as np
from numba import njit, prange
from scipy.stats import pearsonr

from .ragged import RaggedData

RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
BACKGROUND_FREQ = 0.25  # Background frequency for PWM calculation
PFM_TO_PWM_PSEUDOCOUNT = 0.0001  # Pseudocount added to PFM values
PCM_TO_PFM_NUCLEOTIDE_PSEUDOCOUNT = 0.25  # Pseudocount for nucleotide frequency
PCM_TO_PFM_DENOMINATOR_CONSTANT = 1  # Constant added to denominator in PCM to PFM conversion


def pfm_to_pwm(pfm):
    """
    Convert Position Frequency Matrix to Position Weight Matrix.
    
    Parameters
    ----------
    pfm : np.ndarray
        Position Frequency Matrix of shape (4, L) where L is the motif length.
        
    Returns
    -------
    np.ndarray
        Position Weight Matrix computed as log(PFM + pseudo_count) / background.
    """
    background = BACKGROUND_FREQ
    pwm = np.log((pfm + PFM_TO_PWM_PSEUDOCOUNT) / background)
    return pwm


def pcm_to_pfm(pcm):
    """
    Convert Position Count Matrix to Position Frequency Matrix.
    
    Parameters
    ----------
    pcm : np.ndarray
        Position Count Matrix of shape (4, L) where L is the motif length.
        
    Returns
    -------
    np.ndarray
        Position Frequency Matrix with pseudo-counts added.
    """
    number_of_sites = pcm.sum(axis=0)
    nuc_pseudo = PCM_TO_PFM_NUCLEOTIDE_PSEUDOCOUNT
    pfm = (pcm + nuc_pseudo) / (number_of_sites + PCM_TO_PFM_DENOMINATOR_CONSTANT)
    return pfm


@njit
def score_seq(num_site, kmer, model):
    """
    Compute score for a sequence site using a k-mer model.
    
    Parameters
    ----------
    num_site : np.ndarray
        Numerical representation of the DNA sequence site.
    kmer : int
        Length of the k-mer used for indexing.
    model : np.ndarray
        Scoring model matrix.
        
    Returns
    -------
    float
        Computed score for the sequence site.
    """
    score = 0.0
    seq_len = num_site.shape[0]
    for i in range(seq_len - kmer + 1):
        score_idx = 0
        for j in range(kmer):
            score_idx = score_idx * 5 + num_site[i + j]  # Convert to single index
        score += model.flat[score_idx * model.shape[-1] + i]  # Access via flat index

    return score


@njit(parallel=False)
def all_scores(num_seq, model, kmer):
    """
    Standard scanning function for fixed-width motifs (PWM, simple K-mer).
    Model width matches the scoring window.
    """
    length_of_site = model.shape[-1]
    number_of_scores = num_seq.shape[0] - length_of_site + 1
    scores = np.zeros((2, number_of_scores), dtype=np.float32)

    for i in range(number_of_scores):
        # 1. Forward
        num_site = num_seq[i : i + length_of_site]
        scores[0, i] = score_seq(num_site, kmer, model)

        # 2. Reverse Complement
        # Reverse view + Lookup
        rc_site = RC_TABLE[num_site[::-1]]
        scores[1, i] = score_seq(rc_site, kmer, model)

    return scores


#@njit(parallel=True)
def _batch_all_scores_jit(data, offsets, matrix, kmer, is_revcomp, is_bamm=False):
    """
    Universal JIT kernel for PWM and BaMM models in batch mode.
    For BaMM, extended window and 'N' padding are used.
    
    Parameters
    ----------
    data : np.ndarray
        Flattened sequence data array.
    offsets : np.ndarray
        Offsets indicating sequence boundaries in the data array.
    matrix : np.ndarray
        Scoring matrix for motif evaluation.
    kmer : int
        K-mer length parameter for scoring.
    is_revcomp : bool
        Whether to consider reverse complement strand.
    is_bamm : bool, optional
        Whether to use BaMM-specific scoring (default is False).
        
    Returns
    -------
    tuple
        Tuple containing the results array and new offsets array.
    """
    n_seq = len(offsets) - 1
    m = matrix.shape[-1]
    context_len = kmer - 1 if is_bamm else 0
    window_size = m + context_len

    # 1. Calculate new offsets (still N - L + 1)
    new_offsets = np.zeros(n_seq + 1, dtype=np.int64)
    for i in range(n_seq):
        seq_len = offsets[i+1] - offsets[i]
        if seq_len >= m:
            new_offsets[i+1] = seq_len - m + 1
    
    # Cumulative sum
    for i in range(n_seq):
        new_offsets[i+1] += new_offsets[i]
        
    total_scores = new_offsets[n_seq]
    results = np.zeros(total_scores, dtype=np.float32)
    
    # 2. Calculate scores
    for i in prange(n_seq):
        start = offsets[i]
        seq_len = offsets[i+1] - start
        out_start = new_offsets[i]
        n_scores = new_offsets[i+1] - out_start
        
        if n_scores > 0:
            for k in range(n_scores):
                if not is_bamm:
                    # Standard PWM path
                    num_site = data[start + k : start + k + m]
                    if is_revcomp:
                        num_site = RC_TABLE[num_site[::-1]]
                else:
                    # BaMM path with context and padding
                    if not is_revcomp:
                        # Forward: [k - context : k + L]
                        s_idx = k - context_len
                        e_idx = k + m
                        num_site = np.full(window_size, 4, dtype=data.dtype)
                        
                        actual_start = max(0, s_idx)
                        actual_end = min(seq_len, e_idx)
                        dest_start = max(0, -s_idx)
                        cloop = actual_end - actual_start
                        if cloop > 0:
                            num_site[dest_start : dest_start + cloop] = data[start + actual_start : start + actual_end]
                    else:
                        # RC: [k : k + L + context]
                        r_start = k
                        r_end = k + window_size
                        raw_segment = np.full(window_size, 4, dtype=data.dtype)
                        
                        actual_end = min(seq_len, r_end)
                        cloop = actual_end - r_start
                        if cloop > 0:
                            raw_segment[:cloop] = data[start + r_start : start + actual_end]
                        num_site = RC_TABLE[raw_segment[::-1]]

                results[out_start + k] = score_seq(num_site, kmer, matrix)
                
    return results, new_offsets


@njit(parallel=True)
def batch_best_scores_jit(data, offsets, matrix, kmer, is_revcomp, both_strands, is_bamm=False):
    """
    Find the best score for each sequence in RaggedData.
    
    Parameters
    ----------
    data : np.ndarray
        Flattened sequence data array.
    offsets : np.ndarray
        Offsets indicating sequence boundaries in the data array.
    matrix : np.ndarray
        Scoring matrix for motif evaluation.
    kmer : int
        K-mer length parameter for scoring.
    is_revcomp : bool
        Whether to consider reverse complement strand.
    both_strands : bool
        Whether to evaluate both forward and reverse complement strands.
    is_bamm : bool, optional
        Whether to use BaMM-specific scoring (default is False).
        
    Returns
    -------
    np.ndarray
        Array containing the best score for each sequence.
    """
    n_seq = len(offsets) - 1
    m = matrix.shape[-1]
    context_len = kmer - 1 if is_bamm else 0
    window_size = m + context_len
    best_results = np.full(n_seq, -1e9, dtype=np.float32)

    for i in prange(n_seq):
        start = offsets[i]
        seq_len = offsets[i+1] - start
        
        if seq_len < m:
            continue

        n_scores = seq_len - m + 1
        current_best = -1e9

        for k in range(n_scores):
            # Forward strand
            if not is_revcomp or both_strands:
                if not is_bamm:
                    num_site = data[start + k : start + k + m]
                else:
                    s_idx = k - context_len
                    e_idx = k + m
                    num_site = np.full(window_size, 4, dtype=data.dtype)
                    actual_start = max(0, s_idx)
                    actual_end = min(seq_len, e_idx)
                    dest_start = max(0, -s_idx)
                    cloop = actual_end - actual_start
                    if cloop > 0:
                        num_site[dest_start : dest_start + cloop] = data[start + actual_start : start + actual_end]
                
                s_fwd = score_seq(num_site, kmer, matrix)
                if s_fwd > current_best:
                    current_best = s_fwd

            # Reverse strand
            if is_revcomp or both_strands:
                if not is_bamm:
                    num_site = data[start + k : start + k + m]
                    rc_site = RC_TABLE[num_site[::-1]]
                else:
                    r_start = k
                    r_end = k + window_size
                    raw_segment = np.full(window_size, 4, dtype=data.dtype)
                    actual_end = min(seq_len, r_end)
                    cloop = actual_end - r_start
                    if cloop > 0:
                        raw_segment[:cloop] = data[start + r_start : start + actual_end]
                    rc_site = RC_TABLE[raw_segment[::-1]]

                s_rev = score_seq(rc_site, kmer, matrix)
                if s_rev > current_best:
                    current_best = s_rev
        
        best_results[i] = current_best
            
    return best_results


def batch_all_scores(
    sequences: RaggedData,
    matrix: np.ndarray,
    kmer: int = 1,
    is_revcomp: bool = False,
    is_bamm: bool = False
) -> RaggedData:
    """
    Compute scores for all sequences in RaggedData.
    Supports both PWM (is_bamm=False) and BaMM models.
    
    Parameters
    ----------
    sequences : RaggedData
        Input sequences in RaggedData format.
    matrix : np.ndarray
        Scoring matrix for motif evaluation.
    kmer : int, optional
        K-mer length parameter for scoring (default is 1).
    is_revcomp : bool, optional
        Whether to consider reverse complement strand (default is False).
    is_bamm : bool, optional
        Whether to use BaMM-specific scoring (default is False).
        
    Returns
    -------
    RaggedData
        RaggedData object containing computed scores.
    """
    data, offsets = _batch_all_scores_jit(sequences.data, sequences.offsets, matrix, kmer, is_revcomp, is_bamm=is_bamm)
    return RaggedData(data, offsets)


def batch_best_scores(
    sequences: RaggedData,
    matrix: np.ndarray,
    kmer: int = 1,
    is_revcomp: bool = False,
    both_strands: bool = False,
    is_bamm: bool = False
) -> np.ndarray:
    """
    Return best scores for each sequence.
    
    Parameters
    ----------
    sequences : RaggedData
        Input sequences in RaggedData format.
    matrix : np.ndarray
        Scoring matrix for motif evaluation.
    kmer : int, optional
        K-mer length parameter for scoring (default is 1).
    is_revcomp : bool, optional
        Whether to consider reverse complement strand (default is False).
    both_strands : bool, optional
        Whether to evaluate both forward and reverse complement strands (default is False).
    is_bamm : bool, optional
        Whether to use BaMM-specific scoring (default is False).
        
    Returns
    -------
    np.ndarray
        Array containing the best score for each sequence.
    """
    return batch_best_scores_jit(
        sequences.data, sequences.offsets, matrix, kmer, is_revcomp, both_strands, is_bamm=is_bamm
    )


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
    """
    Truncate ROC curve at a specific score threshold.
    
    This function truncates the ROC curve (True Positive Rate vs False Positive Rate)
    at a given score threshold. If interpolation is needed between points, it
    performs linear interpolation to determine the TPR and FPR values at the exact
    score cutoff.
    
    Parameters
    ----------
    tpr : np.ndarray
        True Positive Rate values from the ROC curve.
    fpr : np.ndarray
        False Positive Rate values from the ROC curve.
    thr : np.ndarray
        Threshold values corresponding to each TPR/FPR pair.
    score_cutoff : float
        The score threshold at which to truncate the ROC curve.
    
    Returns
    -------
    tuple
        Tuple containing (truncated_tpr, truncated_fpr, truncated_thresholds).
    """
    if score_cutoff == -np.inf:
        return tpr, fpr, thr

    # thr starts with inf, then decreases
    mask = thr >= score_cutoff
    if not np.any(mask):
        return (
            np.array([tpr[0]], dtype=tpr.dtype),
            np.array([0.0], dtype=fpr.dtype),
            np.array([score_cutoff], dtype=thr.dtype)
        )

    last = int(np.where(mask)[0][-1])

    if thr[last] == score_cutoff or last == len(thr) - 1:
        return tpr[: last + 1], fpr[: last + 1], thr[: last + 1]

    # Score interpolation
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
    """
    Truncate Precision-Recall curve at a specific score threshold.
    
    This function truncates the Precision-Recall curve at a given score threshold.
    If interpolation is needed between points, it performs linear interpolation
    to determine the precision and recall values at the exact score cutoff.
    
    Parameters
    ----------
    rec : np.ndarray
        Recall values from the Precision-Recall curve.
    prec : np.ndarray
        Precision values from the Precision-Recall curve.
    thr : np.ndarray
        Threshold values corresponding to each precision/recall pair.
    score_cutoff : float
        The score threshold at which to truncate the PRC.
    
    Returns
    -------
    tuple
        Tuple containing (truncated_recall, truncated_precision, truncated_thresholds).
    """
    if score_cutoff == -np.inf:
        return rec, prec, thr

    # thr starts with inf, then decreases
    # find i: last index where thr[i] >= score_cutoff
    mask = thr >= score_cutoff
    if not np.any(mask):
        # threshold too high -> almost empty
        return (
            np.array([0.0], dtype=rec.dtype),
            np.array([prec[0]], dtype=prec.dtype),
            np.array([score_cutoff], dtype=thr.dtype)
        )

    last = int(np.where(mask)[0][-1])

    # if we hit the node exactly - just truncate
    if thr[last] == score_cutoff or last == len(thr) - 1:
        return rec[: last + 1], prec[: last + 1], thr[: last + 1]

    # otherwise interpolate between last and last+1 by score
    s0, s1 = float(thr[last]), float(thr[last + 1])  # s0 > cutoff > s1
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
    """
    Standardize partial AUC value to range [0.5, 1].
    
    This function standardizes a raw partial AUC value to a range between 0.5 and 1,
    where 0.5 represents random performance and 1 represents perfect performance.
    This standardization accounts for the theoretical minimum and maximum possible
    partial AUC values for the given conditions.
    
    Parameters
    ----------
    pauc_raw : float
        Raw partial AUC value to standardize.
    pauc_min : float
        Minimum possible partial AUC value for the given conditions.
    pauc_max : float
        Maximum possible partial AUC value for the given conditions.
    
    Returns
    -------
    float
        Standardized partial AUC value in range [0.5, 1].
    """
    denom = (pauc_max - pauc_min)
    if denom <= 0:
        return 0.5
    return 0.5 * (1.0 + (pauc_raw - pauc_min) / denom)



def scores_to_frequencies(ragged_scores: RaggedData) -> RaggedData:
    """
    Convert RaggedData containing scores to frequency representation.
    
    This function computes log-frequency transformation of scores where each
    unique score value is replaced by its negative log-frequency across all
    sequences in the RaggedData structure.
    
    Parameters
    ----------
    ragged_scores : RaggedData
        Input RaggedData containing score values.
        
    Returns
    -------
    RaggedData
        RaggedData with transformed frequency values.
    """
    flat = ragged_scores.data
    n = flat.size
    
    if n == 0:
        return RaggedData(np.zeros(0, dtype=np.float32), ragged_scores.offsets.copy())

    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    surv = np.cumsum(cnt[::-1])[::-1]
    
    # To avoid log10(0)
    eps = 1e-12
    log_p = np.log10(n + eps) - np.log10(surv + eps)
    
    new_data = log_p[inv].astype(np.float32)
    return RaggedData(new_data, ragged_scores.offsets.copy())


@njit(fastmath=True, cache=True)
def _fast_overlap_kernel_numba(data1, offsets1, data2, offsets2, search_range):
    """
    Fast overlap coefficient kernel for RaggedData using JIT compilation.
    
    This kernel computes the overlap coefficient (Szymkiewicz-Simpson coefficient)
    between two sets of ragged sequences, finding the best alignment within
    the specified search range.
    
    Parameters
    ----------
    data1 : np.ndarray
        Flattened data array for first sequence collection.
    offsets1 : np.ndarray
        Offsets for first sequence collection.
    data2 : np.ndarray
        Flattened data array for second sequence collection.
    offsets2 : np.ndarray
        Offsets for second sequence collection.
    search_range : int
        Range of offsets to search for best alignment (from -search_range to +search_range).
        
    Returns
    -------
    tuple
        Tuple containing (best_overlap, best_offset) where:
        best_overlap : Maximum overlap coefficient found.
        best_offset : Offset at which maximum overlap occurs.
    """
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    
    inters = np.zeros(n_offsets, dtype=np.float32)
    sum1s  = np.zeros(n_offsets, dtype=np.float32)
    sum2s  = np.zeros(n_offsets, dtype=np.float32)

    for i in range(n_seq):
        s1 = data1[offsets1[i]:offsets1[i+1]]
        s2 = data2[offsets2[i]:offsets2[i+1]]
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
            local_s1    = np.float32(0.0)
            local_s2    = np.float32(0.0)

            for j in range(overlap):
                v1 = s1[idx1_start + j]
                v2 = s2[idx2_start + j]
                local_s1 += v1
                local_s2 += v2
                # min(v1, v2) = 0.5 * (v1 + v2 - |v1 - v2|)
                local_inter += np.float32(0.5) * (v1 + v2 - abs(v1 - v2))

            inters[k] += local_inter
            sum1s[k]  += local_s1
            sum2s[k]  += local_s2

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
    """
    Fast  Continues Jaccard (CJ) coefficient kernel for RaggedData using JIT compilation.
    
    This kernel computes the continues jaccard coefficient between two sets of
    ragged sequences, finding the best alignment within the specified search range.
    
    Parameters
    ----------
    data1 : np.ndarray
        Flattened data array for first sequence collection.
    offsets1 : np.ndarray
        Offsets for first sequence collection.
    data2 : np.ndarray
        Flattened data array for second sequence collection.
    offsets2 : np.ndarray
        Offsets for second sequence collection.
    search_range : int
        Range of offsets to search for best alignment (from -search_range to +search_range).
        
    Returns
    -------
    tuple
        Tuple containing (best_cj, best_offset) where:
        best_cj : Maximum Czekanowski-Dice coefficient found.
        best_offset : Offset at which maximum coefficient occurs.
    """
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    
    sums  = np.zeros(n_offsets, dtype=np.float32)
    diffs = np.zeros(n_offsets, dtype=np.float32)

    for i in range(n_seq):
        s1 = data1[offsets1[i]:offsets1[i+1]]
        s2 = data2[offsets2[i]:offsets2[i+1]]
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
                local_sum += (v1 + v2)
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


def _fast_pearson_kernel(data1, offsets1, data2, offsets2, search_range):
    """Pearson correlation kernel for RaggedData using numpy built-in functions."""
    
    n_seq = len(offsets1) - 1
    n_offsets = 2 * search_range + 1
    
    correlations = np.zeros(n_offsets, dtype=np.float32)
    pvalues = np.ones(n_offsets, dtype=np.float32)
    valid_correlations = np.zeros(n_offsets, dtype=np.bool_)

    for k in range(n_offsets):
        offset = k - search_range
        
        all_x_values = []
        all_y_values = []
        
        for i in range(n_seq):
            s1 = data1[offsets1[i]:offsets1[i+1]]
            s2 = data2[offsets2[i]:offsets2[i+1]]
            vlen1 = s1.size
            vlen2 = s2.size

            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0

            if idx1_start >= vlen1 or idx2_start >= vlen2:
                continue

            overlap = min(vlen1 - idx1_start, vlen2 - idx2_start)
            if overlap <= 0:
                continue

            x_vals = s1[idx1_start:idx1_start + overlap]
            y_vals = s2[idx2_start:idx2_start + overlap]
            
            all_x_values.extend(x_vals)
            all_y_values.extend(y_vals)

        if len(all_x_values) > 1:  # Need at least 2 points for correlation
            x_array = np.array(all_x_values, dtype=np.float64)
            y_array = np.array(all_y_values, dtype=np.float64)
            
            # Check if either array has zero variance
            if np.var(x_array) > 1e-10 and np.var(y_array) > 1e-10:
                corr_val, pvalue = pearsonr(x_array, y_array)
                correlations[k] = corr_val
                pvalues[k] = pvalue
                valid_correlations[k] = True
            else:
                # If one variable has no variance, correlation is undefined (set to 0)
                correlations[k] = 0.0
                pvalues[k] = 1.0
                valid_correlations[k] = True

    # Find the best correlation among valid ones
    best_corr = -2.0  # Pearson correlation ranges from -1 to 1
    best_offset = 0
    found_valid = False
    best_pvalue = 1.0

    for k in range(n_offsets):
        if valid_correlations[k] and correlations[k] > best_corr:
            best_corr = correlations[k]
            best_pvalue = pvalues[k]
            best_offset = k - search_range
            found_valid = True

    if not found_valid:
        best_corr = 0.0 
        best_pvalue = 1.0

    return best_corr, best_pvalue, best_offset


def format_params(params: dict) -> str:
    keys = sorted(params.keys())
    return "_".join(f"{k}-{params[k]}" for k in keys)
