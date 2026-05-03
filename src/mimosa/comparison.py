"""Functional motif comparison workflows."""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, TypedDict

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.ndimage import convolve1d

from mimosa.batches import (
    MINUS_STRAND,
    PLUS_STRAND,
    SCORE_PADDING,
    SequenceBatch,
    flatten_profile_bundle,
    pack_profile_bundle,
    profile_view,
)
from mimosa.cache import ProfileCacheSpec, fingerprint_batch, fingerprint_model, load_profile_cache, store_profile_cache
from mimosa.functions import (
    apply_score_log_tail_table_to_profile_bundle,
    build_score_log_tail_table,
    calc_co,
    calc_dice,
    prepare_profile_bundle,
    rowwise_co,
    rowwise_cosine,
    rowwise_dice,
    scores_to_empirical_log_tail_bundle,
)
from mimosa.models import (
    GenericModel,
    get_pfm,
    scan_model_strands,
)
from mimosa.validation import (
    validate_cache_mode,
    validate_kernel_size_range,
    validate_non_negative,
    validate_non_negative_int,
    validate_optional_thread_count,
    validate_pfm_top_fraction,
    validate_profile_normalization,
)

logger = logging.getLogger(__name__)
SUPPORTED_PROFILE_METRICS = ("co", "co_rowwise", "dice", "dice_rowwise", "cosine")
SUPPORTED_MOTIF_METRICS = ("pcc", "ed", "cosine")
MetricName = Literal["co", "co_rowwise", "dice", "dice_rowwise", "pcc", "ed", "cosine"]
_ALL_METRICS = frozenset((*SUPPORTED_PROFILE_METRICS, *SUPPORTED_MOTIF_METRICS))


class ComparatorConfig(TypedDict):
    metric: MetricName
    n_permutations: int
    seed: Optional[int]
    n_jobs: Optional[int]
    permute_rows: bool
    pfm_mode: bool
    pfm_top_fraction: float
    distortion_level: float
    search_range: int
    min_kernel_size: int
    max_kernel_size: int
    min_logfpr: Optional[float]
    window_radius: int
    realign_window: int
    profile_normalization: str
    cache_mode: str
    cache_dir: str
    background: Optional[SequenceBatch]


class ComparisonResult(TypedDict, total=False):
    query: str
    target: str
    score: float
    offset: int
    orientation: str
    metric: str
    n_sites: int


ORIENTATION_TIEBREAK = {"++": 0, "+-": 1, "-+": 2, "--": 3}
NUCLEOTIDE_CARDINALITY = 4
AMBIGUOUS_STATE_CARDINALITY = 5
MATRIX_RANK = 2
SIMILARITY_EPS = 1e-9
SURROGATE_SMOOTH_FILTER = np.array([0.25, 0.5, 0.25], dtype=np.float32)
SURROGATE_SIGN_FLIP_PROBABILITY = 0.5

PROFILE_ORIENTATION_PAIRS = (
    ("++", PLUS_STRAND, PLUS_STRAND),
    ("--", MINUS_STRAND, MINUS_STRAND),
    ("+-", PLUS_STRAND, MINUS_STRAND),
    ("-+", MINUS_STRAND, PLUS_STRAND),
)
SUPPORTED_PROFILE_NORMALIZATIONS = {"empirical_log_tail"}

registry: dict[str, Callable] = {}


def _register_comparison_strategy(name: str):
    """Register one comparison strategy."""

    def decorator(fn):
        registry[name] = fn
        return fn

    return decorator


def _validate_metric(metric: str) -> MetricName:
    normalized = str(metric).lower()
    if normalized not in _ALL_METRICS:
        options = ", ".join(sorted(_ALL_METRICS))
        raise ValueError(f"metric must be one of: {options}")
    return normalized  # type: ignore[return-value]


def create_comparator_config(**kwargs) -> ComparatorConfig:
    """Build one validated comparison options dictionary."""
    defaults: ComparatorConfig = {
        "metric": "pcc",
        "n_permutations": 0,
        "seed": None,
        "n_jobs": None,
        "permute_rows": False,
        "pfm_mode": False,
        "pfm_top_fraction": 0.05,
        "distortion_level": 0.4,
        "search_range": 10,
        "min_kernel_size": 3,
        "max_kernel_size": 11,
        "min_logfpr": None,
        "window_radius": 10,
        "realign_window": 3,
        "profile_normalization": "empirical_log_tail",
        "cache_mode": "off",
        "cache_dir": ".mimosa-cache",
        "background": None,
    }
    config = {**defaults, **kwargs}
    legacy_background = config.pop("promoters", None)
    if "background" not in kwargs and legacy_background is not None:
        config["background"] = legacy_background
    config["metric"] = _validate_metric(config["metric"])
    min_kernel_size, max_kernel_size = validate_kernel_size_range(config["min_kernel_size"], config["max_kernel_size"])
    config["min_kernel_size"] = min_kernel_size
    config["max_kernel_size"] = max_kernel_size
    config["min_logfpr"] = validate_non_negative("min_logfpr", config.get("min_logfpr"))
    config["window_radius"] = validate_non_negative_int("window_radius", config.get("window_radius", 10))
    config["realign_window"] = validate_non_negative_int("realign_window", config.get("realign_window", 3))
    config["profile_normalization"] = validate_profile_normalization(
        config.get("profile_normalization", "empirical_log_tail"),
        SUPPORTED_PROFILE_NORMALIZATIONS,
    )
    config["n_jobs"] = validate_optional_thread_count("n_jobs", config.get("n_jobs"))
    config["pfm_top_fraction"] = (
        validate_pfm_top_fraction(config.get("pfm_top_fraction")) or defaults["pfm_top_fraction"]
    )
    config["cache_mode"] = validate_cache_mode(config.get("cache_mode", "off"))
    return config


def _select_best_orientation(candidates):
    """Choose the highest-scoring orientation with deterministic tie-breaking."""
    return max(
        candidates, key=lambda candidate: (float(candidate["score"]), -ORIENTATION_TIEBREAK[candidate["orientation"]])
    )


def _get_profile_background_sequences(sequences, cfg: ComparatorConfig):
    """Return the sequence collection used to fit profile normalization."""
    return cfg.get("background") if cfg.get("background") is not None else sequences


def _cached_batch_fingerprint(runtime_cache: dict, batch, label: str) -> str:
    runtime_key = ("batch_fp", label, id(batch))
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached
    value = fingerprint_batch(batch) or f"no-{label}"
    runtime_cache[runtime_key] = value
    return value


def _resolve_raw_profile_bundle(model: GenericModel, sequences, runtime_cache: dict | None = None):
    """Resolve one raw strand-aware profile bundle before normalization."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("raw_profile_bundle", fingerprint_model(model), sequence_fp)
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    if model.type_key != "scores" and sequences is None:
        raise ValueError("Profile strategy requires sequences when comparing motif models.")

    profile_bundle = scan_model_strands(model, sequences)
    runtime_cache[runtime_key] = profile_bundle
    return profile_bundle


def _fit_profile_normalizer(
    model: GenericModel, background_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Fit normalization parameters from the calibration score sample."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    background_fp = _cached_batch_fingerprint(runtime_cache, background_sequences, "background")
    runtime_key = ("profile_normalizer", fingerprint_model(model), cfg["profile_normalization"], background_fp)
    if runtime_key in runtime_cache:
        return runtime_cache[runtime_key]

    background_bundle = _resolve_raw_profile_bundle(model, background_sequences, runtime_cache)
    calibration_sample = flatten_profile_bundle(background_bundle)
    if cfg["profile_normalization"] != "empirical_log_tail":
        raise ValueError(f"Unsupported profile normalization: {cfg['profile_normalization']}")

    normalizer = build_score_log_tail_table(calibration_sample)
    runtime_cache[runtime_key] = normalizer
    return normalizer


def _apply_profile_normalizer(profile_bundle, normalizer, profile_normalization: str):
    """Apply fitted normalization parameters to one raw profile bundle."""
    if profile_normalization != "empirical_log_tail":
        raise ValueError(f"Unsupported profile normalization: {profile_normalization}")
    return apply_score_log_tail_table_to_profile_bundle(profile_bundle, normalizer)


def _build_profile_cache_spec(
    model: GenericModel, sequences, background_sequences, cfg: ComparatorConfig, profile_kind: str
) -> ProfileCacheSpec:
    """Build one cache descriptor for a normalized profile bundle."""
    return {
        "model": model,
        "sequences": sequences,
        "background": background_sequences,
        "profile_kind": profile_kind,
        "cache_dir": cfg["cache_dir"],
    }


def _resolve_profile_bundle(
    model: GenericModel, sequences, background_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Resolve one model to the normalized strand-aware profile bundle used in profile comparisons."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    profile_kind = cfg["profile_normalization"]
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    background_fp = _cached_batch_fingerprint(runtime_cache, background_sequences, "background")
    runtime_key = (fingerprint_model(model), profile_kind, sequence_fp, background_fp)

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    cache_spec = None
    if cfg["cache_mode"] == "on":
        cache_spec = _build_profile_cache_spec(model, sequences, background_sequences, cfg, profile_kind)
        cached = load_profile_cache(cache_spec)
        if cached is not None:
            runtime_cache[runtime_key] = cached
            logger.debug("Profile cache hit for model '%s'.", model.name)
            return cached

    raw_bundle = _resolve_raw_profile_bundle(model, sequences, runtime_cache)
    normalizer = _fit_profile_normalizer(model, background_sequences, cfg, runtime_cache)
    profile_bundle = _apply_profile_normalizer(raw_bundle, normalizer, profile_kind)
    runtime_cache[runtime_key] = profile_bundle

    if cache_spec is not None:
        store_profile_cache(cache_spec, profile_bundle)
        logger.debug("Stored profile cache for model '%s'.", model.name)

    return profile_bundle


def _prepare_profile_model(
    model: GenericModel,
    sequences,
    background_sequences,
    cfg: ComparatorConfig,
    runtime_cache: dict | None = None,
):
    """Resolve one normalized profile bundle in a contiguous scoring layout."""
    bundle = _resolve_profile_bundle(model, sequences, background_sequences, cfg, runtime_cache)
    return prepare_profile_bundle(bundle)


def _build_window_offsets(window_radius: int) -> np.ndarray:
    """Return symmetric integer offsets for one site-centered window."""
    radius = int(window_radius)
    return np.arange(-radius, radius + 1, dtype=np.int32)


def _empty_positions() -> tuple[np.ndarray, np.ndarray]:
    """Return one empty anchor payload."""
    empty = np.empty(0, dtype=np.int32)
    return empty, empty


@njit(cache=False, nogil=False)
def _collect_best_anchor_positions_numba(scores: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collect one best anchor per row."""
    n_rows = scores.shape[0]
    rows = np.empty(n_rows, dtype=np.int32)
    positions = np.empty(n_rows, dtype=np.int32)
    out_index = 0

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        if length <= 0:
            continue

        best_position = 0
        best_score = scores[row_index, 0]
        for pos in range(1, length):
            score = scores[row_index, pos]
            if score > best_score:
                best_score = score
                best_position = pos

        rows[out_index] = row_index
        positions[out_index] = best_position
        out_index += 1

    return rows[:out_index], positions[:out_index]


@njit(cache=False, nogil=False)
def _count_threshold_anchor_positions_numba(scores: np.ndarray, lengths: np.ndarray, score_threshold: float) -> int:
    """Count threshold-selected anchors."""
    total = 0
    for row_index in range(scores.shape[0]):
        length = int(lengths[row_index])
        for pos in range(length):
            if scores[row_index, pos] >= score_threshold:
                total += 1
    return total


@njit(cache=False, nogil=False)
def _collect_threshold_anchor_positions_numba(
    scores: np.ndarray,
    lengths: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all anchors at or above the configured threshold."""
    total = _count_threshold_anchor_positions_numba(scores, lengths, score_threshold)
    rows = np.empty(total, dtype=np.int32)
    positions = np.empty(total, dtype=np.int32)
    out_index = 0

    for row_index in range(scores.shape[0]):
        length = int(lengths[row_index])
        for pos in range(length):
            if scores[row_index, pos] >= score_threshold:
                rows[out_index] = row_index
                positions[out_index] = pos
                out_index += 1

    return rows, positions


def _collect_anchor_sites(
    scores: np.ndarray,
    lengths: np.ndarray,
    score_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect site anchors in best or threshold mode for one strand-specific score matrix."""
    scores_array = np.ascontiguousarray(scores, dtype=np.float32)
    lengths_array = np.ascontiguousarray(lengths, dtype=np.int32)
    if scores_array.shape[0] == 0:
        return _empty_positions()
    if score_threshold is None:
        return _collect_best_anchor_positions_numba(scores_array, lengths_array)
    return _collect_threshold_anchor_positions_numba(scores_array, lengths_array, float(score_threshold))


def _filter_window_positions(
    rows: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    min_offset: int,
    max_offset: int,
) -> np.ndarray:
    """Return the mask of anchors whose full windows fit in both profiles."""
    if rows.size == 0:
        return np.zeros(0, dtype=bool)

    row_lengths1 = lengths1[rows]
    row_lengths2 = lengths2[rows]

    return (
        (pos1 + min_offset >= 0)
        & (pos1 + max_offset < row_lengths1)
        & (pos2 + min_offset >= 0)
        & (pos2 + max_offset < row_lengths2)
    )


def _empty_candidate_triplets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one empty candidate payload."""
    empty = np.empty(0, dtype=np.int32)
    return empty, empty, empty


def _collect_model1_window_candidates(
    anchor_rows: np.ndarray,
    anchor_pos1: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    shift: int,
    min_offset: int,
    max_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid windows centered on model1 anchors."""
    if anchor_rows.size == 0:
        return _empty_candidate_triplets()

    pos2 = anchor_pos1 + int(shift)
    valid = _filter_window_positions(anchor_rows, anchor_pos1, pos2, lengths1, lengths2, min_offset, max_offset)
    if not np.any(valid):
        return _empty_candidate_triplets()

    return anchor_rows[valid], anchor_pos1[valid], pos2[valid]


@njit(cache=False, nogil=False)
def _collect_model2_window_candidates_numba(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    anchor_rows: np.ndarray,
    anchor_pos2: np.ndarray,
    shift: int,
    realign_window: int,
    min_offset: int,
    max_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid windows centered on model2 anchors and realigned on model1."""
    size = anchor_rows.shape[0]
    rows = np.empty(size, dtype=np.int32)
    pos1 = np.empty(size, dtype=np.int32)
    pos2 = np.empty(size, dtype=np.int32)
    out_index = 0

    for candidate_index in range(size):
        row = int(anchor_rows[candidate_index])
        row_length1 = int(lengths1[row])
        row_length2 = int(lengths2[row])
        if row_length1 <= 0 or row_length2 <= 0:
            continue

        expected_pos1 = int(anchor_pos2[candidate_index]) - shift
        left = max(0, expected_pos1 - realign_window)
        right = min(row_length1 - 1, expected_pos1 + realign_window)
        if left > right:
            continue

        best_pos1 = left
        best_score = scores1[row, left]
        for pos in range(left + 1, right + 1):
            score = scores1[row, pos]
            if score > best_score:
                best_score = score
                best_pos1 = pos

        aligned_pos2 = best_pos1 + shift
        if (
            best_pos1 + min_offset < 0
            or best_pos1 + max_offset >= row_length1
            or aligned_pos2 + min_offset < 0
            or aligned_pos2 + max_offset >= row_length2
        ):
            continue

        rows[out_index] = row
        pos1[out_index] = best_pos1
        pos2[out_index] = aligned_pos2
        out_index += 1

    return rows[:out_index], pos1[:out_index], pos2[:out_index]


def _collect_model2_window_candidates(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    anchor_rows: np.ndarray,
    anchor_pos2: np.ndarray,
    shift: int,
    min_offset: int,
    max_offset: int,
    realign_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid windows centered on model2 anchors and realigned on model1."""
    if anchor_rows.size == 0:
        return _empty_candidate_triplets()

    return _collect_model2_window_candidates_numba(
        np.ascontiguousarray(scores1, dtype=np.float32),
        np.ascontiguousarray(lengths1, dtype=np.int32),
        np.ascontiguousarray(lengths2, dtype=np.int32),
        np.ascontiguousarray(anchor_rows, dtype=np.int32),
        np.ascontiguousarray(anchor_pos2, dtype=np.int32),
        int(shift),
        int(realign_window),
        int(min_offset),
        int(max_offset),
    )


def _merge_window_candidates(
    candidate_set1: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate_set2: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge model-derived candidates with OR semantics and deterministic deduplication."""
    rows1, pos1_1, pos2_1 = candidate_set1
    rows2, pos1_2, pos2_2 = candidate_set2

    if rows1.size == 0:
        return rows2, pos1_2, pos2_2
    if rows2.size == 0:
        return rows1, pos1_1, pos2_1

    rows = np.concatenate((rows1, rows2))
    pos1 = np.concatenate((pos1_1, pos1_2))
    pos2 = np.concatenate((pos2_1, pos2_2))

    order = np.lexsort((pos2, pos1, rows))
    rows = rows[order]
    pos1 = pos1[order]
    pos2 = pos2[order]

    keep = np.ones(rows.size, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (pos1[1:] != pos1[:-1]) | (pos2[1:] != pos2[:-1])
    return rows[keep], pos1[keep], pos2[keep]


def _extract_selected_windows(
    scores: np.ndarray,
    rows: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Extract one dense window matrix from selected rows and positions."""
    if rows.size == 0:
        return np.empty((0, offsets.size), dtype=np.float32)
    cols = positions[:, None] + offsets[None, :]
    return np.asarray(scores[rows[:, None], cols], dtype=np.float32)


def _score_window_collection(metric: str, windows1: np.ndarray, windows2: np.ndarray) -> float:
    """Score one selected window collection with the requested profile metric."""
    if windows1.shape != windows2.shape:
        raise ValueError("Window collections must have identical shapes.")
    if windows1.size == 0:
        return 0.0
    if metric == "co":
        return calc_co(windows1, windows2)
    if metric == "co_rowwise":
        return _mean_finite_row_scores(rowwise_co(windows1, windows2))
    if metric == "dice":
        return calc_dice(windows1, windows2)
    if metric == "dice_rowwise":
        return _mean_finite_row_scores(rowwise_dice(windows1, windows2))
    if metric != "cosine":
        options = ", ".join(repr(metric_name) for metric_name in SUPPORTED_PROFILE_METRICS)
        raise ValueError(f"metric must be one of: {options}")
    return _mean_finite_row_scores(rowwise_cosine(windows1, windows2))


def _mean_finite_row_scores(values: np.ndarray) -> float:
    """Average finite row-wise window scores and treat all-masked inputs as zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _compute_shifted_window_alignment(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    scores2: np.ndarray,
    lengths2: np.ndarray,
    shift: int,
    offsets: np.ndarray,
    min_offset: int,
    max_offset: int,
    query_anchors: tuple[np.ndarray, np.ndarray],
    target_anchors: tuple[np.ndarray, np.ndarray],
    realign_window: int,
    metric: str,
) -> dict[str, int | float]:
    """Evaluate one shift for one oriented pair of normalized score profiles."""
    model1_candidates = _collect_model1_window_candidates(
        query_anchors[0],
        query_anchors[1],
        lengths1,
        lengths2,
        shift,
        min_offset,
        max_offset,
    )
    model2_candidates = _collect_model2_window_candidates(
        scores1,
        lengths1,
        lengths2,
        target_anchors[0],
        target_anchors[1],
        shift,
        min_offset,
        max_offset,
        realign_window,
    )
    rows, pos1, pos2 = _merge_window_candidates(model1_candidates, model2_candidates)
    if rows.size == 0:
        return {"score": 0.0, "shift": int(shift), "n_sites": 0}

    windows1 = _extract_selected_windows(scores1, rows, pos1, offsets)
    windows2 = _extract_selected_windows(scores2, rows, pos2, offsets)
    return {
        "score": _score_window_collection(metric, windows1, windows2),
        "shift": int(shift),
        "n_sites": int(rows.size),
    }


def _score_profile_orientation_pair(
    query_bundle: dict,
    target_bundle: dict,
    query_strand: int,
    target_strand: int,
    offsets: np.ndarray,
    min_offset: int,
    max_offset: int,
    query_anchors: tuple[np.ndarray, np.ndarray],
    target_anchors: tuple[np.ndarray, np.ndarray],
    cfg: ComparatorConfig,
) -> dict:
    """Score one profile orientation across all tested shifts."""
    query_scores = query_bundle["values"][query_strand]
    target_scores = target_bundle["values"][target_strand]
    query_lengths = query_bundle["lengths"]
    target_lengths = target_bundle["lengths"]

    if query_scores.shape[0] != target_scores.shape[0]:
        raise ValueError("Profile bundles must have the same number of rows.")

    best = {"score": 0.0, "shift": 0, "n_sites": 0}
    for shift in range(-int(cfg["search_range"]), int(cfg["search_range"]) + 1):
        candidate = _compute_shifted_window_alignment(
            query_scores,
            query_lengths,
            target_scores,
            target_lengths,
            shift,
            offsets,
            min_offset,
            max_offset,
            query_anchors,
            target_anchors,
            int(cfg["realign_window"]),
            str(cfg["metric"]),
        )
        if float(candidate["score"]) > float(best["score"]) or (
            float(candidate["score"]) == float(best["score"])
            and (
                int(candidate["n_sites"]) > int(best["n_sites"])
                or (
                    int(candidate["n_sites"]) == int(best["n_sites"])
                    and abs(int(candidate["shift"])) < abs(int(best["shift"]))
                )
            )
        ):
            best = candidate

    return {
        "score": float(best["score"]),
        "shift": int(best["shift"]),
        "n_sites": int(best["n_sites"]),
        "target_strand": int(target_strand),
    }


def _score_profile_candidates(query_bundle: dict, target_bundle: dict, pair_specs, cfg: ComparatorConfig) -> list[dict]:
    """Score all requested orientation pairs with the window-based profile algorithm."""
    min_logfpr = cfg["min_logfpr"]
    score_threshold = None if min_logfpr is None or float(min_logfpr) <= 0.0 else float(min_logfpr)
    offsets = _build_window_offsets(int(cfg["window_radius"]))
    min_offset = -int(cfg["window_radius"])
    max_offset = int(cfg["window_radius"])
    query_strands = {int(query_strand) for _, query_strand, _ in pair_specs}
    target_strands = {int(target_strand) for _, _, target_strand in pair_specs}
    query_anchor_cache = {
        strand_index: _collect_anchor_sites(
            query_bundle["values"][strand_index],
            query_bundle["lengths"],
            score_threshold,
        )
        for strand_index in query_strands
    }
    target_anchor_cache = {
        strand_index: _collect_anchor_sites(
            target_bundle["values"][strand_index],
            target_bundle["lengths"],
            score_threshold,
        )
        for strand_index in target_strands
    }
    candidates = []
    for orientation, query_strand, target_strand in pair_specs:
        best = _score_profile_orientation_pair(
            query_bundle,
            target_bundle,
            int(query_strand),
            int(target_strand),
            offsets,
            min_offset,
            max_offset,
            query_anchor_cache[int(query_strand)],
            target_anchor_cache[int(target_strand)],
            cfg,
        )
        best["orientation"] = orientation
        candidates.append(best)
    return candidates


def _build_profile_result(query_name: str, target_name: str, best: dict, metric: str) -> ComparisonResult:
    """Build one profile comparison result payload from the best candidate."""
    return {
        "query": query_name,
        "target": target_name,
        "score": float(best["score"]),
        "offset": int(best["shift"]),
        "orientation": best["orientation"],
        "metric": metric,
        "n_sites": int(best["n_sites"]),
    }


def _attach_profile_null_statistics(
    result: dict,
    query_bundle: dict,
    target_bundle: dict,
    best_target_strand: int,
    cfg: ComparatorConfig,
) -> None:
    """Attach Monte Carlo statistics for one profile comparison result."""
    if cfg["n_permutations"] <= 0:
        return

    obs_score = float(result["score"])
    best_target_profile = profile_view(target_bundle, best_target_strand)

    orientation_pairs = (
        [("++", PLUS_STRAND, 0), ("-+", MINUS_STRAND, 0)]
        if best_target_strand == PLUS_STRAND
        else [("+-", PLUS_STRAND, 0), ("--", MINUS_STRAND, 0)]
    )

    def surrogate_gen(rng):
        surrogate_bundle = _create_surrogate_bundle(best_target_profile, rng, cfg)
        candidates = _score_profile_candidates(query_bundle, surrogate_bundle, orientation_pairs, cfg)
        return float(_select_best_orientation(candidates)["score"])

    nulls, null_mean, null_std = run_montecarlo(
        lambda value: value,
        surrogate_gen,
        cfg["n_permutations"],
        cfg["seed"],
    )
    _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg["n_permutations"])


def _create_surrogate_bundle(profile, rng: np.random.Generator, cfg: ComparatorConfig):
    """Generate one surrogate single-profile bundle by vectorized convolution and renormalization."""
    min_kernel_size = int(cfg["min_kernel_size"])
    max_kernel_size = int(cfg["max_kernel_size"])
    first_odd = min_kernel_size if min_kernel_size % 2 == 1 else min_kernel_size + 1
    n_odd = ((max_kernel_size - first_odd) // 2) + 1
    kernel_size = int(first_odd + 2 * int(rng.integers(0, n_odd)))
    center = kernel_size // 2

    identity_kernel = np.zeros(kernel_size, dtype=np.float32)
    identity_kernel[center] = 1.0
    alpha = float(np.clip(cfg["distortion_level"], 0.0, 1.0))
    kernel = rng.normal(0.0, 1.0, size=kernel_size).astype(np.float32)
    if kernel_size >= len(SURROGATE_SMOOTH_FILTER):
        kernel = np.convolve(kernel, SURROGATE_SMOOTH_FILTER, mode="same").astype(np.float32)
    kernel /= np.linalg.norm(kernel) + 1e-8
    final_kernel = (1.0 - alpha) * identity_kernel + alpha * kernel
    if rng.uniform() < SURROGATE_SIGN_FLIP_PROBABILITY:
        final_kernel = -final_kernel
    final_kernel /= np.linalg.norm(final_kernel) + 1e-8

    values = np.asarray(profile["values"], dtype=np.float32)
    convolved = convolve1d(values, final_kernel, axis=1, mode="constant", cval=0.0)
    mask = (
        np.arange(convolved.shape[1], dtype=np.int32)[None, :]
        < np.asarray(
            profile["lengths"],
            dtype=np.int32,
        )[:, None]
    )
    convolved = np.asarray(convolved, dtype=np.float32)
    convolved[~mask] = SCORE_PADDING

    surrogate = pack_profile_bundle(convolved[None, ...], profile["lengths"], SCORE_PADDING)
    if cfg["profile_normalization"] != "empirical_log_tail":
        raise ValueError(f"Unsupported profile normalization: {cfg['profile_normalization']}")
    return scores_to_empirical_log_tail_bundle(surrogate)


def run_montecarlo(obs_score_func: Callable, surrogate_generator_func: Callable, n_permutations: int, seed, *args):
    """Run a Monte Carlo workflow sequentially while compiled kernels handle parallel work."""
    if n_permutations <= 0:
        return np.array([]), 0.0, 0.0

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31, size=n_permutations)
    results = []
    for seed_value in seeds:
        rng = np.random.default_rng(int(seed_value))
        surrogate = surrogate_generator_func(rng, *args)
        result = obs_score_func(surrogate)
        if result is not None:
            results.append(result)

    null_scores = np.asarray(results, dtype=np.float32)
    if null_scores.size == 0:
        return null_scores, 0.0, 0.0
    return null_scores, float(np.mean(null_scores)), float(np.std(null_scores))


def _update_result_with_null_statistics(
    result: dict, obs_score: float, stats: tuple[np.ndarray, float, float], n_permutations: int
) -> None:
    """Attach Monte Carlo summary statistics to one result payload."""
    null_scores, null_mean, null_std = stats
    if null_scores.size == 0:
        return

    z_score = (obs_score - null_mean) / (null_std + SIMILARITY_EPS)
    p_value = (int(np.sum(null_scores >= obs_score)) + 1.0) / (n_permutations + 1.0)
    result.update(
        {
            "p-value": float(p_value),
            "z-score": float(z_score),
            "null_mean": null_mean,
            "null_std": null_std,
        }
    )


def _is_power_of_four(value: int) -> bool:
    """Return True when the provided value is a positive power of four."""
    if value < 1:
        return False
    while value % NUCLEOTIDE_CARDINALITY == 0:
        value //= NUCLEOTIDE_CARDINALITY
    return value == 1


def _looks_like_alphabet_axis(size: int) -> bool:
    """Return True when one axis likely encodes nucleotide states."""
    return size in (NUCLEOTIDE_CARDINALITY, AMBIGUOUS_STATE_CARDINALITY) or _is_power_of_four(size)


def _normalize_motif_tensor(matrix: np.ndarray) -> np.ndarray:
    """Normalize one motif representation to a tensor with sequence positions last."""
    normalized = np.asarray(matrix)
    if normalized.ndim == MATRIX_RANK:
        rows, cols = normalized.shape
        if not _looks_like_alphabet_axis(rows) and _looks_like_alphabet_axis(cols):
            normalized = normalized.T
        if normalized.shape[0] == AMBIGUOUS_STATE_CARDINALITY:
            normalized = normalized[:NUCLEOTIDE_CARDINALITY, :]
        if normalized.shape[0] > NUCLEOTIDE_CARDINALITY and _is_power_of_four(normalized.shape[0]):
            order = int(round(np.log(normalized.shape[0]) / np.log(NUCLEOTIDE_CARDINALITY)))
            normalized = normalized.reshape((NUCLEOTIDE_CARDINALITY,) * order + (normalized.shape[1],))
        return normalized

    if normalized.shape[0] > AMBIGUOUS_STATE_CARDINALITY:
        normalized = np.moveaxis(normalized, 0, -1)
    clean_slice = tuple(
        slice(0, NUCLEOTIDE_CARDINALITY) if axis < normalized.ndim - 1 else slice(None)
        for axis in range(normalized.ndim)
    )
    return normalized[clean_slice]


def _reverse_complement_motif_tensor(matrix: np.ndarray) -> np.ndarray:
    """Return the reverse-complement tensor for one normalized motif matrix."""
    order = matrix.ndim - 1
    axes = tuple(range(order - 1, -1, -1)) + (order,)
    reverse = np.transpose(matrix, axes=axes)
    for axis in range(order):
        reverse = np.flip(reverse, axis=axis)
    return np.flip(reverse, axis=-1)


def _prepare_motif(matrix: np.ndarray) -> dict:
    """Build flattened forward and reverse-complement views for alignment."""
    normalized = _normalize_motif_tensor(matrix)
    reverse = _reverse_complement_motif_tensor(normalized)
    return {
        "forward": normalized.reshape(-1, normalized.shape[-1]),
        "reverse": reverse.reshape(-1, reverse.shape[-1]),
    }


def _vectorized_pcc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute column-wise Pearson correlation coefficients."""
    x1_centered = x1 - np.mean(x1, axis=0, keepdims=True)
    x2_centered = x2 - np.mean(x2, axis=0, keepdims=True)
    numerator = np.sum(x1_centered * x2_centered, axis=0)
    denominator = np.sqrt(np.sum(x1_centered**2, axis=0)) * np.sqrt(np.sum(x2_centered**2, axis=0))
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > SIMILARITY_EPS)
    return result


def _vectorized_cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute column-wise cosine similarity."""
    numerator = np.sum(x1 * x2, axis=0)
    denominator = np.linalg.norm(x1, axis=0) * np.linalg.norm(x2, axis=0)
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > SIMILARITY_EPS)
    return result


def _score_motif_columns(metric: str, query_columns: np.ndarray, target_columns: np.ndarray) -> float:
    """Score one aligned column block."""
    overlap = query_columns.shape[1]
    if metric == "pcc":
        return float(np.sum(_vectorized_pcc(query_columns, target_columns)) / overlap)
    if metric == "cosine":
        return float(np.sum(_vectorized_cosine(query_columns, target_columns)) / overlap)
    if metric != "ed":
        raise ValueError("metric must be one of: 'pcc', 'ed', 'cosine'")
    distances = np.sqrt(np.sum((query_columns - target_columns) ** 2, axis=0))
    return float(-np.sum(distances) / overlap)


def _align_motif_matrices(query_matrix: np.ndarray, target_matrix: np.ndarray, metric: str) -> tuple[float, int]:
    """Align two prepared matrices and return the best score and offset."""
    query_length = query_matrix.shape[1]
    target_length = target_matrix.shape[1]
    min_overlap = min(query_length, target_length) / 2.0
    best_score = float(-np.inf)
    best_offset = 0

    for offset in range(-(target_length - 1), query_length):
        if offset < 0:
            overlap = min(query_length, target_length + offset)
            if overlap < min_overlap:
                continue
            query_slice = slice(0, overlap)
            target_slice = slice(-offset, -offset + overlap)
        else:
            overlap = min(query_length - offset, target_length)
            if overlap < min_overlap:
                continue
            query_slice = slice(offset, offset + overlap)
            target_slice = slice(0, overlap)

        score = _score_motif_columns(metric, query_matrix[:, query_slice], target_matrix[:, target_slice])
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score, best_offset


def _score_motif_candidates(query: dict, target: dict, metric: str) -> list[dict]:
    """Score all orientation pairs for one prepared motif pair."""
    candidates = []
    for orientation, query_matrix, target_matrix in (
        ("++", query["forward"], target["forward"]),
        ("+-", query["forward"], target["reverse"]),
        ("-+", query["reverse"], target["forward"]),
        ("--", query["reverse"], target["reverse"]),
    ):
        score, offset = _align_motif_matrices(query_matrix, target_matrix, metric)
        candidates.append({"orientation": orientation, "score": score, "offset": offset})
    return candidates


def _best_motif_score(query: dict, target: dict, metric: str) -> float:
    """Return the best orientation score for one motif pair."""
    return float(_select_best_orientation(_score_motif_candidates(query, target, metric))["score"])


def _resolve_motif_matrix(
    model: GenericModel,
    sequences,
    cfg: ComparatorConfig,
    use_pfm_mode: bool,
    runtime_cache: dict | None = None,
):
    """Resolve one motif matrix for direct or PFM-based comparison."""
    if not use_pfm_mode:
        return model.representation
    if sequences is None:
        raise ValueError("sequences are required for pfm_mode")

    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("motif_matrix", fingerprint_model(model), sequence_fp, cfg["pfm_top_fraction"])
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    matrix = get_pfm(model, sequences, top_fraction=cfg["pfm_top_fraction"])
    runtime_cache[runtime_key] = matrix
    return matrix


def _prepare_motif_model(
    model: GenericModel,
    sequences,
    cfg: ComparatorConfig,
    use_pfm_mode: bool,
    runtime_cache: dict | None = None,
):
    """Resolve one motif matrix and its prepared forward/reverse views."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("prepared_motif", fingerprint_model(model), use_pfm_mode, sequence_fp, cfg["pfm_top_fraction"])
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    matrix = _resolve_motif_matrix(model, sequences, cfg, use_pfm_mode, runtime_cache)
    prepared = _prepare_motif(matrix)
    state = (matrix, prepared)
    runtime_cache[runtime_key] = state
    return state


def _permute_motif_matrix(matrix: np.ndarray, rng: np.random.Generator, permute_rows: bool) -> np.ndarray:
    """Shuffle motif columns and optionally nucleotide axes for null generation."""
    permuted = np.array(matrix, copy=True)
    column_order = np.arange(permuted.shape[-1])
    rng.shuffle(column_order)
    permuted = permuted[..., column_order]
    if not permute_rows:
        return permuted

    row_order = rng.permutation(permuted.shape[0])
    for axis in range(permuted.ndim - 1):
        permuted = np.take(permuted, row_order, axis=axis)
    return permuted


def _score_prepared_motif_pair(query: dict, target: dict, metric: str) -> dict:
    """Score one prepared motif pair and return the best orientation candidate."""
    return _select_best_orientation(_score_motif_candidates(query, target, metric))


def _build_motif_result(query_name: str, target_name: str, best: dict, metric: str) -> ComparisonResult:
    """Build one motif comparison result payload from the best candidate."""
    return {
        "query": query_name,
        "target": target_name,
        "score": float(best["score"]),
        "offset": int(best["offset"]),
        "orientation": best["orientation"],
        "metric": metric,
    }


def _attach_motif_null_statistics(
    result: dict, prepared_query: dict, target_matrix: np.ndarray, cfg: ComparatorConfig
) -> None:
    """Attach Monte Carlo statistics for one motif comparison result."""
    if cfg["n_permutations"] <= 0:
        return

    obs_score = float(result["score"])

    def gen_surrogate(rng):
        surrogate = _permute_motif_matrix(target_matrix, rng, cfg["permute_rows"])
        return _best_motif_score(prepared_query, _prepare_motif(surrogate), cfg["metric"])

    nulls, null_mean, null_std = run_montecarlo(
        lambda value: value,
        gen_surrogate,
        cfg["n_permutations"],
        cfg["seed"],
    )
    _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg["n_permutations"])


@_register_comparison_strategy("motif")
def strategy_motif(model1: GenericModel, model2: GenericModel, sequences, cfg: ComparatorConfig) -> ComparisonResult:
    """Matrix-based comparison strategy (PCC/ED/Cosine)."""
    runtime_cache = {}
    use_pfm_mode = cfg["pfm_mode"] or (model1.type_key != model2.type_key)
    _query_matrix, prepared1 = _prepare_motif_model(model1, sequences, cfg, use_pfm_mode, runtime_cache)
    matrix2, prepared2 = _prepare_motif_model(model2, sequences, cfg, use_pfm_mode, runtime_cache)
    best = _score_prepared_motif_pair(prepared1, prepared2, cfg["metric"])
    result = _build_motif_result(model1.name, model2.name, best, cfg["metric"])
    _attach_motif_null_statistics(result, prepared1, matrix2, cfg)
    return result


@_register_comparison_strategy("profile")
def strategy_profile(model1: GenericModel, model2: GenericModel, sequences, cfg: ComparatorConfig) -> ComparisonResult:
    """Window-based profile comparison strategy (CO/rowwise-CO/Dice/Cosine similarity)."""
    runtime_cache = {}
    background_sequences = _get_profile_background_sequences(sequences, cfg)
    bundle1 = _prepare_profile_model(model1, sequences, background_sequences, cfg, runtime_cache)
    bundle2 = _prepare_profile_model(model2, sequences, background_sequences, cfg, runtime_cache)
    best = _select_best_orientation(_score_profile_candidates(bundle1, bundle2, PROFILE_ORIENTATION_PAIRS, cfg))
    result = _build_profile_result(model1.name, model2.name, best, cfg["metric"])
    _attach_profile_null_statistics(
        result,
        bundle1,
        bundle2,
        int(best["target_strand"]),
        cfg,
    )
    return result


def _compare_motif_one_to_many(
    query_model: GenericModel,
    target_models,
    sequences,
    cfg: ComparatorConfig,
) -> list[dict]:
    """Compare one motif query against many targets while reusing prepared query state."""
    target_list = list(target_models)
    if not target_list:
        return []

    query_cache = {}
    use_pfm_modes = {
        bool(cfg["pfm_mode"] or (query_model.type_key != target_model.type_key)) for target_model in target_list
    }
    prepared_query_by_mode = {}
    for use_pfm_mode in use_pfm_modes:
        _query_matrix, prepared_query = _prepare_motif_model(query_model, sequences, cfg, use_pfm_mode, query_cache)
        prepared_query_by_mode[use_pfm_mode] = prepared_query

    def _score_target(target_model: GenericModel) -> dict:
        target_cache = {}
        try:
            use_pfm_mode = bool(cfg["pfm_mode"] or (query_model.type_key != target_model.type_key))
            prepared_query = prepared_query_by_mode[use_pfm_mode]
            target_matrix, prepared_target = _prepare_motif_model(
                target_model,
                sequences,
                cfg,
                use_pfm_mode,
                target_cache,
            )
            best = _score_prepared_motif_pair(prepared_query, prepared_target, cfg["metric"])
            result = _build_motif_result(query_model.name, target_model.name, best, cfg["metric"])
            _attach_motif_null_statistics(result, prepared_query, target_matrix, cfg)
            return result
        finally:
            target_cache.clear()

    return _run_target_comparisons(target_list, cfg["n_jobs"], _score_target)


def _compare_profile_one_to_many(
    query_model: GenericModel, target_models, sequences, cfg: ComparatorConfig
) -> list[dict]:
    """Compare one profile query against many targets while reusing normalized query profiles."""
    target_list = list(target_models)
    if not target_list:
        return []

    query_cache = {}
    background_sequences = _get_profile_background_sequences(sequences, cfg)
    query_bundle = _prepare_profile_model(
        query_model,
        sequences,
        background_sequences,
        cfg,
        query_cache,
    )

    def _score_target(target_model: GenericModel) -> dict:
        target_cache = {}
        try:
            target_bundle = _prepare_profile_model(
                target_model,
                sequences,
                background_sequences,
                cfg,
                target_cache,
            )
            best = _select_best_orientation(
                _score_profile_candidates(query_bundle, target_bundle, PROFILE_ORIENTATION_PAIRS, cfg)
            )
            result = _build_profile_result(query_model.name, target_model.name, best, cfg["metric"])
            _attach_profile_null_statistics(
                result,
                query_bundle,
                target_bundle,
                int(best["target_strand"]),
                cfg,
            )
            return result
        finally:
            target_cache.clear()

    return _run_target_comparisons(target_list, cfg["n_jobs"], _score_target)


def _resolve_target_job_count(n_jobs: int | None) -> int:
    """Resolve one target-level worker count from the compatibility config key."""
    return -1 if n_jobs is None else int(n_jobs)


def _run_target_comparisons(target_models: list[GenericModel], n_jobs: int | None, worker: Callable) -> list[dict]:
    """Execute one worker across targets sequentially or with joblib threads."""
    if not target_models:
        return []

    n_jobs = _resolve_target_job_count(n_jobs)
    if n_jobs == 1 or len(target_models) == 1:
        return [worker(target_model) for target_model in target_models]

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(target_model) for target_model in target_models)


def compare(
    model1: GenericModel,
    model2: GenericModel,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
) -> dict:
    """Main entry point for motif comparison."""
    try:
        strategy_fn = registry[strategy]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}") from exc

    effective_config = dict(config)
    if background is not None:
        effective_config["background"] = background
    return strategy_fn(model1, model2, sequences, effective_config)


def compare_one_to_many(
    query_model: GenericModel,
    target_models,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
) -> list[dict]:
    """Main entry point for one-vs-many motif comparison."""
    effective_config = dict(config)
    if background is not None:
        effective_config["background"] = background

    if strategy == "profile":
        return _compare_profile_one_to_many(query_model, target_models, sequences, effective_config)
    if strategy == "motif":
        return _compare_motif_one_to_many(query_model, target_models, sequences, effective_config)
    available = ", ".join(sorted(registry))
    raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")
