"""Functional motif comparison workflows."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Literal, Optional, TypedDict

import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.ndimage import convolve1d

from mimosa.batches import (
    MINUS_STRAND,
    PLUS_STRAND,
    SCORE_PADDING,
    DenseBatch,
    flatten_profile_bundle,
    pack_profile_bundle,
    profile_view,
)
from mimosa.cache import ProfileCacheSpec, fingerprint_batch, fingerprint_model, load_profile_cache, store_profile_cache
from mimosa.functions import (
    _prepare_profile_bundle_for_scoring,
    _score_prepared_profile_orientations,
    apply_score_log_tail_table_to_profile_bundle,
    build_profile_score_options,
    build_score_log_tail_table,
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
    validate_optional_thread_count,
    validate_pfm_top_fraction,
    validate_profile_normalization,
)

logger = logging.getLogger(__name__)
MetricName = Literal["co", "dice", "pcc", "ed", "cosine"]
_ALL_METRICS = {"co", "dice", "pcc", "ed", "cosine"}


class ComparatorConfig(TypedDict):
    metric: MetricName
    n_permutations: int
    seed: Optional[int]
    numba_threads: Optional[int]
    permute_rows: bool
    pfm_mode: bool
    pfm_top_fraction: float
    distortion_level: float
    search_range: int
    min_kernel_size: int
    max_kernel_size: int
    min_logfpr: Optional[float]
    profile_normalization: str
    cache_mode: str
    cache_dir: str
    promoters: Optional[DenseBatch]


class ComparisonResult(TypedDict, total=False):
    query: str
    target: str
    score: float
    offset: int
    orientation: str
    metric: str


ORIENTATION_TIEBREAK = {"++": 0, "+-": 1, "-+": 2, "--": 3}
NUCLEOTIDE_CARDINALITY = 4
AMBIGUOUS_STATE_CARDINALITY = 5
MATRIX_RANK = 2
SIMILARITY_EPS = 1e-9
SURROGATE_SMOOTH_FILTER = np.array([0.25, 0.5, 0.25], dtype=np.float32)
SURROGATE_SIGN_FLIP_PROBABILITY = 0.5

PROFILE_NORMALIZATION_REGISTRY = {
    "empirical_log_tail": {
        "fit": build_score_log_tail_table,
        "apply_bundle": apply_score_log_tail_table_to_profile_bundle,
        "normalize_bundle": scores_to_empirical_log_tail_bundle,
    },
}
PROFILE_ORIENTATION_PAIRS = (
    ("++", PLUS_STRAND, PLUS_STRAND),
    ("--", MINUS_STRAND, MINUS_STRAND),
    ("+-", PLUS_STRAND, MINUS_STRAND),
    ("-+", MINUS_STRAND, PLUS_STRAND),
)

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
        "numba_threads": None,
        "permute_rows": False,
        "pfm_mode": False,
        "pfm_top_fraction": 0.05,
        "distortion_level": 0.4,
        "search_range": 10,
        "min_kernel_size": 3,
        "max_kernel_size": 11,
        "min_logfpr": None,
        "profile_normalization": "empirical_log_tail",
        "cache_mode": "off",
        "cache_dir": ".mimosa-cache",
        "promoters": None,
    }
    config = {**defaults, **kwargs}
    config["metric"] = _validate_metric(config["metric"])
    min_kernel_size, max_kernel_size = validate_kernel_size_range(config["min_kernel_size"], config["max_kernel_size"])
    config["min_kernel_size"] = min_kernel_size
    config["max_kernel_size"] = max_kernel_size
    config["min_logfpr"] = validate_non_negative("min_logfpr", config.get("min_logfpr"))
    config["profile_normalization"] = validate_profile_normalization(
        config.get("profile_normalization", "empirical_log_tail"),
        PROFILE_NORMALIZATION_REGISTRY,
    )
    config["numba_threads"] = validate_optional_thread_count("numba_threads", config.get("numba_threads"))
    config["pfm_top_fraction"] = (
        validate_pfm_top_fraction(config.get("pfm_top_fraction")) or defaults["pfm_top_fraction"]
    )
    config["cache_mode"] = validate_cache_mode(config.get("cache_mode", "off"))
    return config


@contextmanager
def _numba_thread_scope(numba_threads: int | None):
    """Apply one temporary numba thread setting for the current process."""
    if numba_threads is None:
        yield
        return

    previous_threads = get_num_threads()
    set_num_threads(numba_threads)
    try:
        yield
    finally:
        set_num_threads(previous_threads)


def _select_best_orientation(candidates):
    """Choose the highest-scoring orientation with deterministic tie-breaking."""
    return max(
        candidates, key=lambda candidate: (float(candidate["score"]), -ORIENTATION_TIEBREAK[candidate["orientation"]])
    )


def _get_profile_calibration_sequences(sequences, cfg: ComparatorConfig):
    """Return the sequence collection used to fit profile normalization."""
    return cfg.get("promoters") if cfg.get("promoters") is not None else sequences


def _get_profile_normalization_strategy(name: str) -> dict:
    """Resolve one registered profile-normalization strategy."""
    try:
        return PROFILE_NORMALIZATION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILE_NORMALIZATION_REGISTRY))
        raise ValueError(f"Unsupported profile normalization: {name}. Available: {available}") from exc


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
    model: GenericModel, calibration_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Fit normalization parameters from the calibration score sample."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    calibration_fp = _cached_batch_fingerprint(runtime_cache, calibration_sequences, "calibration")
    runtime_key = ("profile_normalizer", fingerprint_model(model), cfg["profile_normalization"], calibration_fp)
    if runtime_key in runtime_cache:
        return runtime_cache[runtime_key]

    calibration_bundle = _resolve_raw_profile_bundle(model, calibration_sequences, runtime_cache)
    calibration_sample = flatten_profile_bundle(calibration_bundle)
    strategy = _get_profile_normalization_strategy(cfg["profile_normalization"])
    normalizer = (cfg["profile_normalization"], strategy["fit"](calibration_sample))
    runtime_cache[runtime_key] = normalizer
    return normalizer


def _apply_profile_normalizer(profile_bundle, normalizer):
    """Apply fitted normalization parameters to one raw profile bundle."""
    strategy_name, params = normalizer
    strategy = _get_profile_normalization_strategy(strategy_name)
    return strategy["apply_bundle"](profile_bundle, params)


def _build_profile_score_options(cfg: ComparatorConfig) -> dict:
    """Build one profile-scoring options dictionary from comparator config."""
    return build_profile_score_options(
        search_range=cfg["search_range"],
        min_value=0.0 if cfg["min_logfpr"] is None else float(cfg["min_logfpr"]),
        metric=cfg["metric"],
    )


def _build_profile_cache_spec(
    model: GenericModel, sequences, calibration_sequences, cfg: ComparatorConfig, profile_kind: str
) -> ProfileCacheSpec:
    """Build one cache descriptor for a normalized profile bundle."""
    return {
        "model": model,
        "sequences": sequences,
        "promoters": calibration_sequences,
        "profile_kind": profile_kind,
        "cache_dir": cfg["cache_dir"],
    }


def _resolve_profile_bundle(
    model: GenericModel, sequences, calibration_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Resolve one model to the normalized strand-aware profile bundle used in profile comparisons."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    profile_kind = cfg["profile_normalization"]
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    calibration_fp = _cached_batch_fingerprint(runtime_cache, calibration_sequences, "calibration")
    runtime_key = (fingerprint_model(model), profile_kind, sequence_fp, calibration_fp)

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    cache_spec = None
    if cfg["cache_mode"] == "on":
        cache_spec = _build_profile_cache_spec(model, sequences, calibration_sequences, cfg, profile_kind)
        cached = load_profile_cache(cache_spec)
        if cached is not None:
            runtime_cache[runtime_key] = cached
            logger.debug("Profile cache hit for model '%s'.", model.name)
            return cached

    raw_bundle = _resolve_raw_profile_bundle(model, sequences, runtime_cache)
    normalizer = _fit_profile_normalizer(model, calibration_sequences, cfg, runtime_cache)
    profile_bundle = _apply_profile_normalizer(raw_bundle, normalizer)
    runtime_cache[runtime_key] = profile_bundle

    if cache_spec is not None:
        store_profile_cache(cache_spec, profile_bundle)
        logger.debug("Stored profile cache for model '%s'.", model.name)

    return profile_bundle


def _prepare_profile_model(
    model: GenericModel,
    sequences,
    calibration_sequences,
    cfg: ComparatorConfig,
    score_options: dict,
    runtime_cache: dict | None = None,
):
    """Resolve normalized and prepared profile bundles for one model."""
    bundle = _resolve_profile_bundle(model, sequences, calibration_sequences, cfg, runtime_cache)
    prepared = _prepare_profile_bundle_for_scoring(bundle, float(score_options["min_value"]))
    return bundle, prepared


def _score_prepared_profile_pair(
    prepared_query_bundle: dict,
    prepared_target_bundle: dict,
    score_options: dict,
) -> dict:
    """Score one prepared profile pair and return the best orientation candidate."""
    orientation_scores, orientation_offsets = _score_prepared_profile_orientations(
        prepared_query_bundle,
        prepared_target_bundle,
        [(query_strand, target_strand) for _, query_strand, target_strand in PROFILE_ORIENTATION_PAIRS],
        score_options,
    )
    candidates = []
    for (orientation, _query_strand, target_strand), score, offset in zip(
        PROFILE_ORIENTATION_PAIRS, orientation_scores, orientation_offsets, strict=False
    ):
        candidates.append(
            {
                "orientation": orientation,
                "score": float(score),
                "offset": int(offset),
                "target_strand": int(target_strand),
            }
        )
    return _select_best_orientation(candidates)


def _build_profile_result(query_name: str, target_name: str, best: dict, metric: str) -> ComparisonResult:
    """Build one profile comparison result payload from the best candidate."""
    return {
        "query": query_name,
        "target": target_name,
        "score": float(best["score"]),
        "offset": -int(best["offset"]),
        "orientation": best["orientation"],
        "metric": metric,
    }


def _attach_profile_null_statistics(
    result: dict,
    prepared_query_bundle: dict,
    target_bundle: dict,
    best_target_strand: int,
    score_options: dict,
    cfg: ComparatorConfig,
) -> None:
    """Attach Monte Carlo statistics for one profile comparison result."""
    if cfg["n_permutations"] <= 0:
        return

    obs_score = float(result["score"])
    best_target_profile = profile_view(target_bundle, best_target_strand)

    def surrogate_gen(rng):
        surrogate_bundle = _create_surrogate_bundle(best_target_profile, rng, cfg)
        prepared_surrogate = _prepare_profile_bundle_for_scoring(surrogate_bundle, float(score_options["min_value"]))
        scores, _ = _score_prepared_profile_orientations(
            prepared_query_bundle,
            prepared_surrogate,
            [(PLUS_STRAND, 0), (MINUS_STRAND, 0)],
            score_options,
        )
        return float(np.max(scores, initial=0.0))

    nulls, null_mean, null_std = run_montecarlo(
        lambda value: value,
        surrogate_gen,
        cfg["n_permutations"],
        cfg["seed"],
    )
    _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg["n_permutations"])


def _create_surrogate_bundle(profile, rng: np.random.Generator, cfg: ComparatorConfig):
    """Generate one surrogate single-profile bundle by row-wise convolution and renormalization."""
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

    convolved = np.full(profile["values"].shape, SCORE_PADDING, dtype=np.float32)
    for row_index, length in enumerate(profile["lengths"]):
        current_length = int(length)
        if current_length == 0:
            continue
        segment = profile["values"][row_index, :current_length]
        convolved[row_index, :current_length] = convolve1d(segment, final_kernel, axis=0, mode="constant", cval=0.0)

    strategy = _get_profile_normalization_strategy(cfg["profile_normalization"])
    surrogate = pack_profile_bundle(convolved[None, ...], profile["lengths"], SCORE_PADDING)
    return strategy["normalize_bundle"](surrogate)


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
def strategy_profile(
    model1: GenericModel, model2: GenericModel, sequences, cfg: ComparatorConfig
) -> ComparisonResult:
    """Dense masked profile comparison strategy (CO/Dice similarity)."""
    runtime_cache = {}
    calibration_sequences = _get_profile_calibration_sequences(sequences, cfg)
    score_options = _build_profile_score_options(cfg)
    _query_bundle, prepared_bundle1 = _prepare_profile_model(
        model1, sequences, calibration_sequences, cfg, score_options, runtime_cache
    )
    bundle2, prepared_bundle2 = _prepare_profile_model(
        model2, sequences, calibration_sequences, cfg, score_options, runtime_cache
    )
    best = _score_prepared_profile_pair(prepared_bundle1, prepared_bundle2, score_options)
    result = _build_profile_result(model1.name, model2.name, best, cfg["metric"])
    _attach_profile_null_statistics(
        result,
        prepared_bundle1,
        bundle2,
        int(best["target_strand"]),
        score_options,
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
    query_cache = {}
    results = []
    for target_model in target_models:
        target_cache = {}
        use_pfm_mode = cfg["pfm_mode"] or (query_model.type_key != target_model.type_key)
        _query_matrix, prepared_query = _prepare_motif_model(query_model, sequences, cfg, use_pfm_mode, query_cache)
        target_matrix, prepared_target = _prepare_motif_model(target_model, sequences, cfg, use_pfm_mode, target_cache)
        best = _score_prepared_motif_pair(prepared_query, prepared_target, cfg["metric"])
        result = _build_motif_result(query_model.name, target_model.name, best, cfg["metric"])
        _attach_motif_null_statistics(result, prepared_query, target_matrix, cfg)
        results.append(result)
    return results


def _compare_profile_one_to_many(
    query_model: GenericModel, target_models, sequences, cfg: ComparatorConfig
) -> list[dict]:
    """Compare one profile query against many targets while reusing prepared query state."""
    query_cache = {}
    calibration_sequences = _get_profile_calibration_sequences(sequences, cfg)
    score_options = _build_profile_score_options(cfg)
    _query_bundle, prepared_query = _prepare_profile_model(
        query_model,
        sequences,
        calibration_sequences,
        cfg,
        score_options,
        query_cache,
    )
    results = []
    for target_model in target_models:
        target_cache = {}
        target_bundle, prepared_target = _prepare_profile_model(
            target_model,
            sequences,
            calibration_sequences,
            cfg,
            score_options,
            target_cache,
        )
        best = _score_prepared_profile_pair(prepared_query, prepared_target, score_options)
        result = _build_profile_result(query_model.name, target_model.name, best, cfg["metric"])
        _attach_profile_null_statistics(
            result,
            prepared_query,
            target_bundle,
            int(best["target_strand"]),
            score_options,
            cfg,
        )
        results.append(result)
    return results


def compare(
    model1: GenericModel,
    model2: GenericModel,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    promoters=None,
) -> dict:
    """Main entry point for motif comparison."""
    try:
        strategy_fn = registry[strategy]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}") from exc

    effective_config = dict(config)
    if promoters is not None:
        effective_config["promoters"] = promoters
    with _numba_thread_scope(effective_config.get("numba_threads")):
        return strategy_fn(model1, model2, sequences, effective_config)


def compare_one_to_many(
    query_model: GenericModel,
    target_models,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    promoters=None,
) -> list[dict]:
    """Main entry point for one-vs-many motif comparison."""
    effective_config = dict(config)
    if promoters is not None:
        effective_config["promoters"] = promoters

    with _numba_thread_scope(effective_config.get("numba_threads")):
        if strategy == "profile":
            return _compare_profile_one_to_many(query_model, target_models, sequences, effective_config)
        if strategy == "motif":
            return _compare_motif_one_to_many(query_model, target_models, sequences, effective_config)
        available = ", ".join(sorted(registry))
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")
