"""
Functional Comparison Module
============================

This module provides a functional programming approach to motif comparison,
replacing class hierarchies with pure functions and registry-based polymorphism.

Key Features:
- Immutable configuration using frozen dataclasses
- Registry-based strategy pattern using decorators
- Pure functions for all comparison operations
- Elimination of code duplication between different comparator types
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import convolve1d

from mimosa.cache import ProfileCacheSpec, fingerprint_ragged, load_profile_cache, store_profile_cache
from mimosa.execute import run_motali
from mimosa.functions import (
    ProfileScoreOptions,
    apply_score_log_tail_table,
    build_profile_support,
    build_score_log_tail_table,
    fast_profile_score,
    scores_to_empirical_log_tail,
)
from mimosa.io import write_dist, write_fasta
from mimosa.models import (
    GenericModel,
    calculate_threshold_table,
    get_pfm,
    get_score_bounds,
    scan_model,
    write_model,
)
from mimosa.ragged import RaggedData
from mimosa.validation import (
    validate_cache_mode,
    validate_kernel_size_range,
    validate_non_negative,
    validate_pfm_top_fraction,
    validate_profile_normalization,
)

logger = logging.getLogger(__name__)
ORIENTATION_TIEBREAK = {"++": 0, "+-": 1, "-+": 2, "--": 3}
NUCLEOTIDE_CARDINALITY = 4
AMBIGUOUS_STATE_CARDINALITY = 5
MATRIX_RANK = 2
SIMILARITY_EPS = 1e-9
SURROGATE_SMOOTH_FILTER = np.array([0.25, 0.5, 0.25], dtype=np.float32)
SURROGATE_SIGN_FLIP_PROBABILITY = 0.5


@dataclass(frozen=True)
class ProfileNormalizationStrategy:
    """One profile-normalization strategy.

    The strategy owns three pieces of behavior:
    - fitting normalization parameters from a calibration score sample
    - applying those fitted parameters to raw scores
    - re-normalizing surrogate profiles used in Monte Carlo nulls
    """

    fit: Callable[[np.ndarray], Any]
    apply: Callable[[RaggedData, Any], RaggedData]
    normalize_surrogate: Callable[[RaggedData], RaggedData]


PROFILE_NORMALIZATION_REGISTRY = {
    "empirical_log_tail": ProfileNormalizationStrategy(
        fit=build_score_log_tail_table,
        apply=apply_score_log_tail_table,
        normalize_surrogate=scores_to_empirical_log_tail,
    ),
}


@dataclass(frozen=True)
class ComparatorConfig:
    """Immutable configuration for comparison strategies.

    This dataclass enforces dependency injection by requiring all
    parameters to be explicitly passed. The frozen=True makes it
    hashable and thread-safe.
    """

    metric: str
    n_permutations: int = 0
    seed: Optional[int] = None
    n_jobs: int = -1

    permute_rows: bool = False
    pfm_mode: bool = False
    pfm_top_fraction: Optional[float] = 0.05

    distortion_level: float = 0.4
    search_range: int = 10
    min_kernel_size: int = 3
    max_kernel_size: int = 11
    min_logfpr: Optional[float] = None
    profile_normalization: str = "empirical_log_tail"

    motali_err: float = 0.002
    motali_shift: int = 50
    tmp_directory: str = "."
    fasta_path: Optional[str] = None
    cache_mode: str = "off"
    cache_dir: str = ".mimosa-cache"
    promoters: Optional[RaggedData] = dc_field(default=None, compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class ProfileResolutionSpec:
    """Inputs required to resolve one normalized profile signal."""

    model: GenericModel
    sequences: Optional[RaggedData]
    calibration_sequences: Optional[RaggedData]
    strand: str


@dataclass(frozen=True)
class StrandProfileBundle:
    """Resolved profile signals and optional sparse supports for both strands."""

    plus: RaggedData
    minus: RaggedData
    plus_support: Optional[tuple[np.ndarray, np.ndarray]]
    minus_support: Optional[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PreparedMotif:
    """Forward and reverse-complement views of one comparison matrix."""

    forward: np.ndarray
    reverse: np.ndarray


def create_comparator_config(**kwargs) -> ComparatorConfig:
    """Factory function for creating ComparatorConfig."""

    defaults = {
        "metric": "pcc",
        "n_permutations": 0,
        "seed": None,
        "n_jobs": -1,
        "permute_rows": False,
        "pfm_mode": False,
        "pfm_top_fraction": 0.05,
        "distortion_level": 0.4,
        "search_range": 10,
        "min_kernel_size": 3,
        "max_kernel_size": 11,
        "min_logfpr": None,
        "profile_normalization": "empirical_log_tail",
        "motali_err": 0.002,
        "motali_shift": 50,
        "tmp_directory": ".",
        "cache_mode": "off",
        "cache_dir": ".mimosa-cache",
    }

    config_params = {**defaults, **kwargs}

    min_kernel_size, max_kernel_size = validate_kernel_size_range(
        config_params["min_kernel_size"],
        config_params["max_kernel_size"],
    )
    config_params["min_kernel_size"] = min_kernel_size
    config_params["max_kernel_size"] = max_kernel_size
    config_params["min_logfpr"] = validate_non_negative("min_logfpr", config_params.get("min_logfpr"))
    config_params["profile_normalization"] = validate_profile_normalization(
        config_params.get("profile_normalization", "empirical_log_tail"),
        PROFILE_NORMALIZATION_REGISTRY,
    )
    config_params["pfm_top_fraction"] = validate_pfm_top_fraction(config_params.get("pfm_top_fraction"))
    config_params["cache_mode"] = validate_cache_mode(config_params.get("cache_mode", "off"))

    return ComparatorConfig(**config_params)


def _select_best_orientation(candidates):
    """Choose the highest-scoring orientation with deterministic tie-breaking.

    The returned offset is always interpreted in the coordinate system of the
    oriented query/target pair encoded by the orientation label.
    """

    return max(
        candidates,
        key=lambda candidate: (float(candidate["score"]), -ORIENTATION_TIEBREAK[candidate["orientation"]]),
    )


def _get_profile_calibration_sequences(
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> Optional[RaggedData]:
    """Return the sequence collection used to fit profile normalization."""
    return cfg.promoters if cfg.promoters is not None else sequences


def _get_profile_normalization_strategy(name: str) -> ProfileNormalizationStrategy:
    """Resolve one registered profile-normalization strategy."""
    try:
        return PROFILE_NORMALIZATION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILE_NORMALIZATION_REGISTRY))
        raise ValueError(f"Unsupported profile normalization: {name}. Available: {available}") from exc


def _resolve_raw_profile_scores(
    model: GenericModel,
    sequences: Optional[RaggedData],
    strand: str,
    runtime_cache: Optional[dict] = None,
) -> RaggedData:
    """Resolve one raw score profile before normalization."""
    sequence_fp = fingerprint_ragged(sequences) or "no-sequences"
    runtime_key = ("raw_scores", strand, sequence_fp)
    runtime_cache = {} if runtime_cache is None else runtime_cache

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    if model.type_key == "scores":
        scores = scan_model(model, None, strand)
    else:
        if sequences is None:
            raise ValueError("Profile strategy requires sequences when comparing motif models.")
        scores = scan_model(model, sequences, strand)

    runtime_cache[runtime_key] = scores
    return scores


def _fit_profile_normalizer(
    model: GenericModel,
    calibration_sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
    runtime_cache: Optional[dict] = None,
):
    """Fit normalization parameters from the calibration score sample."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    calibration_fp = fingerprint_ragged(calibration_sequences) or "no-calibration"
    runtime_key = ("profile_normalizer", cfg.profile_normalization, calibration_fp)

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    calibration_plus = _resolve_raw_profile_scores(model, calibration_sequences, "+", runtime_cache)
    calibration_minus = _resolve_raw_profile_scores(model, calibration_sequences, "-", runtime_cache)
    calibration_sample = np.concatenate((calibration_plus.data, calibration_minus.data))
    strategy = _get_profile_normalization_strategy(cfg.profile_normalization)
    normalizer = (cfg.profile_normalization, strategy.fit(calibration_sample))

    runtime_cache[runtime_key] = normalizer
    return normalizer


def _apply_profile_normalizer(scores: RaggedData, normalizer) -> RaggedData:
    """Apply fitted normalization parameters to one score profile."""
    strategy_name, params = normalizer
    strategy = _get_profile_normalization_strategy(strategy_name)
    return strategy.apply(scores, params)


def _resolve_profile_signal(
    spec_or_model,
    sequences_or_cfg,
    calibration_sequences: Optional[RaggedData] = None,
    cfg: Optional[ComparatorConfig] = None,
    strand: Optional[str] = None,
    runtime_cache: Optional[dict] = None,
) -> RaggedData:
    """Resolve a model to the profile signal used in profile comparisons."""
    if isinstance(spec_or_model, ProfileResolutionSpec):
        spec = spec_or_model
        if not isinstance(sequences_or_cfg, ComparatorConfig):
            raise TypeError("ComparatorConfig is required when resolving a ProfileResolutionSpec.")
        cfg = sequences_or_cfg
    else:
        if cfg is None or strand is None:
            raise TypeError("cfg and strand are required when resolving a profile from raw arguments.")
        spec = ProfileResolutionSpec(spec_or_model, sequences_or_cfg, calibration_sequences, strand)

    profile_kind = cfg.profile_normalization
    sequence_fp = fingerprint_ragged(spec.sequences) or "no-sequences"
    calibration_fp = fingerprint_ragged(spec.calibration_sequences) or "no-calibration"
    runtime_key = (profile_kind, spec.strand, sequence_fp, calibration_fp)
    runtime_cache = {} if runtime_cache is None else runtime_cache

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    use_disk_cache = cfg.cache_mode == "on"
    if use_disk_cache:
        cache_spec = ProfileCacheSpec(
            model=spec.model,
            sequences=spec.sequences,
            promoters=spec.calibration_sequences,
            strand=spec.strand,
            profile_kind=profile_kind,
            cache_dir=cfg.cache_dir,
        )
        cached = load_profile_cache(cache_spec)
        if cached is not None:
            runtime_cache[runtime_key] = cached
            logger.debug("Profile cache hit for model '%s' (%s strand).", spec.model.name, spec.strand)
            return cached

    scores = _resolve_raw_profile_scores(spec.model, spec.sequences, spec.strand, runtime_cache)
    normalizer = _fit_profile_normalizer(spec.model, spec.calibration_sequences, cfg, runtime_cache)
    profile = _apply_profile_normalizer(scores, normalizer)
    runtime_cache[runtime_key] = profile

    if use_disk_cache:
        store_profile_cache(cache_spec, profile)
        logger.debug("Stored profile cache for model '%s' (%s strand).", spec.model.name, spec.strand)

    return profile


def _resolve_profile_support(
    profile: RaggedData,
    cfg: ComparatorConfig,
    strand: str,
    runtime_cache: Optional[dict] = None,
):
    """Resolve cached sparse support for thresholded profile comparisons."""
    if cfg.min_logfpr is None:
        return None

    min_value = float(cfg.min_logfpr)
    if min_value <= 0.0:
        return None

    runtime_key = ("support", np.float32(min_value), strand)
    runtime_cache = {} if runtime_cache is None else runtime_cache

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    support = build_profile_support(profile.data, profile.offsets, min_value)
    runtime_cache[runtime_key] = support
    return support


def _resolve_profile_bundle(
    model: GenericModel,
    sequences: Optional[RaggedData],
    calibration_sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> StrandProfileBundle:
    """Resolve normalized profile signals and sparse support for both strands."""
    runtime_cache: dict = {}
    plus_spec = ProfileResolutionSpec(model, sequences, calibration_sequences, "+")
    minus_spec = ProfileResolutionSpec(model, sequences, calibration_sequences, "-")
    plus = _resolve_profile_signal(plus_spec, cfg, runtime_cache)
    minus = _resolve_profile_signal(minus_spec, cfg, runtime_cache)
    plus_support = _resolve_profile_support(plus, cfg, "+", runtime_cache)
    minus_support = _resolve_profile_support(minus, cfg, "-", runtime_cache)
    return StrandProfileBundle(plus=plus, minus=minus, plus_support=plus_support, minus_support=minus_support)


def _create_surrogate_ragged(frequencies: RaggedData, rng: np.random.Generator, cfg: ComparatorConfig) -> RaggedData:
    """Surrogate generation: random kernel + delta-spike at a random offset, then smoothing + normalization."""
    data = frequencies.data
    offsets = frequencies.offsets

    min_kernel_size = int(cfg.min_kernel_size)
    max_kernel_size = int(cfg.max_kernel_size)
    first_odd = min_kernel_size if min_kernel_size % 2 == 1 else min_kernel_size + 1
    n_odd = ((max_kernel_size - first_odd) // 2) + 1
    kernel_size = int(first_odd + 2 * int(rng.integers(0, n_odd)))
    center = kernel_size // 2

    identity_kernel = np.zeros(kernel_size, dtype=np.float32)
    identity_kernel[center] = 1.0
    alpha = float(np.clip(cfg.distortion_level, 0.0, 1.0))
    kernel = rng.normal(0.0, 1.0, size=kernel_size).astype(np.float32)
    if kernel_size >= len(SURROGATE_SMOOTH_FILTER):
        kernel = np.convolve(kernel, SURROGATE_SMOOTH_FILTER, mode="same").astype(np.float32)
    kernel /= np.linalg.norm(kernel) + 1e-8
    final_kernel = (1.0 - alpha) * identity_kernel + alpha * kernel
    if rng.uniform() < SURROGATE_SIGN_FLIP_PROBABILITY:
        final_kernel = -final_kernel
    final_kernel /= np.linalg.norm(final_kernel) + 1e-8

    convolved = np.empty_like(data, dtype=np.float32)
    for i in range(len(offsets) - 1):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        segment = data[start:end]
        if segment.size == 0:
            continue
        convolved[start:end] = convolve1d(segment, final_kernel, axis=0, mode="constant", cval=0.0).astype(np.float32)

    convolved_ragged = RaggedData(data=convolved, offsets=offsets)
    strategy = _get_profile_normalization_strategy(cfg.profile_normalization)
    return strategy.normalize_surrogate(convolved_ragged)


def run_montecarlo(
    obs_score_func: Callable,
    surrogate_generator_func: Callable,
    n_permutations: int,
    n_jobs: int,
    seed: Optional[int],
    *args,
) -> Tuple[np.ndarray, float, float]:
    """Generic function to run Monte Carlo permutations in parallel."""
    if n_permutations <= 0:
        return np.array([]), 0.0, 0.0

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31, size=n_permutations)

    def worker(seed_val):
        """Compute one score for a permutation seed."""
        rng = np.random.default_rng(int(seed_val))
        surrogate = surrogate_generator_func(rng, *args)
        return obs_score_func(surrogate)

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(s) for s in seeds)

    null_scores = np.array([r for r in results if r is not None], dtype=np.float32)

    return (null_scores, float(np.mean(null_scores)), float(np.std(null_scores)))


def _update_result_with_null_statistics(
    result: dict,
    obs_score: float,
    stats: tuple[np.ndarray, float, float],
    n_permutations: int,
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


class ComparatorRegistry:
    """Registry for comparison strategies using decorator pattern."""

    def __init__(self):
        """Initialize registry state."""
        self._strategies: Dict[str, Callable] = {}

    def register(self, name: str):
        """Register a callable under the given name."""

        def decorator(fn):
            """Store a callable in the registry."""
            self._strategies[name] = fn
            return fn

        return decorator

    def get(self, name: str):
        """Return a registered callable by name."""
        return self._strategies.get(name)


registry = ComparatorRegistry()


def _is_power_of_four(value: int) -> bool:
    """Return True when the provided value is a positive power of four."""
    if value < 1:
        return False
    while value % NUCLEOTIDE_CARDINALITY == 0:
        value //= NUCLEOTIDE_CARDINALITY
    return value == 1


def _looks_like_alphabet_axis(size: int) -> bool:
    """Return True when an axis likely encodes nucleotide states."""
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


def _prepare_motif(matrix: np.ndarray) -> PreparedMotif:
    """Build flattened forward and reverse-complement views for alignment."""
    normalized = _normalize_motif_tensor(matrix)
    reverse = _reverse_complement_motif_tensor(normalized)
    return PreparedMotif(
        forward=normalized.reshape(-1, normalized.shape[-1]),
        reverse=reverse.reshape(-1, reverse.shape[-1]),
    )


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


def _score_motif_candidates(
    query: PreparedMotif, target: PreparedMotif, metric: str
) -> list[dict[str, float | int | str]]:
    """Score all orientation pairs for one prepared motif pair."""
    candidates = []
    for orientation, query_matrix, target_matrix in (
        ("++", query.forward, target.forward),
        ("+-", query.forward, target.reverse),
        ("-+", query.reverse, target.forward),
        ("--", query.reverse, target.reverse),
    ):
        score, offset = _align_motif_matrices(query_matrix, target_matrix, metric)
        candidates.append({"orientation": orientation, "score": score, "offset": offset})
    return candidates


def _best_motif_score(query: PreparedMotif, target: PreparedMotif, metric: str) -> float:
    """Return the best orientation score for one motif pair."""
    return float(_select_best_orientation(_score_motif_candidates(query, target, metric))["score"])


def _resolve_motif_matrices(
    model1: GenericModel,
    model2: GenericModel,
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the motif representations used for direct alignment."""
    use_pfm_mode = cfg.pfm_mode or (model1.type_key != model2.type_key)
    if not use_pfm_mode:
        return model1.representation, model2.representation
    if sequences is None:
        raise ValueError("sequences are required for pfm_mode")
    return (
        get_pfm(model1, sequences, top_fraction=cfg.pfm_top_fraction),
        get_pfm(model2, sequences, top_fraction=cfg.pfm_top_fraction),
    )


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


@registry.register("motif")
def strategy_motif(
    model1: GenericModel,
    model2: GenericModel,
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> dict:
    """Matrix-based comparison strategy (PCC/ED/Cosine)."""
    matrix1, matrix2 = _resolve_motif_matrices(model1, model2, sequences, cfg)
    prepared1 = _prepare_motif(matrix1)
    prepared2 = _prepare_motif(matrix2)
    best = _select_best_orientation(_score_motif_candidates(prepared1, prepared2, cfg.metric))
    obs_score = float(best["score"])

    result = {
        "query": model1.name,
        "target": model2.name,
        "score": obs_score,
        "offset": int(best["offset"]),
        "orientation": best["orientation"],
        "metric": cfg.metric,
    }

    if cfg.n_permutations > 0:

        def gen_surrogate(rng):
            """Generate one surrogate score."""
            surrogate = _permute_motif_matrix(matrix2, rng, cfg.permute_rows)
            return _best_motif_score(prepared1, _prepare_motif(surrogate), cfg.metric)

        nulls, null_mean, null_std = run_montecarlo(
            lambda value: value,
            gen_surrogate,
            cfg.n_permutations,
            cfg.n_jobs,
            cfg.seed,
        )
        _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg.n_permutations)

    return result


@registry.register("profile")
def strategy_profile(
    model1: GenericModel,
    model2: GenericModel,
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> dict:
    """RaggedData-based comparison strategy (CO/Dice similarity)."""
    calibration_sequences = _get_profile_calibration_sequences(sequences, cfg)
    bundle1 = _resolve_profile_bundle(model1, sequences, calibration_sequences, cfg)
    bundle2 = _resolve_profile_bundle(model2, sequences, calibration_sequences, cfg)
    score_options = ProfileScoreOptions(
        search_range=cfg.search_range,
        min_value=0.0 if cfg.min_logfpr is None else float(cfg.min_logfpr),
        metric=cfg.metric,
    )

    def get_score(S1: RaggedData, S2: RaggedData, support1=None, support2=None) -> Tuple[float, int]:
        """Compute score and offset for two profiles."""
        sc, off = fast_profile_score(S1, S2, score_options, supports=(support1, support2))
        return sc, off

    candidates = []
    for orient, query_profile, target_profile, query_support, target_support in (
        ("++", bundle1.plus, bundle2.plus, bundle1.plus_support, bundle2.plus_support),
        ("--", bundle1.minus, bundle2.minus, bundle1.minus_support, bundle2.minus_support),
        ("+-", bundle1.plus, bundle2.minus, bundle1.plus_support, bundle2.minus_support),
        ("-+", bundle1.minus, bundle2.plus, bundle1.minus_support, bundle2.plus_support),
    ):
        score, off = get_score(query_profile, target_profile, query_support, target_support)
        candidates.append(
            {
                "orientation": orient,
                "score": score,
                "offset": off,
                "query_profile": query_profile,
                "query_support": query_support,
                "target_profile": target_profile,
            }
        )

    best = _select_best_orientation(candidates)
    obs_score = best["score"]
    off = best["offset"]
    orient = best["orientation"]
    f_final = best["target_profile"]

    # fast_profile_score uses the opposite shift sign convention; flip it so the
    # reported offset remains "target start relative to query start" for the
    # oriented pair encoded in `orientation`.
    motif_offset = -off

    result = {
        "query": model1.name,
        "target": model2.name,
        "score": float(obs_score),
        "offset": int(motif_offset),
        "orientation": orient,
        "metric": cfg.metric,
    }

    if cfg.n_permutations > 0:

        def surrogate_gen(rng):
            """Generate one surrogate score."""
            surr = _create_surrogate_ragged(f_final, rng, cfg)
            sc_plus, _ = get_score(bundle1.plus, surr, bundle1.plus_support, None)
            sc_minus, _ = get_score(bundle1.minus, surr, bundle1.minus_support, None)
            return max(sc_plus, sc_minus)

        nulls, null_mean, null_std = run_montecarlo(
            lambda value: value,
            surrogate_gen,
            cfg.n_permutations,
            cfg.n_jobs,
            cfg.seed,
        )
        _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg.n_permutations)

    return result


@registry.register("motali")
def strategy_motali(
    model1: GenericModel,
    model2: GenericModel,
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> dict:
    """External Motali tool wrapper."""
    threshold_sequences = cfg.promoters if cfg.promoters is not None else sequences
    if threshold_sequences is None:
        raise ValueError("Motali strategy requires 'promoters' or 'sequences' for threshold table calculation.")

    with tempfile.TemporaryDirectory(dir=cfg.tmp_directory, ignore_cleanup_errors=True) as tmp:
        ext_1 = ".pfm" if model1.type_key == "pwm" else ".mat"
        ext_2 = ".pfm" if model2.type_key == "pwm" else ".mat"

        if model1.type_key == "sitega":
            ext_1 = ".mat"
        if model2.type_key == "sitega":
            ext_2 = ".mat"

        m1_path = os.path.join(tmp, f"motif_1{ext_1}")
        m2_path = os.path.join(tmp, f"motif_2{ext_2}")
        d1_path = os.path.join(tmp, "thresholds_1.dist")
        d2_path = os.path.join(tmp, "thresholds_2.dist")

        write_model(model1, m1_path)
        write_model(model2, m2_path)

        dist1 = calculate_threshold_table(model1, threshold_sequences)
        dist2 = calculate_threshold_table(model2, threshold_sequences)
        min_1, max_1 = get_score_bounds(model1)
        min_2, max_2 = get_score_bounds(model2)

        write_dist(dist1, max_1, min_1, d1_path)
        write_dist(dist2, max_2, min_2, d2_path)

        fasta_path = cfg.fasta_path
        if fasta_path is None and sequences is not None:
            fasta_path = os.path.join(tmp, "sequences.fa")
            write_fasta(sequences, fasta_path)

        if fasta_path is None:
            raise ValueError("Motali strategy requires 'sequences' or comparator.fasta_path for FASTA input.")

        score, offset, orient = run_motali(
            fasta_path,
            m1_path,
            m2_path,
            model1.type_key,
            model2.type_key,
            d1_path,
            d2_path,
            os.path.join(tmp, "overlap.txt"),
            os.path.join(tmp, "all.txt"),
            os.path.join(tmp, "prc_pass.txt"),
            os.path.join(tmp, "hist_pass.txt"),
            os.path.join(tmp, "sta.txt"),
            shift=cfg.motali_shift,
            err=cfg.motali_err,
        )

        return {
            "query": model1.name,
            "target": model2.name,
            "score": score,
            "offset": int(offset),
            "orientation": orient,
        }


def compare(
    model1: GenericModel,
    model2: GenericModel,
    strategy: str,
    config: ComparatorConfig,
    sequences: Optional[RaggedData] = None,
    promoters: Optional[RaggedData] = None,
) -> dict:
    """Main entry point for motif comparison."""
    strategy_fn = registry.get(strategy)
    if not strategy_fn:
        available = list(registry._strategies.keys())
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")

    effective_config = config if promoters is None else replace(config, promoters=promoters)
    return strategy_fn(model1, model2, sequences, effective_config)
