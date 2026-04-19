"""Functional motif comparison workflows."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import convolve1d

from mimosa.batches import SCORE_PADDING, batch_with_values, flatten_valid
from mimosa.cache import fingerprint_batch, load_profile_cache, store_profile_cache
from mimosa.execute import run_motali
from mimosa.functions import (
    _prepare_profile_bundle_for_scoring,
    apply_score_log_tail_table,
    build_profile_score_options,
    build_score_log_tail_table,
    fast_profile_score_orientations,
    normalize_empirical_log_tail_pair,
    scores_to_empirical_log_tail,
)
from mimosa.io import write_dist, write_fasta
from mimosa.models import (
    GenericModel,
    calculate_threshold_table,
    get_pfm,
    get_score_bounds,
    scan_model_strands,
    write_model,
)
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

PROFILE_NORMALIZATION_REGISTRY = {
    "empirical_log_tail": {
        "fit": build_score_log_tail_table,
        "apply": apply_score_log_tail_table,
        "normalize_surrogate": scores_to_empirical_log_tail,
    },
}

registry: dict[str, Callable] = {}


def _register_comparison_strategy(name: str):
    """Register one comparison strategy."""

    def decorator(fn):
        registry[name] = fn
        return fn

    return decorator


def create_comparator_config(**kwargs) -> dict:
    """Build one validated comparison options dictionary."""
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
        "fasta_path": None,
        "cache_mode": "off",
        "cache_dir": ".mimosa-cache",
        "promoters": None,
    }
    config = {**defaults, **kwargs}
    min_kernel_size, max_kernel_size = validate_kernel_size_range(config["min_kernel_size"], config["max_kernel_size"])
    config["min_kernel_size"] = min_kernel_size
    config["max_kernel_size"] = max_kernel_size
    config["min_logfpr"] = validate_non_negative("min_logfpr", config.get("min_logfpr"))
    config["profile_normalization"] = validate_profile_normalization(
        config.get("profile_normalization", "empirical_log_tail"),
        PROFILE_NORMALIZATION_REGISTRY,
    )
    config["pfm_top_fraction"] = validate_pfm_top_fraction(config.get("pfm_top_fraction"))
    config["cache_mode"] = validate_cache_mode(config.get("cache_mode", "off"))
    return config


def _select_best_orientation(candidates):
    """Choose the highest-scoring orientation with deterministic tie-breaking."""
    return max(
        candidates, key=lambda candidate: (float(candidate["score"]), -ORIENTATION_TIEBREAK[candidate["orientation"]])
    )


def _get_profile_calibration_sequences(sequences, cfg: dict):
    """Return the sequence collection used to fit profile normalization."""
    return cfg.get("promoters") if cfg.get("promoters") is not None else sequences


def _get_profile_normalization_strategy(name: str) -> dict:
    """Resolve one registered profile-normalization strategy."""
    try:
        return PROFILE_NORMALIZATION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILE_NORMALIZATION_REGISTRY))
        raise ValueError(f"Unsupported profile normalization: {name}. Available: {available}") from exc


def _resolve_raw_profile_scores(model: GenericModel, sequences, strand: str, runtime_cache: dict | None = None):
    """Resolve one raw score profile before normalization."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = fingerprint_batch(sequences) or "no-sequences"
    runtime_key = ("raw_scores", model.name, strand, sequence_fp)
    if runtime_key in runtime_cache:
        return runtime_cache[runtime_key]

    if strand not in {"+", "-"}:
        raise ValueError(f"Unsupported strand for profile scoring: {strand}")

    plus_scores, minus_scores = _resolve_raw_profile_score_pair(model, sequences, runtime_cache)
    return plus_scores if strand == "+" else minus_scores


def _resolve_raw_profile_score_pair(model: GenericModel, sequences, runtime_cache: dict | None = None):
    """Resolve raw profile scores for both strands, caching the pair together."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = fingerprint_batch(sequences) or "no-sequences"
    plus_key = ("raw_scores", model.name, "+", sequence_fp)
    minus_key = ("raw_scores", model.name, "-", sequence_fp)

    cached_plus = runtime_cache.get(plus_key)
    cached_minus = runtime_cache.get(minus_key)
    if cached_plus is not None and cached_minus is not None:
        return cached_plus, cached_minus

    if model.type_key != "scores" and sequences is None:
        raise ValueError("Profile strategy requires sequences when comparing motif models.")

    plus_scores, minus_scores = scan_model_strands(model, sequences)
    runtime_cache[plus_key] = plus_scores
    runtime_cache[minus_key] = minus_scores
    return plus_scores, minus_scores


def _fit_profile_normalizer(model: GenericModel, calibration_sequences, cfg: dict, runtime_cache: dict | None = None):
    """Fit normalization parameters from the calibration score sample."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    calibration_fp = fingerprint_batch(calibration_sequences) or "no-calibration"
    runtime_key = ("profile_normalizer", model.name, cfg["profile_normalization"], calibration_fp)
    if runtime_key in runtime_cache:
        return runtime_cache[runtime_key]

    calibration_plus, calibration_minus = _resolve_raw_profile_score_pair(model, calibration_sequences, runtime_cache)
    calibration_sample = np.concatenate((flatten_valid(calibration_plus), flatten_valid(calibration_minus)))
    strategy = _get_profile_normalization_strategy(cfg["profile_normalization"])
    normalizer = (cfg["profile_normalization"], strategy["fit"](calibration_sample))
    runtime_cache[runtime_key] = normalizer
    return normalizer


def _apply_profile_normalizer(scores, normalizer):
    """Apply fitted normalization parameters to one score profile."""
    strategy_name, params = normalizer
    strategy = _get_profile_normalization_strategy(strategy_name)
    return strategy["apply"](scores, params)


def _build_profile_cache_spec(
    model: GenericModel, sequences, calibration_sequences, cfg: dict, strand: str, profile_kind: str
) -> dict:
    """Build one cache descriptor for a normalized profile."""
    return {
        "model": model,
        "sequences": sequences,
        "promoters": calibration_sequences,
        "strand": strand,
        "profile_kind": profile_kind,
        "cache_dir": cfg["cache_dir"],
    }


def _resolve_profile_signal(
    model: GenericModel, sequences, calibration_sequences, cfg: dict, strand: str, runtime_cache=None
):
    """Resolve a model to the normalized profile signal used in profile comparisons."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    profile_kind = cfg["profile_normalization"]
    sequence_fp = fingerprint_batch(sequences) or "no-sequences"
    calibration_fp = fingerprint_batch(calibration_sequences) or "no-calibration"
    runtime_key = (model.name, profile_kind, strand, sequence_fp, calibration_fp)

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    cache_spec = None
    if cfg["cache_mode"] == "on":
        cache_spec = _build_profile_cache_spec(model, sequences, calibration_sequences, cfg, strand, profile_kind)
        cached = load_profile_cache(cache_spec)
        if cached is not None:
            runtime_cache[runtime_key] = cached
            logger.debug("Profile cache hit for model '%s' (%s strand).", model.name, strand)
            return cached

    scores = _resolve_raw_profile_scores(model, sequences, strand, runtime_cache)
    normalizer = _fit_profile_normalizer(model, calibration_sequences, cfg, runtime_cache)
    profile = _apply_profile_normalizer(scores, normalizer)
    runtime_cache[runtime_key] = profile

    if cache_spec is not None:
        store_profile_cache(cache_spec, profile)
        logger.debug("Stored profile cache for model '%s' (%s strand).", model.name, strand)

    return profile


def _resolve_profile_bundle(model: GenericModel, sequences, calibration_sequences, cfg: dict) -> dict:
    """Resolve normalized profile signals for both strands."""
    profile_kind = cfg["profile_normalization"]
    if profile_kind == "empirical_log_tail" and sequences is calibration_sequences:
        plus_cache_spec = None
        minus_cache_spec = None
        if cfg["cache_mode"] == "on":
            plus_cache_spec = _build_profile_cache_spec(model, sequences, calibration_sequences, cfg, "+", profile_kind)
            minus_cache_spec = _build_profile_cache_spec(
                model, sequences, calibration_sequences, cfg, "-", profile_kind
            )
            cached_plus = load_profile_cache(plus_cache_spec)
            cached_minus = load_profile_cache(minus_cache_spec)
            if cached_plus is not None and cached_minus is not None:
                logger.debug("Profile cache hit for model '%s' (both strands).", model.name)
                return {"plus": cached_plus, "minus": cached_minus}

        runtime_cache: dict = {}
        raw_plus, raw_minus = _resolve_raw_profile_score_pair(model, sequences, runtime_cache)
        normalized_plus, normalized_minus = normalize_empirical_log_tail_pair(raw_plus, raw_minus)

        if plus_cache_spec is not None and minus_cache_spec is not None:
            store_profile_cache(plus_cache_spec, normalized_plus)
            store_profile_cache(minus_cache_spec, normalized_minus)
            logger.debug("Stored profile cache for model '%s' (both strands).", model.name)

        return {
            "plus": normalized_plus,
            "minus": normalized_minus,
        }

    runtime_cache: dict = {}
    return {
        "plus": _resolve_profile_signal(model, sequences, calibration_sequences, cfg, "+", runtime_cache),
        "minus": _resolve_profile_signal(model, sequences, calibration_sequences, cfg, "-", runtime_cache),
    }


def _create_surrogate_batch(profile, rng: np.random.Generator, cfg: dict):
    """Generate one surrogate profile batch by row-wise convolution and renormalization."""
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

    surrogate = batch_with_values(profile, convolved, padding_value=SCORE_PADDING)
    strategy = _get_profile_normalization_strategy(cfg["profile_normalization"])
    return strategy["normalize_surrogate"](surrogate)


def run_montecarlo(
    obs_score_func: Callable, surrogate_generator_func: Callable, n_permutations: int, n_jobs: int, seed, *args
):
    """Run a Monte Carlo workflow in parallel."""
    if n_permutations <= 0:
        return np.array([]), 0.0, 0.0

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31, size=n_permutations)

    def worker(seed_value):
        rng = np.random.default_rng(int(seed_value))
        surrogate = surrogate_generator_func(rng, *args)
        return obs_score_func(surrogate)

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(seed_value) for seed_value in seeds)
    null_scores = np.array([result for result in results if result is not None], dtype=np.float32)
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


def _resolve_motif_matrices(
    model1: GenericModel, model2: GenericModel, sequences, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the motif representations used for direct alignment."""
    use_pfm_mode = cfg["pfm_mode"] or (model1.type_key != model2.type_key)
    if not use_pfm_mode:
        return model1.representation, model2.representation
    if sequences is None:
        raise ValueError("sequences are required for pfm_mode")
    return (
        get_pfm(model1, sequences, top_fraction=cfg["pfm_top_fraction"]),
        get_pfm(model2, sequences, top_fraction=cfg["pfm_top_fraction"]),
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


@_register_comparison_strategy("motif")
def strategy_motif(model1: GenericModel, model2: GenericModel, sequences, cfg: dict) -> dict:
    """Matrix-based comparison strategy (PCC/ED/Cosine)."""
    matrix1, matrix2 = _resolve_motif_matrices(model1, model2, sequences, cfg)
    prepared1 = _prepare_motif(matrix1)
    prepared2 = _prepare_motif(matrix2)
    best = _select_best_orientation(_score_motif_candidates(prepared1, prepared2, cfg["metric"]))
    obs_score = float(best["score"])

    result = {
        "query": model1.name,
        "target": model2.name,
        "score": obs_score,
        "offset": int(best["offset"]),
        "orientation": best["orientation"],
        "metric": cfg["metric"],
    }

    if cfg["n_permutations"] > 0:

        def gen_surrogate(rng):
            surrogate = _permute_motif_matrix(matrix2, rng, cfg["permute_rows"])
            return _best_motif_score(prepared1, _prepare_motif(surrogate), cfg["metric"])

        nulls, null_mean, null_std = run_montecarlo(
            lambda value: value,
            gen_surrogate,
            cfg["n_permutations"],
            cfg["n_jobs"],
            cfg["seed"],
        )
        _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg["n_permutations"])

    return result


@_register_comparison_strategy("profile")
def strategy_profile(model1: GenericModel, model2: GenericModel, sequences, cfg: dict) -> dict:
    """Dense masked profile comparison strategy (CO/Dice similarity)."""
    calibration_sequences = _get_profile_calibration_sequences(sequences, cfg)
    bundle1 = _resolve_profile_bundle(model1, sequences, calibration_sequences, cfg)
    bundle2 = _resolve_profile_bundle(model2, sequences, calibration_sequences, cfg)
    score_options = build_profile_score_options(
        search_range=cfg["search_range"],
        min_value=0.0 if cfg["min_logfpr"] is None else float(cfg["min_logfpr"]),
        metric=cfg["metric"],
    )
    prepared_bundle1 = _prepare_profile_bundle_for_scoring(bundle1, float(score_options["min_value"]))
    prepared_bundle2 = _prepare_profile_bundle_for_scoring(bundle2, float(score_options["min_value"]))

    orientation_pairs = (
        ("++", prepared_bundle1["plus"], prepared_bundle2["plus"], bundle2["plus"]),
        ("--", prepared_bundle1["minus"], prepared_bundle2["minus"], bundle2["minus"]),
        ("+-", prepared_bundle1["plus"], prepared_bundle2["minus"], bundle2["minus"]),
        ("-+", prepared_bundle1["minus"], prepared_bundle2["plus"], bundle2["plus"]),
    )
    profile_pairs = [(query_profile, target_profile) for _, query_profile, target_profile, _ in orientation_pairs]
    orientation_scores, orientation_offsets = fast_profile_score_orientations(profile_pairs, score_options)

    candidates = []
    for (orientation, _query_profile, _target_profile, target_profile_raw), score, offset in zip(
        orientation_pairs, orientation_scores, orientation_offsets, strict=False
    ):
        candidates.append(
            {
                "orientation": orientation,
                "score": float(score),
                "offset": int(offset),
                "target_profile_raw": target_profile_raw,
            }
        )

    best = _select_best_orientation(candidates)
    obs_score = float(best["score"])
    motif_offset = -int(best["offset"])

    result = {
        "query": model1.name,
        "target": model2.name,
        "score": obs_score,
        "offset": motif_offset,
        "orientation": best["orientation"],
        "metric": cfg["metric"],
    }

    if cfg["n_permutations"] > 0:
        best_target_profile = best["target_profile_raw"]
        prepared_query_plus = prepared_bundle1["plus"]
        prepared_query_minus = prepared_bundle1["minus"]

        def surrogate_gen(rng):
            surrogate = _create_surrogate_batch(best_target_profile, rng, cfg)
            scores, _ = fast_profile_score_orientations(
                [(prepared_query_plus, surrogate), (prepared_query_minus, surrogate)],
                score_options,
            )
            return float(np.max(scores, initial=0.0))

        nulls, null_mean, null_std = run_montecarlo(
            lambda value: value,
            surrogate_gen,
            cfg["n_permutations"],
            cfg["n_jobs"],
            cfg["seed"],
        )
        _update_result_with_null_statistics(result, obs_score, (nulls, null_mean, null_std), cfg["n_permutations"])

    return result


@_register_comparison_strategy("motali")
def strategy_motali(model1: GenericModel, model2: GenericModel, sequences, cfg: dict) -> dict:
    """External Motali tool wrapper."""
    threshold_sequences = cfg.get("promoters") if cfg.get("promoters") is not None else sequences
    if threshold_sequences is None:
        raise ValueError("Motali strategy requires 'promoters' or 'sequences' for threshold table calculation.")

    with tempfile.TemporaryDirectory(dir=cfg["tmp_directory"], ignore_cleanup_errors=True) as tmp:
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

        fasta_path = cfg.get("fasta_path")
        if fasta_path is None and sequences is not None:
            fasta_path = os.path.join(tmp, "sequences.fa")
            write_fasta(sequences, fasta_path)

        if fasta_path is None:
            raise ValueError("Motali strategy requires 'sequences' or comparator.fasta_path for FASTA input.")

        score, offset, orientation = run_motali(
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
            shift=cfg["motali_shift"],
            err=cfg["motali_err"],
        )

        return {
            "query": model1.name,
            "target": model2.name,
            "score": score,
            "offset": int(offset),
            "orientation": orientation,
        }


def compare(
    model1: GenericModel, model2: GenericModel, strategy: str, config: dict, sequences=None, promoters=None
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
    return strategy_fn(model1, model2, sequences, effective_config)
