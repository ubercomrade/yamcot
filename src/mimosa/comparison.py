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

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import convolve1d

from mimosa.execute import run_motali
from mimosa.functions import (
    _fast_cj_kernel_numba,
    _fast_overlap_kernel_numba,
    _fast_pearson_kernel,
    scores_to_frequencies,
)
from mimosa.io import write_dist, write_fasta
from mimosa.models import GenericModel, calculate_threshold_table, get_pfm, get_score_bounds, scan_model, write_model
from mimosa.ragged import RaggedData


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

    distortion_level: float = 0.4
    search_range: int = 10
    min_kernel_size: int = 3
    max_kernel_size: int = 11

    motali_threshold: float = 0.95
    tmp_directory: str = "."
    fasta_path: Optional[str] = None


def create_comparator_config(**kwargs) -> ComparatorConfig:
    """Factory function for creating ComparatorConfig with backward compatibility."""

    if "motali_tmp_dir" in kwargs and "tmp_directory" not in kwargs:
        kwargs["tmp_directory"] = kwargs.pop("motali_tmp_dir")

    defaults = {
        "metric": "pcc",
        "n_permutations": 0,
        "seed": None,
        "n_jobs": -1,
        "permute_rows": False,
        "pfm_mode": False,
        "distortion_level": 0.4,
        "search_range": 10,
        "min_kernel_size": 3,
        "max_kernel_size": 11,
        "motali_threshold": 0.95,
        "tmp_directory": ".",
    }

    config_params = {**defaults, **kwargs}

    return ComparatorConfig(**config_params)


def _create_surrogate_ragged(frequencies: RaggedData, rng: np.random.Generator, cfg: ComparatorConfig) -> RaggedData:
    """Unified surrogate generation logic (extracted from Data/Universal comparators)."""

    data = frequencies.data
    offsets = frequencies.offsets

    kernel_size = int(rng.integers(cfg.min_kernel_size, cfg.max_kernel_size + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2

    kernel_types = ["smooth", "edge", "double_peak"]
    k_type = str(rng.choice(kernel_types))

    identity_kernel = np.zeros(kernel_size, dtype=np.float32)
    identity_kernel[center] = 1.0

    if k_type == "smooth":
        x = np.linspace(-3, 3, kernel_size)
        base = np.exp(-0.5 * x**2).astype(np.float32)
    elif k_type == "edge":
        base = np.zeros(kernel_size, dtype=np.float32)
        base[max(center - 1, 0)] = -1.0
        base[min(center + 1, kernel_size - 1)] = 1.0
    elif k_type == "double_peak":
        base = np.zeros(kernel_size, dtype=np.float32)
        base[0] = 0.5
        base[-1] = 0.5
        base[center] = -1.0
    else:
        base = identity_kernel.copy()

    noise = rng.normal(0, 1, size=kernel_size).astype(np.float32)
    slope = float(rng.uniform(-1.0, 1.0)) * cfg.distortion_level * 2.0
    gradient = np.linspace(-slope, slope, kernel_size).astype(np.float32)

    distorted_kernel = base + cfg.distortion_level * noise + gradient

    if kernel_size >= 3:
        smooth_filter = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        distorted_kernel = np.convolve(distorted_kernel, smooth_filter, mode="same")

    distorted_kernel /= np.linalg.norm(distorted_kernel) + 1e-8

    alpha = max(0.0, min(1.0, cfg.distortion_level))
    final_kernel = (1.0 - alpha) * identity_kernel + alpha * distorted_kernel
    if rng.uniform() < 0.5:
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

    return scores_to_frequencies(convolved_ragged)


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


@registry.register("tomtom")
def strategy_tomtom(
    model1: GenericModel,
    model2: GenericModel,
    sequences: Optional[RaggedData],
    cfg: ComparatorConfig,
) -> dict:
    """Matrix-based comparison strategy (PCC/ED/Cosine)."""

    def _is_power_of_four(n: int) -> bool:
        """Check whether a value is a power of four."""
        if n < 1:
            return False
        while n % 4 == 0:
            n //= 4
        return n == 1

    def _prep_matrix(mat: np.ndarray):
        """Prepare matrix for comparison by removing N and creating reverse complement."""
        mat = np.asarray(mat)

        if mat.ndim == 2:
            rows, cols = mat.shape
            rows_like_alphabet = rows in (4, 5) or _is_power_of_four(rows)
            cols_like_alphabet = cols in (4, 5) or _is_power_of_four(cols)

            if not rows_like_alphabet and cols_like_alphabet:
                mat = mat.T

            if mat.shape[0] == 5:
                mat = mat[:4, :]

            if mat.shape[0] > 4 and _is_power_of_four(mat.shape[0]):
                order = int(round(np.log(mat.shape[0]) / np.log(4)))
                mat = mat.reshape((4,) * order + (mat.shape[1],))
        else:
            if mat.shape[0] > 5:
                mat = np.moveaxis(mat, 0, -1)
            k = mat.ndim - 1
            clean_slice = tuple(slice(0, 4) if i < k else slice(None) for i in range(mat.ndim))
            mat = mat[clean_slice]

        k = mat.ndim - 1
        rc = mat.copy()
        for i in range(k):
            rc = np.flip(rc, axis=i)
        rc = np.flip(rc, axis=-1)
        return mat.reshape(-1, mat.shape[-1]), rc.reshape(-1, mat.shape[-1])

    def _vectorized_pcc(x1, x2):
        """Vectorized Pearson correlation computation."""
        x1c = x1 - np.mean(x1, axis=0, keepdims=True)
        x2c = x2 - np.mean(x2, axis=0, keepdims=True)
        n = x1c * x2c
        d = np.sqrt(np.sum(x1c**2, axis=0)) * np.sqrt(np.sum(x2c**2, axis=0))
        return np.where(d > 1e-9, np.sum(n, axis=0) / d, 0.0)

    def _vectorized_cosine(x1, x2):
        """Vectorized cosine similarity computation."""
        n = np.sum(x1 * x2, axis=0)
        d = np.linalg.norm(x1, axis=0) * np.linalg.norm(x2, axis=0)
        return np.where(d > 1e-9, n / d, 0.0)

    def _align(a1, a2):
        """Align two matrices and find best score."""
        L1, L2 = a1.shape[1], a2.shape[1]
        best_score, best_off = -np.inf, 0
        min_ov = min(L1, L2) / 2

        if cfg.metric == "ed":
            a1 = (a1 - np.mean(a1, axis=0)) / (np.std(a1, axis=0) + 1e-9)
            a2 = (a2 - np.mean(a2, axis=0)) / (np.std(a2, axis=0) + 1e-9)

        for off in range(-(L2 - 1), L1):
            if off < 0:
                l_ov = min(L1, L2 + off)
                if l_ov < min_ov:
                    continue
                s1, s2 = slice(0, l_ov), slice(-off, -off + l_ov)
            else:
                l_ov = min(L1 - off, L2)
                if l_ov < min_ov:
                    continue
                s1, s2 = slice(off, off + l_ov), slice(0, l_ov)

            c1, c2 = a1[:, s1], a2[:, s2]

            if cfg.metric == "pcc":
                sc = np.sum(_vectorized_pcc(c1, c2)) / l_ov
            elif cfg.metric == "cosine":
                sc = np.sum(_vectorized_cosine(c1, c2)) / l_ov
            else:
                sc = -np.sum(np.sqrt(np.sum((c1 - c2) ** 2, axis=0))) / l_ov

            if sc > best_score:
                best_score, best_off = sc, off
        return best_score, best_off

    def _matrix_shuffle(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Shuffle columns and optionally rows (values) in the original multidimensional matrix.
        """

        shuffled = matrix.copy()

        pos_indices = np.arange(shuffled.shape[-1])
        rng.shuffle(pos_indices)
        shuffled = shuffled[..., pos_indices]

        if cfg.permute_rows:
            alphabet_size = shuffled.shape[0]

            perm = rng.permutation(alphabet_size)

            for axis in range(shuffled.ndim - 1):
                shuffled = np.take(shuffled, perm, axis=axis)

        return shuffled

    use_pfm_mode = cfg.pfm_mode or (model1.type_key != model2.type_key)
    if use_pfm_mode:
        if sequences is None:
            raise ValueError("sequences are required for pfm_mode")
        else:
            matrix1 = get_pfm(model1, sequences)
            matrix2 = get_pfm(model2, sequences)
    else:
        matrix1 = model1.representation
        matrix2 = model2.representation
    f1, _ = _prep_matrix(matrix1)
    f2, f2_rc = _prep_matrix(matrix2)

    s_pp, off_pp = _align(f1, f2)
    s_pm, off_pm = _align(f1, f2_rc)

    if s_pm > s_pp:
        obs_score, off, orient = s_pm, off_pm, "+-"
    else:
        obs_score, off, orient = s_pp, off_pp, "++"

    result = {
        "query": model1.name,
        "target": model2.name,
        "score": float(obs_score),
        "offset": int(off),
        "orientation": orient,
        "metric": cfg.metric,
    }

    if cfg.n_permutations > 0:

        def gen_surrogate(rng):
            """Generate one surrogate score."""
            rnd = matrix2.copy()
            idx = np.arange(rnd.shape[-1])
            rng.shuffle(idx)
            rnd = rnd[..., idx]
            if cfg.permute_rows:
                alphabet_size = rnd.shape[0]
                perm = rng.permutation(alphabet_size)
                for axis in range(rnd.ndim - 1):
                    rnd = np.take(rnd, perm, axis=axis)
            rnd_flat, rnd_rc_flat = _prep_matrix(rnd)
            s1, _ = _align(f1, rnd_flat)
            s2, _ = _align(f1, rnd_rc_flat)
            return max(s1, s2)

        nulls, n_m, n_s = run_montecarlo(lambda x: x, gen_surrogate, cfg.n_permutations, cfg.n_jobs, cfg.seed)

        if nulls.size > 0:
            z = (obs_score - n_m) / (n_s + 1e-9)
            pv = (int(np.sum(nulls >= obs_score)) + 1.0) / (cfg.n_permutations + 1.0)
            result.update({"p-value": float(pv), "z-score": float(z), "null_mean": n_m, "null_std": n_s})

    return result


@registry.register("universal")
def strategy_universal(
    model1: GenericModel, model2: GenericModel, sequences: RaggedData, cfg: ComparatorConfig
) -> dict:
    """RaggedData-based comparison strategy (CJ/CO/Corr)."""
    if sequences is None:
        raise ValueError("Universal strategy requires 'sequences' argument.")

    freq1_plus = scan_model(model1, sequences, "+")
    freq2_plus = scan_model(model2, sequences, "+")
    freq2_minus = scan_model(model2, sequences, "-")

    freq1_plus = scores_to_frequencies(freq1_plus)
    freq2_plus = scores_to_frequencies(freq2_plus)
    freq2_minus = scores_to_frequencies(freq2_minus)

    def get_score(S1: RaggedData, S2: RaggedData) -> Tuple[float, int]:
        """Compute score and offset for two profiles."""
        if cfg.metric == "cj":
            sc, off = _fast_cj_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, cfg.search_range)
            return sc, off
        elif cfg.metric == "co":
            sc, off = _fast_overlap_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, cfg.search_range)
            return sc, off
        elif cfg.metric == "corr":
            sc, _, off = _fast_pearson_kernel(S1.data, S1.offsets, S2.data, S2.offsets, cfg.search_range)
            return sc, off
        else:
            raise ValueError(f"Unknown metric: {cfg.metric}")

    sc_pp, off_pp = get_score(freq1_plus, freq2_plus)
    sc_pm, off_pm = get_score(freq1_plus, freq2_minus)

    if sc_pm > sc_pp:
        obs_score, off, orient, f_final = sc_pm, off_pm, "+-", freq2_minus
    else:
        obs_score, off, orient, f_final = sc_pp, off_pp, "++", freq2_plus

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
            sc, _ = get_score(freq1_plus, surr)
            return sc

        nulls, n_m, n_s = run_montecarlo(lambda x: x, surrogate_gen, cfg.n_permutations, cfg.n_jobs, cfg.seed)

        if nulls.size > 0:
            z = (obs_score - n_m) / (n_s + 1e-9)
            pv = (int(np.sum(nulls >= obs_score)) + 1.0) / (cfg.n_permutations + 1.0)
            result.update({"p-value": float(pv), "z-score": float(z), "null_mean": n_m, "null_std": n_s})

    return result


@registry.register("motali")
def strategy_motali(
    model1: GenericModel, model2: GenericModel, sequences: Optional[RaggedData], cfg: ComparatorConfig
) -> dict:
    """External Motali tool wrapper."""
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

        dist1 = calculate_threshold_table(model1, sequences)
        dist2 = calculate_threshold_table(model2, sequences)
        min_1, max_1 = get_score_bounds(model1)
        min_2, max_2 = get_score_bounds(model2)

        write_dist(dist1, max_1, min_1, d1_path)
        write_dist(dist2, max_2, min_2, d2_path)

        fasta_path = cfg.fasta_path
        if fasta_path is None and sequences is not None:
            fasta_path = os.path.join(tmp, "sequences.fa")
            write_fasta(sequences, fasta_path)

        if fasta_path is None:
            pass

        score = run_motali(
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
        )

        return {"query": model1.name, "target": model2.name, "score": score}


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

    return strategy_fn(model1, model2, sequences, config)
