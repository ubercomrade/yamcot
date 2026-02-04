"""
comparison
==========

Implementations of motif comparison metrics.  Comparing motifs is
useful for identifying similar patterns discovered in different datasets
or cross‑validation folds.  This module defines a common
interface for comparison algorithms and several concrete
implementations
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import convolve1d

from mimosa.execute import run_motali
from mimosa.functions import (
    _fast_cj_kernel_numba,
    _fast_overlap_kernel_numba,
    _fast_pearson_kernel,
    pfm_to_pwm,
    scores_to_frequencies,
)
from mimosa.io import write_fasta
from mimosa.models import MotifModel, RaggedScores
from mimosa.ragged import RaggedData, ragged_from_list


class GeneralMotifComparator(ABC):
    """
    Abstract base class for motif comparators.

    This class defines the common interface for all motif comparison algorithms.
    Concrete implementations should inherit from this class and implement
    the compare method.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the comparator.

        Parameters
        ----------
        name : str
            Name of the comparator instance.
        """
        self.name = name

    @abstractmethod
    def compare(
        self,
        motif_1: MotifModel,
        motif_2: MotifModel,
        sequences: RaggedData | None = None,
    ) -> dict:
        """
        Compare motifs from two collections.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        motifs_1 : List[MotifModel]
            First collection of motifs to compare.
        motifs_2 : List[MotifModel]
            Second collection of motifs to compare.
        sequences : RaggedData or None
            Sequences for frequency calculation (if needed by the implementation).

        Returns
        -------
        pd.DataFrame
            DataFrame containing comparison results.
        """
        raise NotImplementedError


class TomtomComparator(GeneralMotifComparator):
    """
    Comparator for motifs using Euclidean Distance (ED) or Pearson Correlation (PCC).
    Includes Monte Carlo p-value estimation.
    """

    def __init__(
        self,
        metric: str = "pcc",
        n_permutations: int = 1000,
        permute_rows: bool = False,
        pfm_mode: bool = False,
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize comparator.

        Parameters
        ----------
        metric : str
            'pcc' or 'ed'.
        n_permutations : int
            Number of Monte Carlo permutations for p-value calculation.
            Set to 0 to disable.
        permute_rows : bool
            If True, shuffles values within each column (destroys nucleotide structure).
            If False, only shuffles columns (positions).
        n_jobs : int
            Number of parallel jobs. -1 to use all cores.
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(name=f"TomtomComparator_{metric.upper()}")
        self.metric = metric.lower()
        if self.metric not in ["pcc", "ed", "cosine"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'pcc', 'ed', or 'cosine'.")

        self.n_permutations = n_permutations
        self.permute_rows = permute_rows
        self.pfm_mode = pfm_mode
        self.n_jobs = n_jobs
        self.seed = seed

    def _prepare_matrix(self, matrix: np.ndarray):
        """
        Удаляет 'N', создает Reverse Complement для k-меров и преобразует в (4^k, L).
        """
        # Определяем структуру осей
        # Предполагаем, что позиция L — это последняя ось.
        # Если L первая (L, 4, 4...), перемещаем её в конец для удобства.
        if matrix.shape[0] > 5:
            matrix = np.moveaxis(matrix, 0, -1)

        k = matrix.ndim - 1  # Порядок k-мера (1 для моно, 2 для ди, 3 для три)
        L = matrix.shape[-1]

        # Удаляем 'N' (индекс 4) по всем нуклеотидным осям
        # Создаем динамический срез: (slice(0,4), slice(0,4), ..., slice(None))
        clean_slice = tuple(slice(0, 4) if i < k else slice(None) for i in range(matrix.ndim))
        matrix = matrix[clean_slice]

        # Создание Reverse Complement (RC)
        # Complement: Инвертируем каждую нуклеотидную ось (A<->T, C<->G)
        # В алфавите [A, C, G, T] это делается простым flip по оси
        rc_matrix = matrix.copy()
        for axis in range(k):
            rc_matrix = np.flip(rc_matrix, axis=axis)

        rc_matrix = np.flip(rc_matrix, axis=-1)

        flat_matrix = matrix.reshape(-1, L)
        flat_rc_matrix = rc_matrix.reshape(-1, L)

        return flat_matrix, flat_rc_matrix

    def _randomize_matrix(self, matrix: np.ndarray, rng: np.random.Generator):
        """
        Shuffle columns and optionally rows (values) in the original multidimensional matrix.

        This function implements a surrogate generation procedure where the nucleotide
        structure can be partially or completely destroyed depending on the permute_rows setting.

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix to randomize.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        np.ndarray
            Randomized matrix with shuffled columns and optionally rows.
        """
        # Work with a copy of the full dimensionality
        shuffled = matrix.copy()

        # 1. Shuffle columns (positions) along the last axis
        # Indices for the last axis
        pos_indices = np.arange(shuffled.shape[-1])
        rng.shuffle(pos_indices)
        shuffled = shuffled[..., pos_indices]

        if self.permute_rows:
            # 1. Определяем размер алфавита (обычно 4 для A, C, G, T)
            alphabet_size = shuffled.shape[0]
            # 2. Генерируем одну общую перестановку для всех осей
            perm = rng.permutation(alphabet_size)

            # 3. Применяем перестановку ко всем осям, кроме последней (позиции)
            # Это сохраняет структуру зависимостей (например, AA перейдет в GG)
            for axis in range(shuffled.ndim - 1):
                shuffled = np.take(shuffled, perm, axis=axis)

        return shuffled

    def _vectorized_pcc(self, m1: np.ndarray, m2: np.ndarray):
        """
        Compute vectorized Pearson Correlation Coefficient between columns of m1 and m2.

        Parameters
        ----------
        m1 : np.ndarray
            Shape (4, L1) matrix representing first motif
        m2 : np.ndarray
            Shape (4, L2) matrix representing second motif

        Returns
        -------
        correlations : np.ndarray
            Array of correlations between corresponding columns
        """
        # Center both matrices by subtracting column means
        m1_centered = m1 - np.mean(m1, axis=0, keepdims=True)
        m2_centered = m2 - np.mean(m2, axis=0, keepdims=True)

        # Compute standard deviations for normalization
        m1_stds = np.sqrt(np.sum(m1_centered**2, axis=0))
        m2_stds = np.sqrt(np.sum(m2_centered**2, axis=0))

        # Handle zero-variance columns by setting std to 1 (will result in 0 correlation)
        m1_stds = np.where(m1_stds == 0, 1, m1_stds)
        m2_stds = np.where(m2_stds == 0, 1, m2_stds)

        # Compute dot product between centered matrices
        numerator = np.sum(m1_centered * m2_centered, axis=0)

        # Compute correlations
        denominators = m1_stds * m2_stds
        correlations = np.where(denominators != 0, numerator / denominators, 0.0)

        return correlations

    def _vectorized_cosine(self, m1: np.ndarray, m2: np.ndarray):
        """
        Compute vectorized Cosine Similarity between columns of m1 and m2.

        Parameters
        ----------
        m1 : np.ndarray
            Shape (N, L) matrix (N=4 для моно, 16 для динуклеотидов)
        m2 : np.ndarray
            Shape (N, L) matrix

        Returns
        -------
        similarities : np.ndarray
            Array of cosine similarities between corresponding columns
        """
        # 1. Вычисляем скалярное произведение (числитель)
        # Суммируем по оси строк (нуклеотидов)
        numerator = np.sum(m1 * m2, axis=0)

        # 2. Вычисляем L2-нормы (длины векторов) для каждой колонки
        norm1 = np.sqrt(np.sum(m1**2, axis=0))
        norm2 = np.sqrt(np.sum(m2**2, axis=0))

        # 3. Обработка нулевых векторов (чтобы не делить на 0)
        # Если норма 0, значит в колонке все веса 0.
        denominators = norm1 * norm2

        # 4. Вычисляем сходство
        # Где знаменатель > 0, делим. Где 0 — возвращаем 0.0
        similarities = np.where(denominators > 1e-9, numerator / denominators, 0.0)

        return similarities

    def _align_motifs(self, m1: np.ndarray, m2: np.ndarray):
        """
        Align two motifs by sliding one along the other and computing the best score.

        Parameters
        ----------
        m1 : np.ndarray
            First motif matrix of shape (4, L1).
        m2 : np.ndarray
            Second motif matrix of shape (4, L2).

        Returns
        -------
        tuple
            Tuple containing (best_score, best_offset) where:
            best_score : Best alignment score found.
            best_offset : Offset at which best score occurs.
        """
        L1 = m1.shape[1]
        L2 = m2.shape[1]

        # Z-norm
        if self.metric == "ed":
            m1 = (m1 - np.mean(m1, axis=0)) / (np.std(m1, axis=0) + 1e-9)
            m2 = (m2 - np.mean(m2, axis=0)) / (np.std(m2, axis=0) + 1e-9)

        best_score = -np.inf if self.metric == "ed" else -np.inf
        best_offset = 0

        min_offset = -(L2 - 1)
        max_offset = L1 - 1
        min_overlap = min(L2, L1) / 2

        for offset in range(min_offset, max_offset + 1):
            if offset < 0:
                len_overlap = min(L1, L2 + offset)
                if len_overlap < min_overlap:
                    continue
                s1, s2 = slice(0, len_overlap), slice(-offset, -offset + len_overlap)
            else:
                len_overlap = min(L1 - offset, L2)
                if len_overlap < min_overlap:
                    continue
                s1, s2 = slice(offset, offset + len_overlap), slice(0, len_overlap)

            cols1, cols2 = m1[:, s1], m2[:, s2]

            if self.metric == "ed":
                # Compute sum of column-wise Euclidean distances
                # This is the sum of ||col1_i - col2_i|| for each column pair
                column_distances = np.sqrt(np.sum((cols1 - cols2) ** 2, axis=0))
                current_score = -np.sum(column_distances) / len_overlap
            elif self.metric == "pcc":
                # Use vectorized PCC computation
                correlations = self._vectorized_pcc(cols1, cols2)
                current_score = np.sum(correlations) / len_overlap
            elif self.metric == "cosine":
                # Use vectorized PCC computation
                correlations = self._vectorized_cosine(cols1, cols2)
                current_score = np.sum(correlations) / len_overlap
            else:
                # Euclidean distances
                column_distances = np.sqrt(np.sum((cols1 - cols2) ** 2, axis=0))
                current_score = -np.sum(column_distances) / len_overlap
            if current_score > best_score:
                best_score = current_score
                best_offset = offset

        return best_score, best_offset

    def _run_single_permutation(self, m1_flat: np.ndarray, m2_orig_matrix: np.ndarray, seed: int):
        """
        Worker function for parallel execution.

        Generates one surrogate for m2 and compares it with m1.

        Parameters
        ----------
        m1_flat : np.ndarray
            Flattened version of the first motif matrix.
        m2_orig_matrix : np.ndarray
            Original matrix for the second motif (before flattening).
        seed : int
            Random seed for this permutation.

        Returns
        -------
        float
            Maximum alignment score between m1 and the randomized m2.
        """
        rng = np.random.default_rng(seed)

        # 1. Randomize the original m2 matrix (full dimensionality)
        m2_rand_matrix = self._randomize_matrix(m2_orig_matrix, rng)

        # 2. Prepare randomized matrix (flatten + rc)
        m2_rand_flat, m2_rand_rc_flat = self._prepare_matrix(m2_rand_matrix)

        # 3. Compare
        score_pp, _ = self._align_motifs(m1_flat, m2_rand_flat)
        score_pm, _ = self._align_motifs(m1_flat, m2_rand_rc_flat)

        return max(score_pp, score_pm)

    def compare(
        self,
        motif_1: MotifModel,
        motif_2: MotifModel,
        sequences: RaggedData | None = None,
    ) -> dict:
        """
        Compare two motif models with optional p-value calculation.

        Parameters
        ----------
        motif_1 : MotifModel
            First motif model to compare.
        motif_2 : MotifModel
            Second motif model to compare.
        sequences : RaggedData or None
            Sequences for comparison (required if pfm_mode is True).

        Returns
        -------
        dict
            Dictionary containing comparison results.
        """

        if self.pfm_mode:
            if sequences is None:
                raise ValueError("sequences are required for pfm_mode")
            m1_flat, _ = self._prepare_matrix(pfm_to_pwm(motif_1.get_pfm(sequences, top_fraction=0.1)))
            (
                m2_flat,
                m2_rc_flat,
            ) = self._prepare_matrix(pfm_to_pwm(motif_2.get_pfm(sequences, top_fraction=0.1)))
        else:
            m1_flat, _ = self._prepare_matrix(motif_1.matrix)
            m2_flat, m2_rc_flat = self._prepare_matrix(motif_2.matrix)

        # --- Observed Score ---
        obs_score_pp, obs_off_pp = self._align_motifs(m1_flat, m2_flat)
        obs_score_pm, obs_off_pm = self._align_motifs(m1_flat, m2_rc_flat)

        if obs_score_pm > obs_score_pp:
            obs_score = obs_score_pm
            obs_offset = obs_off_pm
            orientation = "+-"
        else:
            obs_score = obs_score_pp
            obs_offset = obs_off_pp
            orientation = "++"

        result = {
            "query": motif_1.name,
            "target": motif_2.name,
            "score": float(obs_score),
            "offset": int(obs_offset),
            "orientation": orientation,
            "metric": self.metric,
        }

        # --- Monte Carlo Permutations ---
        if self.n_permutations > 0:
            base_rng = np.random.default_rng(self.seed)
            seeds = base_rng.integers(0, 2**31, size=self.n_permutations)

            m2 = pfm_to_pwm(motif_2.pfm) if self.pfm_mode else motif_2.matrix
            # Run in parallel
            null_scores = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._run_single_permutation)(m1_flat, m2, int(seeds[i])) for i in range(self.n_permutations)
            )

            null_scores = np.array(null_scores)

            # --- P-value calculation ---
            mean_null = np.mean(null_scores)
            std_null = np.std(null_scores)
            z_score = (obs_score - mean_null) / (std_null + 1e-9)

            n_ge = int(np.sum(null_scores >= obs_score))
            p_value = (n_ge + 1.0) / (self.n_permutations + 1.0)
            # p_value = stats.norm.sf(abs(z_score))

            result.update(
                {
                    "p-value": float(p_value),
                    "z-score": float(z_score),
                    "null_mean": float(mean_null),
                    "null_std": float(std_null),
                }
            )

        return result


class DataComparator:
    """
    Comparator implementation using Jaccard or Overlap metrics
    with permutation-based statistics, working directly with RaggedData objects.
    """

    def __init__(
        self,
        name: str = "DataComparator",
        metric: str = "cj",
        n_permutations: int = 1000,
        distortion_level: float = 0.4,
        n_jobs: int = -1,
        seed: Optional[int] = None,
        search_range: int = 10,
        min_kernel_size: int = 3,
        max_kernel_size: int = 11,
    ) -> None:
        self.name = name
        self.metric = metric
        self.n_permutations = n_permutations
        self.distortion_level = distortion_level
        self.n_jobs = n_jobs
        self.seed = seed
        self.search_range = search_range
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    @staticmethod
    def _compute_metric_internal(S1: RaggedData, S2: RaggedData, search_range: int, metric: str):
        """Internal dispatcher for metric computation kernels."""
        if metric == "cj":
            # Returns (best_cj, best_offset)
            return _fast_cj_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        elif metric == "co":
            # Returns (best_co, best_offset)
            return _fast_overlap_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        elif metric == "corr":
            # Returns (correlation, p-value, best_offset)
            return _fast_pearson_kernel(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _single_compare(self, profile1: RaggedData, profile2: RaggedData):
        """Perform a single comparison between two motifs."""
        # Observed scores for both orientations
        obs_res = self._compute_metric_internal(profile1, profile2, self.search_range, self.metric)

        orientation = "."
        obs_score = float(obs_res[0])
        obs_offset = int(obs_res[-1])  # Offset is always the last element

        result = {"score": obs_score, "offset": obs_offset, "orientation": orientation, "metric": self.metric}

        if self.n_permutations > 0:
            base_rng = np.random.default_rng(self.seed)
            seeds = base_rng.integers(0, 2**31, size=self.n_permutations)

            # Use MotifComparator's surrogate logic for 'cj' and 'co'
            # For 'corr', we use the same surrogate logic if permutations are requested
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._compute_surrogate_score)(
                    profile1,
                    profile2,
                    np.random.default_rng(int(seeds[i])),
                )
                for i in range(self.n_permutations)
            )

            null_scores = np.array([r[0] for r in results if r is not None], dtype=np.float32)
            null_mean = float(np.mean(null_scores))
            null_std = float(np.std(null_scores))

            z_score = (obs_score - null_mean) / (null_std + 1e-9)
            n_ge = int(np.sum(null_scores >= obs_score))
            p_value = (n_ge + 1.0) / (self.n_permutations + 1.0)
            # p_value = stats.norm.sf(abs(z_score))

            result.update(
                {
                    "p-value": p_value,
                    "z-score": z_score,
                    "null_mean": null_mean,
                    "null_std": null_std,
                }
            )

        return result

    def _compute_surrogate_score(self, freq1: RaggedData, freq2: RaggedData, rng: np.random.Generator):
        """Helper for parallel permutation execution."""
        # Reusing the static method from MotifComparator as requested by "preserving algorithmic nuances"
        surrogate = self._generate_single_surrogate(
            freq2,
            rng,
            min_kernel_size=self.min_kernel_size,
            max_kernel_size=self.max_kernel_size,
            distortion_level=self.distortion_level,
        )
        return self._compute_metric_internal(freq1, surrogate, self.search_range, self.metric)

    @staticmethod
    def _generate_single_surrogate(
        frequencies: RaggedData,
        rng: np.random.Generator,
        min_kernel_size: int = 3,
        max_kernel_size: int = 11,
        distortion_level: float = 1.0,
    ) -> RaggedData:
        """
        Generate a single surrogate frequency profile using convolution with a distorted kernel.

        This function implements a sophisticated surrogate generation algorithm that creates
        distorted versions of the input frequency profiles. The "distortion" logic refers to
        how the identity kernel is systematically modified through several techniques:

        1. Base kernel selection (smooth, edge, double_peak patterns)
        2. Noise addition with controlled amplitude
        3. Gradient application to introduce directional bias
        4. Smoothing to reduce artifacts
        5. Convex combination with identity kernel based on distortion level
        6. Sign flipping for additional variation

        Parameters
        ----------
        frequencies : RaggedData
            Input frequency profile to generate surrogate from.
        rng : np.random.Generator
            Random number generator instance.
        min_kernel_size : int, optional
            Minimum size of the convolution kernel (default is 3).
        max_kernel_size : int, optional
            Maximum size of the convolution kernel (default is 11).
        distortion_level : float, optional
            Level of distortion to apply (0.0 to 1.0, default is 1.0).

        Returns
        -------
        RaggedData
            Surrogate frequency profile generated from the input.
        """
        # For simplicity in surrogate generation, we use dense adapter
        dense_adapter = RaggedScores.from_numba(frequencies)
        X = dense_adapter.values
        lengths = dense_adapter.lengths

        kernel_size = int(rng.integers(min_kernel_size, max_kernel_size + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        center = kernel_size // 2

        kernel_types = ["smooth", "edge", "double_peak"]
        kernel_type = str(rng.choice(kernel_types))

        identity_kernel = np.zeros(kernel_size, dtype=np.float32)
        identity_kernel[center] = 1.0

        if kernel_type == "smooth":
            x = np.linspace(-3, 3, kernel_size)
            base = np.exp(-0.5 * x**2).astype(np.float32)
        elif kernel_type == "edge":
            base = np.zeros(kernel_size, dtype=np.float32)
            base[max(center - 1, 0)] = -1.0
            base[min(center + 1, kernel_size - 1)] = 1.0
        elif kernel_type == "double_peak":
            base = np.zeros(kernel_size, dtype=np.float32)
            base[0] = 0.5
            base[-1] = 0.5
            base[center] = -1.0
        else:
            base = identity_kernel.copy()

        noise = rng.normal(0, 1, size=kernel_size).astype(np.float32)
        slope = float(rng.uniform(-1.0, 1.0)) * distortion_level * 2.0
        gradient = np.linspace(-slope, slope, kernel_size).astype(np.float32)

        distorted_kernel = base + distortion_level * noise + gradient

        if kernel_size >= 3:
            smooth_filter = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            distorted_kernel = np.convolve(distorted_kernel, smooth_filter, mode="same")

        distorted_kernel /= np.linalg.norm(distorted_kernel) + 1e-8

        alpha = max(0.0, min(1.0, distortion_level))
        final_kernel = (1.0 - alpha) * identity_kernel + alpha * distorted_kernel
        if rng.uniform() < 0.5:
            final_kernel = -final_kernel
        final_kernel /= np.linalg.norm(final_kernel) + 1e-8

        convolved = convolve1d(X, final_kernel, axis=1, mode="constant", cval=0.0).astype(np.float32)

        # Convert back to RaggedData
        convolved_list = [convolved[i, : lengths[i]] for i in range(len(lengths))]
        convolved_ragged = ragged_from_list(convolved_list, dtype=np.float32)

        return scores_to_frequencies(convolved_ragged)

    def compare(
        self,
        profile1: RaggedData,
        profile2: RaggedData,
    ) -> dict | None:
        """
        Compare two RaggedData objects directly.
        """

        # Calculate comparison metrics
        out = self._single_compare(
            profile1=profile1,
            profile2=profile2,
        )

        # Create generic identifiers for the data
        result = {"query": "Data1", "target": "Data2"}
        result.update(out)

        return result


class UniversalMotifComparator(GeneralMotifComparator):
    """
    Universal comparator implementation that integrates functionality from both
    MotifComparator and CorrelationComparator. Supports Jaccard ('cj'),
    Overlap ('co'), and Pearson Correlation ('corr') metrics with optional
    permutation-based statistics.
    """

    def __init__(
        self,
        name: str = "UniversalMotifComparator",
        metric: str = "cj",
        n_permutations: int = 1000,
        distortion_level: float = 0.4,
        n_jobs: int = -1,
        seed: Optional[int] = None,
        min_kernel_size: int = 3,
        max_kernel_size: int = 11,
        search_range: int = 10,
    ) -> None:
        """
        Initialize the unified comparator.

        Parameters
        ----------
        name : str
            Name of the comparator instance.
        metric : str
            Similarity metric to use: 'cj' (Continuous Jaccard),
            'co' (Continuous Overlap), or 'corr' (Pearson Correlation).
        n_permutations : int
            Number of permutations for statistical significance testing.
        distortion_level : float
            Level of distortion for surrogate generation (used for 'cj' and 'co').
        n_jobs : int
            Number of parallel jobs for permutations.
        seed : int, optional
            Random seed for reproducibility.
        search_range : int
            Range to search for optimal offset alignment.
        """
        super().__init__(name)
        self.metric = metric.lower()
        if self.metric not in ["cj", "co", "corr"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cj', 'co', or 'corr'.")

        self.n_permutations = n_permutations
        self.distortion_level = distortion_level
        self.n_jobs = n_jobs
        self.seed = seed
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.search_range = search_range

    @staticmethod
    def _compute_metric_internal(S1: RaggedData, S2: RaggedData, search_range: int, metric: str):
        """Internal dispatcher for metric computation kernels."""
        if metric == "cj":
            return _fast_cj_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        elif metric == "co":
            return _fast_overlap_kernel_numba(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        elif metric == "corr":
            # Returns (correlation, p-value, offset)
            return _fast_pearson_kernel(S1.data, S1.offsets, S2.data, S2.offsets, search_range)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _single_compare(self, motif1: MotifModel, motif2: MotifModel, sequences: RaggedData):
        """Perform a single comparison between two motifs."""
        freq1_plus = motif1.get_frequencies(sequences, strand="+")
        freq2_plus = motif2.get_frequencies(sequences, strand="+")
        freq2_minus = motif2.get_frequencies(sequences, strand="-")

        # Observed scores for both orientations
        res_pp = self._compute_metric_internal(freq1_plus, freq2_plus, self.search_range, self.metric)
        res_pm = self._compute_metric_internal(freq1_plus, freq2_minus, self.search_range, self.metric)

        # Extract scores for comparison (first element of return tuple for all kernels)
        score_pp = res_pp[0]
        score_pm = res_pm[0]

        if score_pm > score_pp:
            orientation = "+-"
            obs_res = res_pm
            freq1, freq2 = freq1_plus, freq2_minus
        else:
            orientation = "++"
            obs_res = res_pp
            freq1, freq2 = freq1_plus, freq2_plus

        obs_score = float(obs_res[0])
        obs_offset = int(obs_res[-1])  # Offset is always the last element

        result = {"score": obs_score, "offset": obs_offset, "orientation": orientation, "metric": self.metric}

        if self.n_permutations > 0:
            base_rng = np.random.default_rng(self.seed)
            seeds = base_rng.integers(0, 2**31, size=self.n_permutations)

            # Use MotifComparator's surrogate logic for 'cj' and 'co'
            # For 'corr', we use the same surrogate logic if permutations are requested
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._compute_surrogate_score)(
                    freq1,
                    freq2,
                    np.random.default_rng(int(seeds[i])),
                )
                for i in range(self.n_permutations)
            )

            null_scores = np.array([r[0] for r in results if r is not None], dtype=np.float32)
            null_mean = float(np.mean(null_scores))
            null_std = float(np.std(null_scores))

            z_score = (obs_score - null_mean) / (null_std + 1e-9)
            n_ge = int(np.sum(null_scores >= obs_score))
            p_value = (n_ge + 1.0) / (self.n_permutations + 1.0)
            # p_value = stats.norm.sf(abs(z_score))

            result.update(
                {
                    "p-value": p_value,
                    "z-score": z_score,
                    "null_mean": null_mean,
                    "null_std": null_std,
                }
            )

        return result

    def _compute_surrogate_score(self, freq1: RaggedData, freq2: RaggedData, rng: np.random.Generator):
        """Helper for parallel permutation execution."""
        # Reusing the static method from MotifComparator as requested by "preserving algorithmic nuances"
        surrogate = self._generate_single_surrogate(
            freq2,
            rng,
            min_kernel_size=self.min_kernel_size,
            max_kernel_size=self.max_kernel_size,
            distortion_level=self.distortion_level,
        )
        return self._compute_metric_internal(freq1, surrogate, self.search_range, self.metric)

    @staticmethod
    def _generate_single_surrogate(
        frequencies: RaggedData,
        rng: np.random.Generator,
        min_kernel_size: int = 3,
        max_kernel_size: int = 11,
        distortion_level: float = 1.0,
    ) -> RaggedData:
        """
        Generate a single surrogate frequency profile using convolution with a distorted kernel.

        This function implements a sophisticated surrogate generation algorithm that creates
        distorted versions of the input frequency profiles. The "distortion" logic refers to
        how the identity kernel is systematically modified through several techniques:

        1. Base kernel selection (smooth, edge, double_peak patterns)
        2. Noise addition with controlled amplitude
        3. Gradient application to introduce directional bias
        4. Smoothing to reduce artifacts
        5. Convex combination with identity kernel based on distortion level
        6. Sign flipping for additional variation

        Parameters
        ----------
        frequencies : RaggedData
            Input frequency profile to generate surrogate from.
        rng : np.random.Generator
            Random number generator instance.
        min_kernel_size : int, optional
            Minimum size of the convolution kernel (default is 3).
        max_kernel_size : int, optional
            Maximum size of the convolution kernel (default is 11).
        distortion_level : float, optional
            Level of distortion to apply (0.0 to 1.0, default is 1.0).

        Returns
        -------
        RaggedData
            Surrogate frequency profile generated from the input.
        """
        # For simplicity in surrogate generation, we use dense adapter
        dense_adapter = RaggedScores.from_numba(frequencies)
        X = dense_adapter.values
        lengths = dense_adapter.lengths

        kernel_size = int(rng.integers(min_kernel_size, max_kernel_size + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        center = kernel_size // 2

        kernel_types = ["smooth", "edge", "double_peak"]
        kernel_type = str(rng.choice(kernel_types))

        identity_kernel = np.zeros(kernel_size, dtype=np.float32)
        identity_kernel[center] = 1.0

        if kernel_type == "smooth":
            x = np.linspace(-3, 3, kernel_size)
            base = np.exp(-0.5 * x**2).astype(np.float32)
        elif kernel_type == "edge":
            base = np.zeros(kernel_size, dtype=np.float32)
            base[max(center - 1, 0)] = -1.0
            base[min(center + 1, kernel_size - 1)] = 1.0
        elif kernel_type == "double_peak":
            base = np.zeros(kernel_size, dtype=np.float32)
            base[0] = 0.5
            base[-1] = 0.5
            base[center] = -1.0
        else:
            base = identity_kernel.copy()

        noise = rng.normal(0, 1, size=kernel_size).astype(np.float32)
        slope = float(rng.uniform(-1.0, 1.0)) * distortion_level * 2.0
        gradient = np.linspace(-slope, slope, kernel_size).astype(np.float32)

        distorted_kernel = base + distortion_level * noise + gradient

        if kernel_size >= 3:
            smooth_filter = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            distorted_kernel = np.convolve(distorted_kernel, smooth_filter, mode="same")

        distorted_kernel /= np.linalg.norm(distorted_kernel) + 1e-8

        alpha = max(0.0, min(1.0, distortion_level))
        final_kernel = (1.0 - alpha) * identity_kernel + alpha * distorted_kernel
        if rng.uniform() < 0.5:
            final_kernel = -final_kernel
        final_kernel /= np.linalg.norm(final_kernel) + 1e-8

        convolved = convolve1d(X, final_kernel, axis=1, mode="constant", cval=0.0).astype(np.float32)

        # Convert back to RaggedData
        convolved_list = [convolved[i, : lengths[i]] for i in range(len(lengths))]
        convolved_ragged = ragged_from_list(convolved_list, dtype=np.float32)

        return scores_to_frequencies(convolved_ragged)

    def compare(self, motif_1: MotifModel, motif_2: MotifModel, sequences: RaggedData | None = None) -> dict:
        """
        Compare two motif models pairwise.

        Parameters
        ----------
        motif_1 : MotifModel
            First motif to compare.
        motif_2 : MotifModel
            Second motif to compare.
        sequences : RaggedData or None
            Sequences for frequency calculation.

        Returns
        -------
        dict
            Dictionary containing comparison results with statistical information.
        """
        if sequences is None:
            raise ValueError("Sequences list is required for this comparator.")

        # Calculate comparison metrics
        out = self._single_compare(
            motif1=motif_1,
            motif2=motif_2,
            sequences=sequences,
        )

        # Merge identification info with results
        result = {"query": motif_1.name, "target": motif_2.name}
        result.update(out)

        return result


class MotaliComparator(GeneralMotifComparator):
    """Comparator that wraps the Motali program.

    This comparator uses an external Motali program to compute similarity
    between Position Frequency Matrices (PFMs).
    """

    def __init__(self, fasta_path: str, threshold: float = 0.95, tmp_directory: str = ".") -> None:
        """
        Initialize the MotaliComparator.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file containing sequences for comparison.
        threshold : float, optional
            Minimum score threshold for filtering results (default is 0.95).
        tmp_directory : str, optional
            Directory for temporary files (default is '.', the current working directory).
        """
        super().__init__(name="motali")
        self.threshold = threshold
        self.tmp_directory = tmp_directory
        self.fasta_path = fasta_path

    def compare(self, motif_1: MotifModel, motif_2: MotifModel, sequences: RaggedData | None = None) -> dict:
        """
        Compare two motif models using the Motali program.

        Parameters
        ----------
        motif_1 : MotifModel
            First motif to compare.
        motif_2 : MotifModel
            Second motif to compare.
        sequences : RaggedData or None
            Sequences for comparison (not used in this implementation).

        Returns
        -------
        dict or None
            Dictionary containing comparison results with columns:
            - query: name of the first motif
            - target: name of the second motif
            - score: similarity score computed by Motali
            Returns None if the score is below the threshold.
        """
        with tempfile.TemporaryDirectory(dir=self.tmp_directory, ignore_cleanup_errors=True) as tmp:
            # Determine file extensions based on model types
            type_1 = motif_1.model_type
            type_2 = motif_2.model_type

            if type_1 == "sitega":
                type_1 = "sga"

            if type_2 == "sitega":
                type_2 = "sga"

            # Set file extensions based on model types
            ext_1 = ".pfm" if type_1 == "pwm" else ".mat"
            ext_2 = ".pfm" if type_2 == "pwm" else ".mat"

            m1_path = os.path.join(tmp, f"motif_1{ext_1}")
            m2_path = os.path.join(tmp, f"motif_2{ext_2}")

            d1_path = os.path.join(tmp, "thresholds_1.dist")
            d2_path = os.path.join(tmp, "thresholds_2.dist")

            overlap_path = os.path.join(tmp, "overlap.txt")
            all_path = os.path.join(tmp, "all.txt")
            sta_path = os.path.join(tmp, "sta.txt")
            prc_path = os.path.join(tmp, "prc_pass.txt")
            hist_path = os.path.join(tmp, "hist_pass.txt")

            # Write motifs using polymorphic write method
            motif_1.write(m1_path)
            motif_2.write(m2_path)

            # Write distance thresholds
            motif_1.write_dist(d1_path)
            motif_2.write_dist(d2_path)

            fasta_path = self.fasta_path
            if fasta_path is None and sequences is not None:
                fasta_path = os.path.join(tmp, "sequences.fa")
                write_fasta(sequences, fasta_path)
            score = run_motali(
                fasta_path,
                m1_path,
                m2_path,
                type_1,
                type_2,
                d1_path,
                d2_path,
                overlap_path,
                all_path,
                prc_path,
                hist_path,
                sta_path,
            )

            result = {"query": motif_1.name, "target": motif_2.name, "score": score}

            return result
