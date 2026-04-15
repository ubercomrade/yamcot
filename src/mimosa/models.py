"""
Model utilities and registry-backed motif strategies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from mimosa.functions import (
    batch_all_scores,
    build_score_log_tail_table,
    lookup_score_for_tail_probability,
    pcm_to_pfm,
    pfm_to_pwm,
    scores_to_empirical_log_tail,
)
from mimosa.io import (
    parse_file_content,
    read_bamm,
    read_dimont,
    read_meme,
    read_pfm,
    read_scores,
    read_sitega,
    read_slim,
    write_pfm,
    write_sitega,
)
from mimosa.ragged import RaggedData

StrandMode = Literal["best", "+", "-"]
_SEQ_DECODER = np.array(["A", "C", "G", "T", "N"], dtype="U1")
_RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)


@dataclass(eq=False)
class GenericModel:
    """Motif model container with mutable runtime caches in ``config``."""

    type_key: str
    name: str
    representation: Any = dc_field(hash=False)
    length: int
    config: dict = dc_field(default_factory=dict, hash=False)


class ModelRegistry:
    """Registry for model strategies using decorator pattern."""

    def __init__(self):
        """Initialize registry state."""
        self._strategies: Dict[str, type] = {}

    def register(self, key: str):
        """Decorator to register a model strategy class."""

        def decorator(strategy_cls):
            """Store a callable in the registry."""
            self._strategies[key] = strategy_cls
            logging.info(f"Registered model strategy: {key} -> {strategy_cls.__name__}")
            return strategy_cls

        return decorator

    def get(self, key: str) -> type:
        """Get strategy class by key."""
        if key not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Model strategy '{key}' not found. Available: {available}")
        return self._strategies[key]


registry = ModelRegistry()


def scan_model(model: GenericModel, sequences: Optional[RaggedData], strand: Optional[StrandMode] = None) -> RaggedData:
    """Universal scanning function that dispatches to the appropriate strategy."""
    strategy_cls = registry.get(model.type_key)
    strand_mode = strand or model.config.get("strand_mode", "best")
    return strategy_cls.scan(model, sequences, strand_mode)


def write_model(model: GenericModel, path: str) -> None:
    """Universal write function that dispatches to the appropriate strategy."""
    strategy_cls = registry.get(model.type_key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    strategy_cls.write(model, path)


def read_model(path: str, model_type: str, **kwargs) -> GenericModel:
    """Factory function for creating models from files."""
    strategy_cls = registry.get(model_type)
    return strategy_cls.load(path, kwargs)


def get_score_bounds(model: GenericModel) -> tuple[float, float]:
    """Return theoretical minimum and maximum scores for a model."""
    strategy_cls = registry.get(model.type_key)
    if not hasattr(strategy_cls, "score_bounds"):
        raise NotImplementedError(f"Model strategy '{model.type_key}' does not implement score_bounds")
    return strategy_cls.score_bounds(model)


def _score_bounds_from_representation(representation: np.ndarray) -> tuple[float, float]:
    """
    Compute theoretical score bounds from model tensor representation.
    """

    minimum = representation.min(axis=tuple(range(representation.ndim - 1))).sum()
    maximum = representation.max(axis=tuple(range(representation.ndim - 1))).sum()
    return minimum, maximum


def _get_float32_representation(model: GenericModel) -> np.ndarray:
    """Return a cached float32 representation for repeated scans."""
    representation = np.asarray(model.representation)
    meta = (id(representation), representation.shape, representation.dtype.str)

    cached_meta = model.config.get("_representation_float32_meta")
    cached = model.config.get("_representation_float32")
    if cached_meta == meta and cached is not None:
        return cached

    converted = np.asarray(representation, dtype=np.float32)
    model.config["_representation_float32_meta"] = meta
    model.config["_representation_float32"] = converted
    return converted


def calculate_threshold_table(model: GenericModel, sequences: RaggedData, strand: StrandMode = "best") -> np.ndarray:
    """Calculate a score-to-log-tail lookup table on explicitly provided sequences."""
    ragged_scores = scan_model(model, sequences, strand=strand)
    return build_score_log_tail_table(ragged_scores.data).astype(np.float64, copy=False)


def get_frequencies(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
    """Calculate per-position empirical log-tail values."""
    scores = scan_model(model, sequences, strand)

    return scores_to_empirical_log_tail(scores)


def get_scores(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
    """Calculate motif scores for each position using pure functions."""
    return scan_model(model, sequences, strand)


def _empty_hit_arrays() -> dict[str, np.ndarray]:
    """Return a consistent empty hit-array payload."""
    return {
        "seq_index": np.empty(0, dtype=np.int64),
        "start": np.empty(0, dtype=np.int64),
        "strand_idx": np.empty(0, dtype=np.int8),
        "score": np.empty(0, dtype=np.float32),
    }


def _scan_both_strands(model: GenericModel, sequences: RaggedData) -> tuple[RaggedData, RaggedData]:
    """Scan sequences on both strands."""
    return scan_model(model, sequences, strand="+"), scan_model(model, sequences, strand="-")


def _collect_best_hits(
    sequences: RaggedData,
    s_fwd_ragged: RaggedData,
    s_rev_ragged: RaggedData,
) -> dict[str, np.ndarray]:
    """Collect the single best hit per sequence as numeric arrays."""
    seq_indices: list[int] = []
    starts: list[int] = []
    strand_indices: list[int] = []
    scores: list[float] = []

    for seq_idx in range(sequences.num_sequences):
        s_fwd = s_fwd_ragged.get_slice(seq_idx)
        s_rev = s_rev_ragged.get_slice(seq_idx)

        f_max = s_fwd.max() if s_fwd.size > 0 else -np.inf
        r_max = s_rev.max() if s_rev.size > 0 else -np.inf

        if not np.isfinite(f_max) and not np.isfinite(r_max):
            continue

        seq_indices.append(seq_idx)
        if f_max >= r_max:
            starts.append(int(np.argmax(s_fwd)))
            strand_indices.append(0)
            scores.append(float(f_max))
        else:
            starts.append(int(np.argmax(s_rev)))
            strand_indices.append(1)
            scores.append(float(r_max))

    if not seq_indices:
        return _empty_hit_arrays()

    return {
        "seq_index": np.asarray(seq_indices, dtype=np.int64),
        "start": np.asarray(starts, dtype=np.int64),
        "strand_idx": np.asarray(strand_indices, dtype=np.int8),
        "score": np.asarray(scores, dtype=np.float32),
    }


def _collect_threshold_hits(
    sequences: RaggedData,
    s_fwd_ragged: RaggedData,
    s_rev_ragged: RaggedData,
    score_threshold: float,
) -> dict[str, np.ndarray]:
    """Collect all hits above threshold as numeric arrays."""
    del sequences
    seq_indices_parts = []
    start_parts = []
    strand_parts = []
    score_parts = []

    for seq_idx in range(s_fwd_ragged.num_sequences):
        s_fwd = s_fwd_ragged.get_slice(seq_idx)
        s_rev = s_rev_ragged.get_slice(seq_idx)

        f_pos = np.flatnonzero(s_fwd >= score_threshold)
        if f_pos.size > 0:
            seq_indices_parts.append(np.full(f_pos.size, seq_idx, dtype=np.int64))
            start_parts.append(f_pos.astype(np.int64, copy=False))
            strand_parts.append(np.zeros(f_pos.size, dtype=np.int8))
            score_parts.append(s_fwd[f_pos].astype(np.float32, copy=False))

        r_pos = np.flatnonzero(s_rev >= score_threshold)
        if r_pos.size > 0:
            seq_indices_parts.append(np.full(r_pos.size, seq_idx, dtype=np.int64))
            start_parts.append(r_pos.astype(np.int64, copy=False))
            strand_parts.append(np.ones(r_pos.size, dtype=np.int8))
            score_parts.append(s_rev[r_pos].astype(np.float32, copy=False))

    if not seq_indices_parts:
        return _empty_hit_arrays()

    return {
        "seq_index": np.concatenate(seq_indices_parts),
        "start": np.concatenate(start_parts),
        "strand_idx": np.concatenate(strand_parts),
        "score": np.concatenate(score_parts),
    }


def _collect_hits(
    model: GenericModel,
    sequences: RaggedData,
    mode: str,
    score_threshold: Optional[float],
) -> dict[str, np.ndarray]:
    """Collect motif hits as numeric arrays."""
    s_fwd_ragged, s_rev_ragged = _scan_both_strands(model, sequences)
    if mode == "best":
        return _collect_best_hits(sequences, s_fwd_ragged, s_rev_ragged)
    return _collect_threshold_hits(sequences, s_fwd_ragged, s_rev_ragged, float(score_threshold))


def _scores_to_log_tail_array(scores: np.ndarray, threshold_table: np.ndarray) -> np.ndarray:
    """Convert a score array to log-tail values using an explicit lookup table."""
    if scores.size == 0:
        return np.empty(0, dtype=np.float64)

    scores_col = threshold_table[:, 0]
    log_tail_col = threshold_table[:, 1]
    idx = np.searchsorted(-scores_col, -scores.astype(np.float64, copy=False), side="left")
    idx = np.clip(idx, 0, len(log_tail_col) - 1)
    return log_tail_col[idx]


def _resolve_hit_threshold_table(
    model: GenericModel,
    sequences: RaggedData,
    background_sequences: Optional[RaggedData],
    threshold_table: Optional[np.ndarray],
) -> np.ndarray:
    """Resolve the explicit log-tail table used for hit extraction and annotation."""
    if threshold_table is not None:
        return np.asarray(threshold_table, dtype=np.float64)

    calibration_sequences = background_sequences if background_sequences is not None else sequences
    return calculate_threshold_table(model, calibration_sequences, strand="best")


def _extract_site_matrix(
    sequences: RaggedData,
    seq_indices: np.ndarray,
    starts: np.ndarray,
    motif_length: int,
    strand_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract numeric motif windows for a set of hits."""
    n_hits = seq_indices.size
    sites = np.empty((n_hits, motif_length), dtype=sequences.data.dtype)

    for hit_idx in range(n_hits):
        seq = sequences.get_slice(int(seq_indices[hit_idx]))
        start = int(starts[hit_idx])
        sites[hit_idx] = seq[start : start + motif_length]

    if strand_indices is not None:
        minus_mask = strand_indices == 1
        if np.any(minus_mask):
            sites[minus_mask] = _RC_TABLE[sites[minus_mask, ::-1]]

    return sites


def _site_matrix_to_strings(site_matrix: np.ndarray) -> np.ndarray:
    """Convert numeric site windows into DNA strings."""
    if site_matrix.size == 0:
        return np.empty(0, dtype=object)

    decoded = _SEQ_DECODER[np.clip(site_matrix, 0, 4)]
    return np.fromiter(("".join(row) for row in decoded), dtype=object, count=decoded.shape[0])


def _build_pcm_from_site_matrix(site_matrix: np.ndarray, motif_length: int) -> np.ndarray:
    """Build a position count matrix from numeric sites."""
    pcm = np.zeros((4, motif_length), dtype=np.float32)
    if site_matrix.size == 0:
        return pcm

    valid_mask = site_matrix < 4
    col_idx = np.broadcast_to(np.arange(motif_length, dtype=np.int64), site_matrix.shape)
    np.add.at(pcm, (site_matrix[valid_mask], col_idx[valid_mask]), 1.0)
    return pcm


def _sort_hit_arrays(hit_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort hits by sequence index ascending and score descending."""
    if hit_arrays["score"].size == 0:
        return hit_arrays

    order = np.lexsort((-hit_arrays["score"], hit_arrays["seq_index"]))
    return {key: values[order] for key, values in hit_arrays.items()}


def _select_top_hit_arrays(hit_arrays: dict[str, np.ndarray], top_fraction: Optional[float]) -> dict[str, np.ndarray]:
    """Keep only the top-scoring hits."""
    if top_fraction is None or hit_arrays["score"].size == 0:
        return hit_arrays

    n_hits = hit_arrays["score"].size
    n_keep = max(1, int(n_hits * top_fraction))
    if n_keep >= n_hits:
        return hit_arrays

    keep_idx = np.argpartition(hit_arrays["score"], n_hits - n_keep)[-n_keep:]
    keep_idx = keep_idx[np.argsort(hit_arrays["score"][keep_idx])[::-1]]
    return {key: values[keep_idx] for key, values in hit_arrays.items()}


def get_sites(
    model: GenericModel,
    sequences: RaggedData,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    background_sequences: Optional[RaggedData] = None,
    threshold_table: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Find motif binding sites in sequences."""
    if mode not in ["best", "threshold"]:
        raise ValueError(f"mode must be 'best' or 'threshold', got {mode}")
    if mode == "threshold" and fpr_threshold is None:
        raise ValueError("fpr_threshold is required for mode='threshold'")

    resolved_threshold_table = _resolve_hit_threshold_table(model, sequences, background_sequences, threshold_table)

    score_threshold = (
        _tail_probability_to_score(float(fpr_threshold), resolved_threshold_table)
        if mode == "threshold" and fpr_threshold is not None
        else None
    )
    if score_threshold is not None:
        logger = logging.getLogger(__name__)
        logger.info(f"FPR threshold: {fpr_threshold} -> score threshold: {score_threshold:.4f}")

    hit_arrays = _collect_hits(model, sequences, mode, score_threshold)
    hit_arrays = _sort_hit_arrays(hit_arrays)

    if hit_arrays["score"].size == 0:
        df = pd.DataFrame(
            {
                "seq_index": np.empty(0, dtype=np.int64),
                "start": np.empty(0, dtype=np.int64),
                "end": np.empty(0, dtype=np.int64),
                "strand": np.empty(0, dtype=object),
                "score": np.empty(0, dtype=np.float32),
                "log_tail": np.empty(0, dtype=np.float64),
                "site": np.empty(0, dtype=object),
            }
        )
    else:
        site_matrix = _extract_site_matrix(
            sequences,
            hit_arrays["seq_index"],
            hit_arrays["start"],
            model.length,
            hit_arrays["strand_idx"],
        )
        df = pd.DataFrame(
            {
                "seq_index": hit_arrays["seq_index"],
                "start": hit_arrays["start"],
                "end": hit_arrays["start"] + model.length,
                "strand": np.where(hit_arrays["strand_idx"] == 0, "+", "-"),
                "score": hit_arrays["score"],
                "log_tail": _scores_to_log_tail_array(hit_arrays["score"], resolved_threshold_table),
                "site": _site_matrix_to_strings(site_matrix),
            }
        )

    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(df)} site(s) in {sequences.num_sequences} sequence(s)")
    return df


def get_pfm(
    model: GenericModel,
    sequences: RaggedData,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    background_sequences: Optional[RaggedData] = None,
    threshold_table: Optional[np.ndarray] = None,
    top_fraction: Optional[float] = None,
    pseudocount: float = 0.25,
    force_recompute: bool = False,
) -> np.ndarray:
    """Construct Position Frequency Matrix (PFM) from binding sites."""
    del force_recompute

    logger = logging.getLogger(__name__)
    logger.info(f"Computing PFM for model: {model.name}")

    if mode not in ["best", "threshold"]:
        raise ValueError(f"mode must be 'best' or 'threshold', got {mode}")
    if mode == "threshold" and fpr_threshold is None:
        raise ValueError("fpr_threshold is required for mode='threshold'")

    resolved_threshold_table = None
    if mode == "threshold":
        resolved_threshold_table = _resolve_hit_threshold_table(model, sequences, background_sequences, threshold_table)

    score_threshold = (
        _tail_probability_to_score(float(fpr_threshold), resolved_threshold_table)
        if mode == "threshold" and fpr_threshold is not None and resolved_threshold_table is not None
        else None
    )

    hit_arrays = _collect_hits(model, sequences, mode, score_threshold)
    if hit_arrays["score"].size == 0:
        raise ValueError("No sites found")

    selected_hits = _select_top_hit_arrays(hit_arrays, top_fraction)
    if top_fraction is not None:
        logger.info(f"Selected top {top_fraction * 100:.1f}%: {selected_hits['score'].size} sites")

    site_matrix = _extract_site_matrix(
        sequences,
        selected_hits["seq_index"],
        selected_hits["start"],
        model.length,
        selected_hits["strand_idx"],
    )
    pcm = _build_pcm_from_site_matrix(site_matrix, model.length)

    pfm = pcm_to_pfm(pcm, pseudocount=pseudocount).astype(np.float32, copy=False)

    return pfm


def _tail_probability_to_score(tail_probability: float, threshold_table: np.ndarray) -> float:
    """Convert a tail-probability threshold to the corresponding score cutoff."""
    return lookup_score_for_tail_probability(threshold_table, tail_probability)


def _scan_with_batch_kernel(
    model: GenericModel,
    sequences: RaggedData,
    strand: StrandMode,
    *,
    with_context: bool = False,
) -> RaggedData:
    """Scan a tensor-based motif model with the shared batch scoring kernel."""
    representation = _get_float32_representation(model)
    kmer = model.config.get("kmer", 1)

    if strand == "+":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
    elif strand == "-":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
    elif strand == "best":
        sf = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
        sr = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
        return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
    else:
        raise ValueError(f"Invalid strand mode: {strand}")


@registry.register("pwm")
class PwmStrategy:
    """PWM (Position Weight Matrix) strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with PWM model."""
        return _scan_with_batch_kernel(model, sequences, strand, with_context=False)

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Write PWM model to file."""
        pfm = model.config.get("_source_pfm")
        if pfm is None:
            raise ValueError("PWM serialization requires the source PFM in model.config['_source_pfm'].")
        write_pfm(np.asarray(pfm, dtype=np.float32), model.name, model.length, path)

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for PWM model."""
        return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load PWM model from file."""
        _, ext = os.path.splitext(path.lower())

        if ext == ".pkl":
            model = joblib.load(path)
            if not isinstance(model, GenericModel):
                raise TypeError(f"Unsupported PWM pickle payload: expected GenericModel, got {type(model)!r}")
            if model.config.get("_source_pfm") is None:
                raise ValueError(
                    "Unsupported PWM pickle format: source PFM is missing from model.config['_source_pfm']."
                )
            return model
        elif ext == ".meme":
            pfm, info, _ = read_meme(path, index=kwargs.get("index", 0))
            name, length = info
        elif ext == ".pfm":
            pfm, length = read_pfm(path)
            name = os.path.splitext(os.path.basename(path))[0]
        else:
            raise ValueError(f"Unsupported PWM format: {path}")

        pwm = pfm_to_pwm(pfm)

        pwm_ext = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32, copy=False)

        return GenericModel(
            type_key="pwm",
            name=name,
            length=int(length),
            representation=pwm_ext,
            config={"kmer": 1, "_source_pfm": pfm},
        )


@registry.register("sitega")
class SitegaStrategy:
    """SiteGA strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with PWM model."""
        return _scan_with_batch_kernel(model, sequences, strand, with_context=False)

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Write SiteGA model to file."""
        write_sitega(model, path)

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for SiteGA model."""
        minimum = model.config.get("minimum")
        maximum = model.config.get("maximum")
        if minimum is not None and maximum is not None:
            return minimum, maximum
        else:
            return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load SiteGA model from file."""
        _, ext = os.path.splitext(path.lower())

        if ext == ".pkl":
            return joblib.load(path)
        elif ext == ".mat":
            representation, name, length, minimum, maximum = read_sitega(path)
        else:
            raise ValueError(f"Unsupported SiteGA format: {path}")

        return GenericModel(
            type_key="sitega",
            name=name,
            length=int(length),
            representation=np.asarray(representation, dtype=np.float32),
            config={"kmer": 2, "minimum": float(minimum), "maximum": float(maximum)},
        )


@registry.register("bamm")
class BammStrategy:
    """BaMM (Bayesian Markov Model) strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with BaMM model."""
        return _scan_with_batch_kernel(model, sequences, strand, with_context=True)

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Write BaMM model to file."""
        joblib.dump(model, path)

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for BaMM model."""
        return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load BaMM model from file."""

        if not path.endswith(".ihbcp") and not os.path.exists(path):
            ihbcp_path = f"{path}.ihbcp"
            if os.path.exists(ihbcp_path):
                path = ihbcp_path
            else:
                raise FileNotFoundError(f"BaMM file not found: {path}")

        _, max_order, _ = parse_file_content(path)
        target_order = kwargs.get("order")
        target_order = max_order if target_order is None else int(target_order)
        if target_order > max_order:
            target_order = max_order

        representation = read_bamm(path, target_order)
        name = os.path.splitext(os.path.basename(path))[0]

        return GenericModel(
            type_key="bamm",
            name=name,
            length=representation.shape[-1],
            representation=np.asarray(representation, dtype=np.float32),
            config={"kmer": representation.ndim - 1, "order": target_order},
        )


@registry.register("dimont")
class DimontStrategy:
    """Dimont motif model strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with a Dimont motif model."""
        return _scan_with_batch_kernel(model, sequences, strand, with_context=model.config.get("kmer", 1) > 1)

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Persist Dimont models via pickle serialization."""
        joblib.dump(model, path)

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for a Dimont model."""
        return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load a Dimont XML model."""
        _, ext = os.path.splitext(path.lower())

        if ext == ".pkl":
            return joblib.load(path)
        elif ext == ".xml":
            representation, length, span = read_dimont(path)
        else:
            raise ValueError(f"Unsupported Dimont format: {path}")

        name = os.path.splitext(os.path.basename(path))[0]
        return GenericModel(
            type_key="dimont",
            name=name,
            length=length,
            representation=np.asarray(representation, dtype=np.float32),
            config={"kmer": span + 1},
        )


@registry.register("slim")
class SlimStrategy:
    """Slim motif model strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with a Slim motif model."""
        return _scan_with_batch_kernel(model, sequences, strand, with_context=model.config.get("kmer", 1) > 1)

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Persist Slim models via pickle serialization."""
        joblib.dump(model, path)

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for a Slim model."""
        return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load a Slim XML model."""
        _, ext = os.path.splitext(path.lower())

        if ext == ".pkl":
            return joblib.load(path)
        elif ext == ".xml":
            representation, length, span = read_slim(path)
        else:
            raise ValueError(f"Unsupported Slim format: {path}")

        name = os.path.splitext(os.path.basename(path))[0]
        return GenericModel(
            type_key="slim",
            name=name,
            length=length,
            representation=np.asarray(representation, dtype=np.float32),
            config={"kmer": span + 1},
        )


@registry.register("scores")
class ScoresStrategy:
    """Numerical score-profile strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: Optional[RaggedData] = None, strand: StrandMode = "best") -> RaggedData:
        """Return stored score profiles directly."""
        return model.config["scores_data"]

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Score profiles are not writable through model serialization."""
        raise NotImplementedError("Score profiles cannot be written to files")

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return score bounds for numerical score profiles."""
        scores_data = model.config["scores_data"]
        if scores_data.data.size == 0:
            return 0.0, 0.0
        return float(np.min(scores_data.data)), float(np.max(scores_data.data))

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load numerical score profiles from a FASTA-like text file."""

        scores_data = read_scores(path)
        name = os.path.splitext(os.path.basename(path))[0]

        return GenericModel(
            type_key="scores",
            name=name,
            length=0,
            representation=None,
            config={"scores_data": scores_data},
        )
