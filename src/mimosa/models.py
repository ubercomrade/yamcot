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
    apply_score_log_tail_table,
    batch_all_scores,
    build_score_log_tail_table,
    lookup_log_tail,
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
    write_sitega,
)
from mimosa.ragged import RaggedData

StrandMode = Literal["best", "+", "-"]


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


def _get_threshold_tables(model: GenericModel) -> Dict[str, np.ndarray]:
    """Return cached threshold tables keyed by strand mode."""
    tables = model.config.get("_threshold_tables")
    if tables is None:
        tables = {}
        legacy_table = model.config.get("_threshold_table")
        if legacy_table is not None:
            tables["best"] = legacy_table
        model.config["_threshold_tables"] = tables
    return tables


def _resolve_threshold_table(model: GenericModel, strand: StrandMode = "best") -> Optional[np.ndarray]:
    """Return a cached threshold table for the requested strand mode."""
    tables = _get_threshold_tables(model)
    table = tables.get(strand)
    if table is not None:
        return table

    if strand == "best":
        legacy_table = model.config.get("_threshold_table")
        if legacy_table is not None:
            tables["best"] = legacy_table
            return legacy_table

    return None


def calculate_threshold_table(model: GenericModel, promoters: RaggedData, strand: StrandMode = "best") -> np.ndarray:
    """Calculate a score-to-log-tail lookup table on the provided background."""
    ragged_scores = scan_model(model, promoters, strand=strand)
    table = build_score_log_tail_table(ragged_scores.data)
    tables = _get_threshold_tables(model)
    tables[strand] = table
    if strand == "best":
        model.config["_threshold_table"] = table

    return table.astype(np.float64)


def scores_to_background_log_tail(
    model: GenericModel,
    ragged_scores: RaggedData,
    promoters: RaggedData,
    strand: StrandMode = "best",
) -> RaggedData:
    """Convert scores to background-calibrated log-tail values using both strands."""
    del strand
    threshold_table = calculate_threshold_table(model, promoters, strand="best")
    return apply_score_log_tail_table(ragged_scores, threshold_table)


def scores_to_log_fpr(
    model: GenericModel,
    ragged_scores: RaggedData,
    promoters: RaggedData,
    strand: StrandMode = "best",
) -> RaggedData:
    """Backward-compatible alias for background-calibrated log-tail conversion."""
    return scores_to_background_log_tail(model, ragged_scores, promoters, strand=strand)


def get_frequencies(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
    """Calculate per-position empirical log-tail values."""
    scores = scan_model(model, sequences, strand)

    return scores_to_empirical_log_tail(scores)


def get_scores(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
    """Calculate motif scores for each position using pure functions."""
    return scan_model(model, sequences, strand)


def get_sites(
    model: GenericModel,
    sequences: RaggedData,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Find motif binding sites in sequences."""
    threshold_table = _resolve_threshold_table(model, "best")
    if mode not in ["best", "threshold"]:
        raise ValueError(f"mode must be 'best' or 'threshold', got {mode!r}")
    if mode == "threshold" and fpr_threshold is None:
        raise ValueError("fpr_threshold is required for mode='threshold'")
    if mode == "threshold" and threshold_table is None:
        raise ValueError("Model has no threshold table")

    score_threshold = (
        _tail_probability_to_score(model, fpr_threshold, strand="best")
        if mode == "threshold" and fpr_threshold is not None
        else None
    )
    if score_threshold is not None:
        logger = logging.getLogger(__name__)
        logger.info(f"FPR threshold: {fpr_threshold} -> score threshold: {score_threshold:.4f}")

    def add_site(seq_idx: int, seq: np.ndarray, pos: int, strand_idx: int, score: float):
        """Add a site to results."""
        if pos + model.length > len(seq):
            return

        site_seq = seq[pos : pos + model.length]
        strand = "+" if strand_idx == 0 else "-"

        if strand_idx == 1:
            site_seq = _rc_sequence(site_seq)

        results.append(
            {
                "seq_index": seq_idx,
                "start": int(pos),
                "end": int(pos + model.length),
                "strand": strand,
                "score": score,
                "log_tail": _score_to_log_tail(model, score, strand="best"),
                "site": _int_to_seq(site_seq),
            }
        )

    results = []

    s_fwd_ragged = scan_model(model, sequences, strand="+")
    s_rev_ragged = scan_model(model, sequences, strand="-")
    n_seq = sequences.num_sequences

    for seq_idx in range(n_seq):
        seq = sequences.get_slice(seq_idx)
        s_fwd = s_fwd_ragged.get_slice(seq_idx)
        s_rev = s_rev_ragged.get_slice(seq_idx)

        if mode == "best":
            f_max = s_fwd.max() if s_fwd.size > 0 else -np.inf
            r_max = s_rev.max() if s_rev.size > 0 else -np.inf

            if not np.isfinite(f_max) and not np.isfinite(r_max):
                continue
            if f_max >= r_max:
                best_pos = int(np.argmax(s_fwd))
                best_score = float(f_max)
                add_site(seq_idx, seq, best_pos, 0, best_score)
            else:
                best_pos = int(np.argmax(s_rev))
                best_score = float(r_max)
                add_site(seq_idx, seq, best_pos, 1, best_score)

        else:
            f_pos = np.where(s_fwd >= score_threshold)[0]
            for pos in f_pos:
                add_site(seq_idx, seq, int(pos), 0, float(s_fwd[pos]))

            r_pos = np.where(s_rev >= score_threshold)[0]
            for pos in r_pos:
                add_site(seq_idx, seq, int(pos), 1, float(s_rev[pos]))

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values(["seq_index", "score"], ascending=[True, False]).reset_index(drop=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(df)} site(s) in {sequences.num_sequences} sequence(s)")
    return df


def get_pfm(
    model: GenericModel,
    sequences: RaggedData,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    top_fraction: Optional[float] = None,
    pseudocount: float = 0.25,
    force_recompute: bool = False,
) -> np.ndarray:
    """Construct Position Frequency Matrix (PFM) from binding sites."""
    cache_key = (
        id(sequences.data),
        id(sequences.offsets),
        sequences.data.shape,
        sequences.offsets.shape,
        mode,
        None if fpr_threshold is None else float(fpr_threshold),
        None if top_fraction is None else float(top_fraction),
        float(pseudocount),
    )
    cached_pfm = model.config.get("_derived_pfm")
    cached_key = model.config.get("_derived_pfm_key")
    if cached_pfm is not None and cached_key == cache_key and not force_recompute:
        logger = logging.getLogger(__name__)
        logger.info(f"Returning cached derived PFM for model: {model.name}")
        return cached_pfm

    logger = logging.getLogger(__name__)
    logger.info(f"Computing PFM for model: {model.name}")

    sites_df = get_sites(model, sequences, mode=mode, fpr_threshold=fpr_threshold)
    if len(sites_df) == 0:
        raise ValueError("No sites found")

    sites_df = sites_df.sort_values(by=["score"], axis=0, ascending=False)

    if top_fraction is not None:
        n_keep = max(1, int(len(sites_df) * top_fraction))
        sites_df = sites_df.nlargest(n_keep, "score")
        logger = logging.getLogger(__name__)
        logger.info(f"Selected top {top_fraction * 100:.1f}%: {n_keep} sites")

    pcm = np.zeros((4, model.length), dtype=np.float32)
    nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    for site_str in sites_df["site"]:
        for pos, nuc in enumerate(site_str):
            if nuc in nuc_map:
                pcm[nuc_map[nuc], pos] += 1.0

    pfm = pcm_to_pfm(pcm, pseudocount=pseudocount).astype(np.float32, copy=False)

    model.config["_derived_pfm"] = pfm
    model.config["_derived_pfm_key"] = cache_key

    return pfm


def _score_to_log_tail(model: GenericModel, score: float, strand: StrandMode = "best") -> float:
    """Convert score to log-tail value using a cached lookup table."""
    threshold_table = _resolve_threshold_table(model, strand)
    if threshold_table is None:
        return np.nan

    return lookup_log_tail(threshold_table, score)


def _tail_probability_to_score(model: GenericModel, tail_probability: float, strand: StrandMode = "best") -> float:
    """Convert a tail-probability threshold to the corresponding score cutoff."""
    threshold_table = _resolve_threshold_table(model, strand)
    if threshold_table is None:
        raise ValueError("Model has no threshold table")

    return lookup_score_for_tail_probability(threshold_table, tail_probability)


def _int_to_seq(seq_int: np.ndarray) -> str:
    """Convert integer-encoded sequence to ACGT string."""
    decoder = np.array(["A", "C", "G", "T", "N"], dtype="U1")
    safe_seq = np.clip(seq_int, 0, 4)
    return "".join(decoder[safe_seq])


def _rc_sequence(seq_int: np.ndarray) -> np.ndarray:
    """Return reverse complement of sequence."""
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    return rc_table[seq_int[::-1]]


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
        if pfm is not None:
            with open(path, "w") as f:
                f.write(f">{model.name}\n")
                np.savetxt(f, pfm[:4, :].T, fmt="%.8f", delimiter="\t")

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return theoretical min/max score for PWM model."""
        return _score_bounds_from_representation(model.representation)

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load PWM model from file."""
        _, ext = os.path.splitext(path.lower())

        if ext == ".pkl":
            return joblib.load(path)
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

        target_order = kwargs.get("order")
        target_order = 0 if target_order is None else int(target_order)
        _, max_order, _ = parse_file_content(path)
        if target_order > max_order:
            target_order = max_order

        representation = read_bamm(path, target_order)
        name = os.path.splitext(os.path.basename(path))[0]

        return GenericModel(
            type_key="bamm",
            name=name,
            length=representation.shape[-1],
            representation=np.asarray(representation, dtype=np.float32),
            config={"kmer": representation.ndim - 1},
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
