"""
Functional Models Module
========================

This module provides a functional programming approach to motif models,
replacing class hierarchies with immutable data structures and registry-based polymorphism.

Key Features:
- Immutable data containers using frozen dataclasses
- Registry-based strategy pattern using decorators
- Pure functions for all operations
- Dependency injection for all dependencies
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from mimosa.functions import batch_all_scores, pfm_to_pwm, scores_to_frequencies
from mimosa.io import parse_file_content, read_bamm, read_fasta, read_meme, read_pfm, read_sitega, write_sitega
from mimosa.ragged import RaggedData

StrandMode = Literal["best", "+", "-"]


@dataclass(frozen=True)
class GenericModel:
    """Immutable motif model container.

    This dataclass provides immutability but the representation field
    is excluded from hashing due to numpy array unhashability.

    Attributes
    ----------
    type_key : str
        Model type identifier for registry dispatch
    name : str
        Human-readable name
    representation : Any
        Numeric representation of the motif
    length : int
        Length of the motif
    config : dict
        Additional configuration parameters
    """

    type_key: str
    name: str
    representation: Any = dc_field(hash=False)
    length: int
    config: dict = dc_field(default_factory=dict, hash=False)

    def __hash__(self):
        """Custom hash implementation excluding unhashable fields."""
        return hash((self.type_key, self.name, self.length))


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


def scan_model(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
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


@functools.lru_cache(maxsize=32)
def calculate_threshold_table(model: GenericModel, promoters: RaggedData) -> np.ndarray:
    """Pure function to calculate threshold table."""

    ragged_scores = scan_model(model, promoters, strand="best")

    flat_scores = ragged_scores.data

    if flat_scores.size == 0:
        return np.array([[0.0, 0.0]], dtype=np.float64)

    scores_sorted = np.sort(flat_scores)[::-1]
    n_total = flat_scores.size

    unique_scores, inverse, counts = np.unique(scores_sorted, return_inverse=True, return_counts=True)
    unique_scores = unique_scores[::-1]
    counts = counts[::-1]

    cum_counts = np.cumsum(counts)
    fpr_values = cum_counts / n_total
    log_fpr_values = -np.log10(fpr_values)

    table = np.column_stack([unique_scores, np.abs(log_fpr_values)])
    table = table.astype(np.float64)
    model.config["_threshold_table"] = table

    return table.astype(np.float64)


def get_frequencies(model: GenericModel, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
    """Calculate per-position hit frequencies using pure functions."""
    scores = scan_model(model, sequences, strand)

    return scores_to_frequencies(scores)


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
    threshold_table = model.config.get("_threshold_table")
    if mode not in ["best", "threshold"]:
        raise ValueError(f"mode must be 'best' or 'threshold', got {mode!r}")
    if mode == "threshold" and fpr_threshold is None:
        raise ValueError("fpr_threshold is required for mode='threshold'")
    if mode == "threshold" and threshold_table is None:
        raise ValueError("Model has no threshold table")

    score_threshold = (
        _frequency_to_score(model, fpr_threshold) if mode == "threshold" and fpr_threshold is not None else None
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
                "frequency": _score_to_frequency(model, score),
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
            f_max = s_fwd.max() if s_fwd.size > 0 else -1e9
            r_max = s_rev.max() if s_rev.size > 0 else -1e9

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

    cached_pfm = model.config.get("_pfm")
    if cached_pfm is not None and not force_recompute:
        logger = logging.getLogger(__name__)
        logger.info(f"Returning cached PFM for model: {model.name}")
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

    pfm = np.full((4, model.length), pseudocount, dtype=np.float32)
    nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    for site_str in sites_df["site"]:
        for pos, nuc in enumerate(site_str):
            if nuc in nuc_map:
                pfm[nuc_map[nuc], pos] += 1.0

    pfm = pfm / pfm.sum(axis=0, keepdims=True)

    model.config["_pfm"] = pfm

    return pfm


def _score_to_frequency(model: GenericModel, score: float) -> float:
    """Convert score to frequency using threshold table."""
    threshold_table = model.config.get("_threshold_table")
    if threshold_table is None:
        return np.nan

    scores_col = threshold_table[:, 0]
    logfpr_col = threshold_table[:, 1]

    if score >= scores_col[0]:
        return float(logfpr_col[0])
    if score <= scores_col[-1]:
        return float(logfpr_col[-1])

    idx = np.searchsorted(-scores_col, -score, side="left")
    if idx >= len(logfpr_col):
        return float(logfpr_col[-1])
    return float(logfpr_col[idx])


def _frequency_to_score(model: GenericModel, frequency: float) -> float:
    """Convert frequency to score using threshold table."""
    threshold_table = model.config.get("_threshold_table")
    if threshold_table is None:
        raise ValueError("Model has no threshold table")

    if frequency <= 0:
        return float(threshold_table[0, 0])

    target_logfpr = -np.log10(frequency)

    scores_col = threshold_table[:, 0]
    logfpr_col = threshold_table[:, 1]

    mask = logfpr_col >= target_logfpr
    if not np.any(mask):
        return float(scores_col[-1])

    last_valid = np.where(mask)[0][-1]
    return float(scores_col[last_valid])


def _int_to_seq(seq_int: np.ndarray) -> str:
    """Convert integer-encoded sequence to ACGT string."""
    decoder = np.array(["A", "C", "G", "T", "N"], dtype="U1")
    safe_seq = np.clip(seq_int, 0, 4)
    return "".join(decoder[safe_seq])


def _rc_sequence(seq_int: np.ndarray) -> np.ndarray:
    """Return reverse complement of sequence."""
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    return rc_table[seq_int[::-1]]


@registry.register("pwm")
class PwmStrategy:
    """PWM (Position Weight Matrix) strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with PWM model."""
        representation = model.representation.astype(np.float32)
        kmer = model.config.get("kmer", 1)

        if strand == "+":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False)
        elif strand == "-":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False)
            sr = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            raise ValueError(f"Invalid strand mode: {strand}")

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Write PWM model to file."""
        pfm = model.config.get("_pfm")
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

        pwm_ext = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0)

        return GenericModel(
            type_key="pwm", name=name, length=int(length), representation=pwm_ext, config={"kmer": 1, "_pfm": pfm}
        )


@registry.register("sitega")
class SitegaStrategy:
    """SiteGA strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with PWM model."""
        representation = model.representation.astype(np.float32)
        kmer = model.config.get("kmer", 2)

        if strand == "+":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False)
        elif strand == "-":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False)
            sr = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            raise ValueError(f"Invalid strand mode: {strand}")

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
            representation=representation,
            config={"kmer": 2, "minimum": float(minimum), "maximum": float(maximum)},
        )


@registry.register("bamm")
class BammStrategy:
    """BaMM (Bayesian Markov Model) strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: RaggedData, strand: StrandMode) -> RaggedData:
        """Scan sequences with BaMM model."""
        representation = model.representation.astype(np.float32)
        kmer = model.config.get("kmer", 2)

        if strand == "+":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=True)
        elif strand == "-":
            return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=True)
            sr = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            raise ValueError(f"Invalid strand mode: {strand}")

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
            hbcp_path = f"{path}.hbcp"
            if os.path.exists(ihbcp_path):
                path = ihbcp_path
                bg_path = hbcp_path if os.path.exists(hbcp_path) else None
            else:
                raise FileNotFoundError(f"BaMM file not found: {path}")
        else:
            bg_path = kwargs.get("bg_path")
            if bg_path is None:
                base_path = path.replace(".ihbcp", "")
                hbcp_path = f"{base_path}.hbcp"
                if os.path.exists(hbcp_path):
                    bg_path = hbcp_path

        target_order = kwargs.get("order", 2)
        _, max_order, length = parse_file_content(path)
        if target_order > max_order:
            target_order = max_order

        representation = read_bamm(path, bg_path, target_order)
        name = os.path.splitext(os.path.basename(path))[0]

        return GenericModel(
            type_key="bamm",
            name=name,
            length=representation.shape[-1],
            representation=representation,
            config={"kmer": 2},
        )


@registry.register("profile")
class ProfileStrategy:
    """Profile data strategy implementation."""

    @staticmethod
    def scan(model: GenericModel, sequences: Optional[RaggedData] = None, strand: StrandMode = "best") -> RaggedData:
        """For profile comparisons, return the profile data directly."""
        return model.config["profile_data"]

    @staticmethod
    def write(model: GenericModel, path: str) -> None:
        """Profile models are not writable in the traditional sense."""
        raise NotImplementedError("Profile models cannot be written to files")

    @staticmethod
    def score_bounds(model: GenericModel) -> tuple[float, float]:
        """Return score bounds for profile data."""
        profile_data = model.config["profile_data"]
        if profile_data.data.size == 0:
            return 0.0, 0.0
        return float(np.min(profile_data.data)), float(np.max(profile_data.data))

    @staticmethod
    def load(path: str, kwargs: dict) -> GenericModel:
        """Load profile data from FASTA file."""

        profile_data = read_fasta(path)
        name = os.path.splitext(os.path.basename(path))[0]

        return GenericModel(
            type_key="profile",
            name=name,
            length=0,
            representation=None,
            config={"profile_data": profile_data},
        )
