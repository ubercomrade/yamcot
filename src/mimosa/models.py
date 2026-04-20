"""Model utilities and registry-backed motif handlers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from mimosa.batches import (
    MINUS_STRAND,
    PLUS_STRAND,
    SCORE_PADDING,
    make_strand_bundle,
    pack_batch,
    profile_row_values,
    row_values,
)
from mimosa.functions import (
    batch_all_scores,
    batch_all_scores_strands,
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
from mimosa.validation import validate_site_mode

StrandMode = Literal["best", "+", "-"]
_SEQ_DECODER = np.array(["A", "C", "G", "T", "N"], dtype="U1")
_RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
_NUCLEOTIDE_CARDINALITY = 4


@dataclass(eq=False)
class GenericModel:
    """Motif model container."""

    type_key: str
    name: str
    representation: Any
    length: int
    config: dict


registry: Dict[str, dict] = {}


def _register_model_handler(key: str, *, scan, load, write, score_bounds, scan_both=None) -> None:
    """Register one model handler bundle."""
    registry[key] = {
        "scan": scan,
        "scan_both": scan_both,
        "load": load,
        "write": write,
        "score_bounds": score_bounds,
    }


def register_model_handler(key: str, *, scan, load, write, score_bounds, scan_both=None) -> None:
    """Register one public model handler bundle."""
    _register_model_handler(
        key,
        scan=scan,
        load=load,
        write=write,
        score_bounds=score_bounds,
        scan_both=scan_both,
    )


def _get_model_handler(key: str) -> dict:
    """Return one registered handler bundle."""
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Model strategy '{key}' not found. Available: {available}") from exc


def scan_model(model: GenericModel, sequences=None, strand: Optional[StrandMode] = None):
    """Universal scanning function that dispatches to the appropriate handler."""
    handler = _get_model_handler(model.type_key)
    strand_mode = strand or model.config.get("strand_mode", "best")
    return handler["scan"](model, sequences, strand_mode)


def scan_model_strands(model: GenericModel, sequences=None):
    """Scan one model on both strands, using a shared backend call when available."""
    handler = _get_model_handler(model.type_key)
    scan_both = handler.get("scan_both")
    if scan_both is not None:
        plus_scores, minus_scores = scan_both(model, sequences)
    else:
        plus_scores = handler["scan"](model, sequences, "+")
        minus_scores = handler["scan"](model, sequences, "-")
    return make_strand_bundle(plus_scores, minus_scores)


def write_model(model: GenericModel, path: str) -> None:
    """Universal write function that dispatches to the appropriate handler."""
    handler = _get_model_handler(model.type_key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handler["write"](model, path)


def read_model(path: str, model_type: str, **kwargs) -> GenericModel:
    """Factory function for creating models from files."""
    handler = _get_model_handler(model_type)
    return handler["load"](path, kwargs)


def get_score_bounds(model: GenericModel) -> tuple[float, float]:
    """Return theoretical minimum and maximum scores for a model."""
    return _get_model_handler(model.type_key)["score_bounds"](model)


def _score_bounds_from_representation(representation: np.ndarray) -> tuple[float, float]:
    """Compute theoretical score bounds from one model tensor."""
    minimum = representation.min(axis=tuple(range(representation.ndim - 1))).sum()
    maximum = representation.max(axis=tuple(range(representation.ndim - 1))).sum()
    return minimum, maximum


def calculate_threshold_table(model: GenericModel, sequences, strand: StrandMode = "best") -> np.ndarray:
    """Calculate a score-to-log-tail lookup table on explicitly provided sequences."""
    scores = scan_model(model, sequences, strand=strand)
    return build_score_log_tail_table(scores["values"][scores["mask"]]).astype(np.float64, copy=False)


def get_frequencies(model: GenericModel, sequences, strand: Optional[StrandMode] = None):
    """Calculate per-position empirical log-tail values."""
    return scores_to_empirical_log_tail(scan_model(model, sequences, strand))


def get_scores(model: GenericModel, sequences, strand: Optional[StrandMode] = None):
    """Calculate motif scores for each position."""
    return scan_model(model, sequences, strand)


def _empty_hit_arrays() -> dict[str, np.ndarray]:
    """Return a consistent empty hit-array payload."""
    return {
        "seq_index": np.empty(0, dtype=np.int64),
        "start": np.empty(0, dtype=np.int64),
        "strand_idx": np.empty(0, dtype=np.int8),
        "score": np.empty(0, dtype=np.float32),
    }


def _scan_both_strands(model: GenericModel, sequences):
    """Scan sequences on both strands."""
    return scan_model_strands(model, sequences)


def _collect_best_hits(sequences, score_bundle) -> dict[str, np.ndarray]:
    """Collect the single best hit per sequence as numeric arrays."""
    seq_indices: list[int] = []
    starts: list[int] = []
    strand_indices: list[int] = []
    scores: list[float] = []

    for seq_idx in range(len(sequences["lengths"])):
        s_fwd = profile_row_values(score_bundle, PLUS_STRAND, seq_idx)
        s_rev = profile_row_values(score_bundle, MINUS_STRAND, seq_idx)
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


def _collect_threshold_hits(score_bundle, score_threshold: float) -> dict[str, np.ndarray]:
    """Collect all hits above threshold as numeric arrays."""
    seq_indices_parts = []
    start_parts = []
    strand_parts = []
    score_parts = []

    for seq_idx in range(len(score_bundle["lengths"])):
        s_fwd = profile_row_values(score_bundle, PLUS_STRAND, seq_idx)
        s_rev = profile_row_values(score_bundle, MINUS_STRAND, seq_idx)

        f_pos = np.flatnonzero(s_fwd >= score_threshold)
        if f_pos.size > 0:
            seq_indices_parts.append(np.full(f_pos.size, seq_idx, dtype=np.int64))
            start_parts.append(f_pos.astype(np.int64, copy=False))
            strand_parts.append(np.full(f_pos.size, PLUS_STRAND, dtype=np.int8))
            score_parts.append(s_fwd[f_pos].astype(np.float32, copy=False))

        r_pos = np.flatnonzero(s_rev >= score_threshold)
        if r_pos.size > 0:
            seq_indices_parts.append(np.full(r_pos.size, seq_idx, dtype=np.int64))
            start_parts.append(r_pos.astype(np.int64, copy=False))
            strand_parts.append(np.full(r_pos.size, MINUS_STRAND, dtype=np.int8))
            score_parts.append(s_rev[r_pos].astype(np.float32, copy=False))

    if not seq_indices_parts:
        return _empty_hit_arrays()

    return {
        "seq_index": np.concatenate(seq_indices_parts),
        "start": np.concatenate(start_parts),
        "strand_idx": np.concatenate(strand_parts),
        "score": np.concatenate(score_parts),
    }


def _collect_hits(model: GenericModel, sequences, mode: str, score_threshold: Optional[float]) -> dict[str, np.ndarray]:
    """Collect motif hits as numeric arrays."""
    score_bundle = _scan_both_strands(model, sequences)
    if mode == "best":
        return _collect_best_hits(sequences, score_bundle)
    return _collect_threshold_hits(score_bundle, float(score_threshold))


def _scores_to_log_tail_array(scores: np.ndarray, threshold_table: np.ndarray) -> np.ndarray:
    """Convert one score array to log-tail values using an explicit lookup table."""
    if scores.size == 0:
        return np.empty(0, dtype=np.float64)

    scores_col = threshold_table[:, 0]
    log_tail_col = threshold_table[:, 1]
    idx = np.searchsorted(-scores_col, -scores.astype(np.float64, copy=False), side="left")
    idx = np.clip(idx, 0, len(log_tail_col) - 1)
    return log_tail_col[idx]


def _resolve_hit_threshold_table(model: GenericModel, sequences, background_sequences, threshold_table) -> np.ndarray:
    """Resolve the explicit log-tail table used for hit extraction and annotation."""
    if threshold_table is not None:
        return np.asarray(threshold_table, dtype=np.float64)

    calibration_sequences = background_sequences if background_sequences is not None else sequences
    return calculate_threshold_table(model, calibration_sequences, strand="best")


def _resolve_hits(model: GenericModel, sequences, selection: dict, *, include_threshold_table: bool) -> dict:
    """Resolve hit arrays and optional threshold metadata for one request."""
    mode = validate_site_mode(selection.get("mode", "best"), selection.get("fpr_threshold"))
    threshold_table = None
    score_threshold = None

    if include_threshold_table or mode == "threshold":
        threshold_table = _resolve_hit_threshold_table(
            model,
            sequences,
            selection.get("background_sequences"),
            selection.get("threshold_table"),
        )

    if mode == "threshold":
        score_threshold = lookup_score_for_tail_probability(threshold_table, float(selection["fpr_threshold"]))
        logging.getLogger(__name__).info(
            "FPR threshold: %s -> score threshold: %.4f", selection["fpr_threshold"], score_threshold
        )

    hit_arrays = _sort_hit_arrays(_collect_hits(model, sequences, mode, score_threshold))
    return {
        "hit_arrays": hit_arrays,
        "threshold_table": threshold_table,
        "score_threshold": score_threshold,
    }


def _extract_site_matrix(
    sequences, seq_indices: np.ndarray, starts: np.ndarray, motif_length: int, strand_indices=None
):
    """Extract numeric motif windows for a set of hits."""
    n_hits = seq_indices.size
    sites = np.empty((n_hits, motif_length), dtype=sequences["values"].dtype)

    for hit_idx in range(n_hits):
        seq = row_values(sequences, int(seq_indices[hit_idx]))
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

    valid_mask = site_matrix < _NUCLEOTIDE_CARDINALITY
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


def _empty_sites_frame() -> pd.DataFrame:
    """Return an empty site table with the public schema."""
    return pd.DataFrame(
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


def _build_sites_frame(
    model: GenericModel, sequences, hit_arrays: dict[str, np.ndarray], threshold_table: np.ndarray
) -> pd.DataFrame:
    """Build the public site table from resolved hits."""
    if hit_arrays["score"].size == 0:
        return _empty_sites_frame()

    site_matrix = _extract_site_matrix(
        sequences,
        hit_arrays["seq_index"],
        hit_arrays["start"],
        model.length,
        hit_arrays["strand_idx"],
    )
    return pd.DataFrame(
        {
            "seq_index": hit_arrays["seq_index"],
            "start": hit_arrays["start"],
            "end": hit_arrays["start"] + model.length,
            "strand": np.where(hit_arrays["strand_idx"] == 0, "+", "-"),
            "score": hit_arrays["score"],
            "log_tail": _scores_to_log_tail_array(hit_arrays["score"], threshold_table),
            "site": _site_matrix_to_strings(site_matrix),
        }
    )


def _hits_to_pfm(model: GenericModel, sequences, hit_arrays: dict[str, np.ndarray], pseudocount: float) -> np.ndarray:
    """Convert a selected hit set to a PFM."""
    if hit_arrays["score"].size == 0:
        raise ValueError("No sites found")

    site_matrix = _extract_site_matrix(
        sequences,
        hit_arrays["seq_index"],
        hit_arrays["start"],
        model.length,
        hit_arrays["strand_idx"],
    )
    pcm = _build_pcm_from_site_matrix(site_matrix, model.length)
    return pcm_to_pfm(pcm, pseudocount=pseudocount).astype(np.float32, copy=False)


def get_sites(
    model: GenericModel,
    sequences,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    background_sequences=None,
    threshold_table: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Find motif binding sites in sequences."""
    resolved = _resolve_hits(
        model,
        sequences,
        {
            "mode": mode,
            "fpr_threshold": fpr_threshold,
            "background_sequences": background_sequences,
            "threshold_table": threshold_table,
        },
        include_threshold_table=True,
    )
    df = _build_sites_frame(model, sequences, resolved["hit_arrays"], resolved["threshold_table"])
    logging.getLogger(__name__).info("Found %s site(s) in %s sequence(s)", len(df), len(sequences["lengths"]))
    return df


def get_pfm(
    model: GenericModel,
    sequences,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    background_sequences=None,
    threshold_table: Optional[np.ndarray] = None,
    top_fraction: Optional[float] = None,
    pseudocount: float = 0.25,
    force_recompute: bool = False,
) -> np.ndarray:
    """Construct a Position Frequency Matrix from binding sites."""
    del force_recompute

    logger = logging.getLogger(__name__)
    logger.info("Computing PFM for model: %s", model.name)
    resolved = _resolve_hits(
        model,
        sequences,
        {
            "mode": mode,
            "fpr_threshold": fpr_threshold,
            "background_sequences": background_sequences,
            "threshold_table": threshold_table,
        },
        include_threshold_table=mode == "threshold",
    )
    selected_hits = _select_top_hit_arrays(resolved["hit_arrays"], top_fraction)
    if top_fraction is not None:
        logger.info("Selected top %.1f%%: %s sites", top_fraction * 100.0, selected_hits["score"].size)
    return _hits_to_pfm(model, sequences, selected_hits, pseudocount)


def _scan_with_batch_kernel(model: GenericModel, sequences, strand: StrandMode, *, with_context: bool = False):
    """Scan a tensor-based motif model with the shared Numba batch kernel."""
    representation = np.asarray(model.representation, dtype=np.float32)
    kmer = int(model.config.get("kmer", 1))

    if strand == "+":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
    if strand == "-":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
    if strand == "best":
        sf, sr = batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)
        values = np.full(sf["values"].shape, SCORE_PADDING, dtype=np.float32)
        values[sf["mask"]] = np.maximum(sf["values"][sf["mask"]], sr["values"][sr["mask"]])
        return pack_batch(values, sf["mask"], sf["lengths"], SCORE_PADDING)
    raise ValueError(f"Invalid strand mode: {strand}")


def _scan_with_batch_kernel_strands(model: GenericModel, sequences, *, with_context: bool = False):
    """Scan a tensor-based motif model on both strands in one shared Numba call."""
    representation = np.asarray(model.representation, dtype=np.float32)
    kmer = int(model.config.get("kmer", 1))
    return batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)


def _score_bounds_from_model(model: GenericModel) -> tuple[float, float]:
    """Return theoretical min/max score for tensor-based motif models."""
    return _score_bounds_from_representation(np.asarray(model.representation))


def _scan_pwm(model: GenericModel, sequences, strand: StrandMode):
    return _scan_with_batch_kernel(model, sequences, strand, with_context=False)


def _scan_pwm_both(model: GenericModel, sequences):
    return _scan_with_batch_kernel_strands(model, sequences, with_context=False)


def _write_pwm(model: GenericModel, path: str) -> None:
    pfm = model.config.get("_source_pfm")
    if pfm is None:
        raise ValueError("PWM serialization requires the source PFM in model.config['_source_pfm'].")
    write_pfm(np.asarray(pfm, dtype=np.float32), model.name, model.length, path)


def _load_pwm(path: str, kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())

    if ext == ".pkl":
        model = joblib.load(path)
        if not isinstance(model, GenericModel):
            raise TypeError(f"Unsupported PWM pickle payload: expected GenericModel, got {type(model)!r}")
        if model.config.get("_source_pfm") is None:
            raise ValueError("Unsupported PWM pickle format: source PFM is missing from model.config['_source_pfm'].")
        return model

    if ext == ".meme":
        pfm, info, _ = read_meme(path, index=kwargs.get("index", 0))
        name, length = info
    elif ext == ".pfm":
        pfm, length = read_pfm(path)
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        raise ValueError(f"Unsupported PWM format: {path}")

    pwm = pfm_to_pwm(pfm)
    pwm_ext = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32, copy=False)
    return GenericModel("pwm", name, pwm_ext, int(length), {"kmer": 1, "_source_pfm": pfm})


def _scan_sitega(model: GenericModel, sequences, strand: StrandMode):
    return _scan_with_batch_kernel(model, sequences, strand, with_context=False)


def _scan_sitega_both(model: GenericModel, sequences):
    return _scan_with_batch_kernel_strands(model, sequences, with_context=False)


def _write_sitega(model: GenericModel, path: str) -> None:
    write_sitega(model, path)


def _load_sitega(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return joblib.load(path)
    if ext != ".mat":
        raise ValueError(f"Unsupported SiteGA format: {path}")

    representation, name, length, minimum, maximum = read_sitega(path)
    return GenericModel(
        "sitega",
        name,
        np.asarray(representation, dtype=np.float32),
        int(length),
        {"kmer": 2, "minimum": float(minimum), "maximum": float(maximum)},
    )


def _sitega_score_bounds(model: GenericModel) -> tuple[float, float]:
    minimum = model.config.get("minimum")
    maximum = model.config.get("maximum")
    if minimum is not None and maximum is not None:
        return float(minimum), float(maximum)
    return _score_bounds_from_model(model)


def _scan_bamm(model: GenericModel, sequences, strand: StrandMode):
    return _scan_with_batch_kernel(model, sequences, strand, with_context=True)


def _scan_bamm_both(model: GenericModel, sequences):
    return _scan_with_batch_kernel_strands(model, sequences, with_context=True)


def _load_bamm(path: str, kwargs: dict) -> GenericModel:
    if not path.endswith(".ihbcp") and not os.path.exists(path):
        ihbcp_path = f"{path}.ihbcp"
        if os.path.exists(ihbcp_path):
            path = ihbcp_path
        else:
            raise FileNotFoundError(f"BaMM file not found: {path}")

    _, max_order, _ = parse_file_content(path)
    target_order = kwargs.get("order")
    target_order = max_order if target_order is None else min(int(target_order), max_order)
    representation = read_bamm(path, target_order)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel(
        "bamm",
        name,
        np.asarray(representation, dtype=np.float32),
        representation.shape[-1],
        {"kmer": representation.ndim - 1, "order": target_order},
    )


def _dump_model(model: GenericModel, path: str) -> None:
    joblib.dump(model, path)


def _scan_dimont(model: GenericModel, sequences, strand: StrandMode):
    return _scan_with_batch_kernel(model, sequences, strand, with_context=int(model.config.get("kmer", 1)) > 1)


def _scan_dimont_both(model: GenericModel, sequences):
    return _scan_with_batch_kernel_strands(model, sequences, with_context=int(model.config.get("kmer", 1)) > 1)


def _load_dimont(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return joblib.load(path)
    if ext != ".xml":
        raise ValueError(f"Unsupported Dimont format: {path}")

    representation, length, span = read_dimont(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("dimont", name, np.asarray(representation, dtype=np.float32), length, {"kmer": span + 1})


def _scan_slim(model: GenericModel, sequences, strand: StrandMode):
    return _scan_with_batch_kernel(model, sequences, strand, with_context=int(model.config.get("kmer", 1)) > 1)


def _scan_slim_both(model: GenericModel, sequences):
    return _scan_with_batch_kernel_strands(model, sequences, with_context=int(model.config.get("kmer", 1)) > 1)


def _load_slim(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return joblib.load(path)
    if ext != ".xml":
        raise ValueError(f"Unsupported Slim format: {path}")

    representation, length, span = read_slim(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("slim", name, np.asarray(representation, dtype=np.float32), length, {"kmer": span + 1})


def _scan_scores(model: GenericModel, _sequences=None, _strand: StrandMode = "best"):
    return model.config["scores_data"]


def _scan_scores_both(model: GenericModel, _sequences=None):
    scores = model.config["scores_data"]
    return scores, scores


def _write_scores(_model: GenericModel, _path: str) -> None:
    raise NotImplementedError("Score profiles cannot be written to files")


def _scores_score_bounds(model: GenericModel) -> tuple[float, float]:
    values = model.config["scores_data"]["values"]
    mask = model.config["scores_data"]["mask"]
    if not np.any(mask):
        return 0.0, 0.0
    valid = values[mask]
    return float(np.min(valid)), float(np.max(valid))


def _load_scores(path: str, _kwargs: dict) -> GenericModel:
    scores_data = read_scores(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("scores", name, None, 0, {"scores_data": scores_data})


_register_model_handler(
    "pwm",
    scan=_scan_pwm,
    scan_both=_scan_pwm_both,
    load=_load_pwm,
    write=_write_pwm,
    score_bounds=_score_bounds_from_model,
)
_register_model_handler(
    "sitega",
    scan=_scan_sitega,
    scan_both=_scan_sitega_both,
    load=_load_sitega,
    write=_write_sitega,
    score_bounds=_sitega_score_bounds,
)
_register_model_handler(
    "bamm",
    scan=_scan_bamm,
    scan_both=_scan_bamm_both,
    load=_load_bamm,
    write=_dump_model,
    score_bounds=_score_bounds_from_model,
)
_register_model_handler(
    "dimont",
    scan=_scan_dimont,
    scan_both=_scan_dimont_both,
    load=_load_dimont,
    write=_dump_model,
    score_bounds=_score_bounds_from_model,
)
_register_model_handler(
    "slim",
    scan=_scan_slim,
    scan_both=_scan_slim_both,
    load=_load_slim,
    write=_dump_model,
    score_bounds=_score_bounds_from_model,
)
_register_model_handler(
    "scores",
    scan=_scan_scores,
    scan_both=_scan_scores_both,
    load=_load_scores,
    write=_write_scores,
    score_bounds=_scores_score_bounds,
)
