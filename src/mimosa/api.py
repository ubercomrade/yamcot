"""High-level public API for motif comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from mimosa.batches import make_sequence_batch
from mimosa.comparison import compare, create_comparator_config
from mimosa.io import read_fasta
from mimosa.models import GenericModel, read_model
from mimosa.validation import validate_file_exists, validate_positive_int

ModelRef = Union[GenericModel, str, Path]
SequenceRef = Union[dict, str, Path]

_STRATEGY_ALIASES = {
    "profile": "profile",
    "motif": "motif",
    "motali": "motali",
}

_DEFAULT_METRICS = {
    "profile": "co",
    "motif": "pcc",
}

_ALLOWED_METRICS = {
    "profile": {"co", "dice"},
    "motif": {"pcc", "ed", "cosine"},
}


def create_config(
    model1: ModelRef,
    model2: ModelRef,
    model1_type: Optional[str] = None,
    model2_type: Optional[str] = None,
    strategy: str = "profile",
    sequences: Optional[SequenceRef] = None,
    promoters: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[dict] = None,
    model1_kwargs: Optional[Dict[str, Any]] = None,
    model2_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> dict:
    """Build a unified comparison configuration."""
    normalized_strategy = _normalize_strategy(strategy)
    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    effective_kwargs = dict(comparator_kwargs)
    default_metric = _DEFAULT_METRICS.get(normalized_strategy)
    if comparator is None and default_metric is not None and "metric" not in effective_kwargs:
        effective_kwargs["metric"] = default_metric

    resolved_comparator = comparator or create_comparator_config(**effective_kwargs)
    return {
        "model1": model1,
        "model2": model2,
        "model1_type": model1_type,
        "model2_type": model2_type,
        "strategy": normalized_strategy,
        "sequences": sequences,
        "promoters": promoters,
        "num_sequences": num_sequences,
        "seq_length": seq_length,
        "seed": seed,
        "comparator": resolved_comparator,
        "model1_kwargs": model1_kwargs or {},
        "model2_kwargs": model2_kwargs or {},
    }


def compare_motifs(
    model1: ModelRef,
    model2: ModelRef,
    model1_type: Optional[str] = None,
    model2_type: Optional[str] = None,
    strategy: str = "profile",
    sequences: Optional[SequenceRef] = None,
    promoters: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[dict] = None,
    model1_kwargs: Optional[Dict[str, Any]] = None,
    model2_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> dict:
    """Single-call entry point for motif comparison."""
    config = create_config(
        model1=model1,
        model2=model2,
        model1_type=model1_type,
        model2_type=model2_type,
        strategy=strategy,
        sequences=sequences,
        promoters=promoters,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        model1_kwargs=model1_kwargs,
        model2_kwargs=model2_kwargs,
        **comparator_kwargs,
    )
    return run_comparison(config)


def run_comparison(config: dict) -> dict:
    """Execute one comparison using the unified config."""
    strategy = _normalize_strategy(config["strategy"])
    model1 = _resolve_model(config["model1"], config.get("model1_type"), config.get("model1_kwargs", {}))
    model2 = _resolve_model(config["model2"], config.get("model2_type"), config.get("model2_kwargs", {}))
    _validate_models_for_strategy(strategy, model1, model2)
    _validate_comparator_for_strategy(strategy, config["comparator"])

    promoters = None
    if config.get("promoters") is not None:
        promoters = _resolve_sequences(config["promoters"], config)

    needs_sequences = _needs_sequences(strategy, config["comparator"], model1, model2)
    if strategy == "motali" and config.get("sequences") is None and promoters is not None:
        sequences = None
    else:
        sequences = _resolve_sequences(config.get("sequences"), config) if needs_sequences else None

    if strategy == "motali" and sequences is None:
        sequences = promoters

    return compare(
        model1=model1,
        model2=model2,
        strategy=strategy,
        config=config["comparator"],
        sequences=sequences,
        promoters=promoters,
    )


def _normalize_strategy(strategy: str) -> str:
    """Normalize strategy aliases to internal names."""
    resolved = _STRATEGY_ALIASES.get(strategy.lower())
    if resolved is None:
        available = ", ".join(sorted(_STRATEGY_ALIASES))
        raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")
    return resolved


def _resolve_model(model: ModelRef, model_type: Optional[str], kwargs: Dict[str, Any]) -> GenericModel:
    """Convert one model reference to GenericModel."""
    if isinstance(model, GenericModel):
        return model
    if isinstance(model, (str, Path)):
        if model_type is None:
            raise ValueError("model_type is required when model is provided as a file path.")
        return read_model(str(model), model_type, **kwargs)
    raise TypeError(f"Unsupported model reference type: {type(model)!r}")


def _needs_sequences(strategy: str, comparator: dict, model1: GenericModel, model2: GenericModel) -> bool:
    """Return True if the selected strategy requires sequence input."""
    if strategy == "profile":
        return model1.type_key != "scores" or model2.type_key != "scores"
    if strategy == "motali":
        return True
    return strategy == "motif" and (comparator["pfm_mode"] or model1.type_key != model2.type_key)


def _validate_models_for_strategy(strategy: str, model1: GenericModel, model2: GenericModel) -> None:
    """Validate model combinations for the selected comparison strategy."""
    if strategy == "motif" and ("scores" in {model1.type_key, model2.type_key}):
        raise ValueError("Motif strategy does not support score-profile inputs.")

    if strategy == "motali":
        invalid = {model1.type_key, model2.type_key} - {"pwm", "sitega"}
        if invalid:
            invalid_types = ", ".join(sorted(invalid))
            raise ValueError(f"Motali strategy supports only pwm and sitega models, got: {invalid_types}")


def _validate_comparator_for_strategy(strategy: str, comparator: dict) -> None:
    """Validate comparator options for the selected strategy."""
    allowed_metrics = _ALLOWED_METRICS.get(strategy)
    if allowed_metrics is not None and comparator["metric"] not in allowed_metrics:
        options = ", ".join(sorted(allowed_metrics))
        raise ValueError(f"Strategy '{strategy}' requires one of the following metrics: {options}")


def _resolve_sequences(source: Optional[SequenceRef], config: dict):
    """Resolve one sequence source to a dense masked batch."""
    if source is None:
        return _generate_random_sequences(config["num_sequences"], config["seq_length"], config["seed"])
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        path = validate_file_exists(source, "Sequence file")
        return read_fasta(path)
    raise TypeError(f"Unsupported sequence source type: {type(source)!r}")


def _generate_random_sequences(num_sequences: int, seq_length: int, seed: int):
    """Generate random A/C/G/T integer-encoded sequences."""
    num_sequences = validate_positive_int("num_sequences", num_sequences)
    seq_length = validate_positive_int("seq_length", seq_length)
    rng = np.random.default_rng(seed)
    rows = [rng.integers(0, 4, size=seq_length, dtype=np.int8) for _ in range(num_sequences)]
    return make_sequence_batch(rows)
