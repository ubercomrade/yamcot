"""High-level public API for motif comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union

import numpy as np

from mimosa.batches import SequenceBatch, make_sequence_batch
from mimosa.comparison import (
    SUPPORTED_MOTIF_METRICS,
    SUPPORTED_PROFILE_METRICS,
    ComparatorConfig,
    compare,
    create_comparator_config,
)
from mimosa.comparison import (
    compare_one_to_many as compare_one_to_many_models,
)
from mimosa.io import read_fasta
from mimosa.models import GenericModel, read_model
from mimosa.validation import validate_file_exists, validate_positive_int

ModelRef = Union[GenericModel, str, Path]
SequenceRef = Union[SequenceBatch, str, Path]

_STRATEGY_ALIASES = {
    "profile": "profile",
    "motif": "motif",
}

_DEFAULT_METRICS = {
    "profile": "co",
    "motif": "pcc",
}

_ALLOWED_METRICS = {
    "profile": frozenset(SUPPORTED_PROFILE_METRICS),
    "motif": frozenset(SUPPORTED_MOTIF_METRICS),
}


class ComparisonConfig(TypedDict):
    model1: ModelRef
    model2: ModelRef
    model1_type: Optional[str]
    model2_type: Optional[str]
    strategy: str
    sequences: Optional[SequenceRef]
    background: Optional[SequenceRef]
    num_sequences: int
    seq_length: int
    seed: int
    comparator: ComparatorConfig
    model1_kwargs: Dict[str, Any]
    model2_kwargs: Dict[str, Any]


class OneToManyConfig(TypedDict):
    query: ModelRef
    targets: List[ModelRef]
    query_type: Optional[str]
    target_type: Optional[str]
    strategy: str
    sequences: Optional[SequenceRef]
    background: Optional[SequenceRef]
    num_sequences: int
    seq_length: int
    seed: int
    comparator: ComparatorConfig
    query_kwargs: Dict[str, Any]
    target_kwargs: Dict[str, Any]


def create_config(
    model1: ModelRef,
    model2: ModelRef,
    model1_type: Optional[str] = None,
    model2_type: Optional[str] = None,
    strategy: str = "profile",
    sequences: Optional[SequenceRef] = None,
    background: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[ComparatorConfig] = None,
    model1_kwargs: Optional[Dict[str, Any]] = None,
    model2_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> ComparisonConfig:
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
        "background": background,
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
    background: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[ComparatorConfig] = None,
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
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        model1_kwargs=model1_kwargs,
        model2_kwargs=model2_kwargs,
        **comparator_kwargs,
    )
    return run_comparison(config)


def create_many_config(
    query: ModelRef,
    targets: List[ModelRef],
    query_type: Optional[str] = None,
    target_type: Optional[str] = None,
    strategy: str = "profile",
    sequences: Optional[SequenceRef] = None,
    background: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[ComparatorConfig] = None,
    query_kwargs: Optional[Dict[str, Any]] = None,
    target_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> OneToManyConfig:
    """Build a unified one-vs-many comparison configuration."""
    normalized_strategy = _normalize_strategy(strategy)
    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    effective_kwargs = dict(comparator_kwargs)
    default_metric = _DEFAULT_METRICS.get(normalized_strategy)
    if comparator is None and default_metric is not None and "metric" not in effective_kwargs:
        effective_kwargs["metric"] = default_metric

    resolved_comparator = comparator or create_comparator_config(**effective_kwargs)
    return {
        "query": query,
        "targets": _normalize_targets(targets),
        "query_type": query_type,
        "target_type": target_type,
        "strategy": normalized_strategy,
        "sequences": sequences,
        "background": background,
        "num_sequences": num_sequences,
        "seq_length": seq_length,
        "seed": seed,
        "comparator": resolved_comparator,
        "query_kwargs": query_kwargs or {},
        "target_kwargs": target_kwargs or {},
    }


def compare_one_to_many(
    query: ModelRef,
    targets: List[ModelRef],
    query_type: Optional[str] = None,
    target_type: Optional[str] = None,
    strategy: str = "profile",
    sequences: Optional[SequenceRef] = None,
    background: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[ComparatorConfig] = None,
    query_kwargs: Optional[Dict[str, Any]] = None,
    target_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> List[dict]:
    """Single-call entry point for one-vs-many motif comparison."""
    config = create_many_config(
        query=query,
        targets=targets,
        query_type=query_type,
        target_type=target_type,
        strategy=strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        query_kwargs=query_kwargs,
        target_kwargs=target_kwargs,
        **comparator_kwargs,
    )
    return run_one_to_many(config)


def run_comparison(config: ComparisonConfig) -> dict:
    """Execute one comparison using the unified config."""
    strategy = _normalize_strategy(config["strategy"])
    model1 = _resolve_model(config["model1"], config.get("model1_type"), config.get("model1_kwargs", {}))
    model2 = _resolve_model(config["model2"], config.get("model2_type"), config.get("model2_kwargs", {}))
    _validate_models_for_strategy(strategy, model1, model2)
    _validate_comparator_for_strategy(strategy, config["comparator"])

    background = None
    if config.get("background") is not None:
        background = _resolve_sequences(config["background"], config)

    needs_sequences = _needs_sequences(strategy, config["comparator"], model1, model2)
    sequences = _resolve_sequences(config.get("sequences"), config) if needs_sequences else None

    return compare(
        model1=model1,
        model2=model2,
        strategy=strategy,
        config=config["comparator"],
        sequences=sequences,
        background=background,
    )


def run_one_to_many(config: OneToManyConfig) -> List[dict]:
    """Execute one comparison of a single query against many targets."""
    strategy = _normalize_strategy(config["strategy"])
    query_model = _resolve_model(config["query"], config.get("query_type"), config.get("query_kwargs", {}))
    _validate_comparator_for_strategy(strategy, config["comparator"])

    target_refs = list(config.get("targets", []))
    if not target_refs:
        return []

    background = None
    if config.get("background") is not None:
        background = _resolve_sequences(config["background"], config)

    target_models = _resolve_target_models(
        target_refs,
        config.get("target_type"),
        config.get("target_kwargs", {}),
        strategy,
        query_model,
    )
    needs_sequences = any(
        _needs_sequences(strategy, config["comparator"], query_model, target_model)
        for target_model in target_models
    )
    sequences = _resolve_sequences(config.get("sequences"), config) if needs_sequences else None

    return compare_one_to_many_models(
        query_model=query_model,
        target_models=iter(target_models),
        strategy=strategy,
        config=config["comparator"],
        sequences=sequences,
        background=background,
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


def _normalize_targets(targets) -> List[ModelRef]:
    """Normalize one targets collection and reject scalar inputs."""
    if isinstance(targets, (str, Path, GenericModel)):
        raise TypeError("targets must be a list of model references, not a single model.")
    return list(targets)


def _resolve_target_models(
    targets: Iterable[ModelRef],
    model_type: Optional[str],
    kwargs: Dict[str, Any],
    strategy: str,
    query_model: GenericModel,
) -> tuple[GenericModel, ...]:
    """Resolve and validate target models once for one-vs-many comparison."""
    resolved_targets: list[GenericModel] = []
    for target in targets:
        target_model = _resolve_model(target, model_type, kwargs)
        _validate_models_for_strategy(strategy, query_model, target_model)
        resolved_targets.append(target_model)
    return tuple(resolved_targets)


def _needs_sequences(strategy: str, comparator: ComparatorConfig, model1: GenericModel, model2: GenericModel) -> bool:
    """Return True if the selected strategy requires sequence input."""
    if strategy == "profile":
        return model1.type_key != "scores" or model2.type_key != "scores"
    return strategy == "motif" and (comparator["pfm_mode"] or model1.type_key != model2.type_key)


def _validate_models_for_strategy(strategy: str, model1: GenericModel, model2: GenericModel) -> None:
    """Validate model combinations for the selected comparison strategy."""
    if strategy == "motif" and ("scores" in {model1.type_key, model2.type_key}):
        raise ValueError("Motif strategy does not support score-profile inputs.")

def _validate_comparator_for_strategy(strategy: str, comparator: ComparatorConfig) -> None:
    """Validate comparator options for the selected strategy."""
    allowed_metrics = _ALLOWED_METRICS.get(strategy)
    if allowed_metrics is not None and comparator["metric"] not in allowed_metrics:
        options = ", ".join(sorted(allowed_metrics))
        raise ValueError(f"Strategy '{strategy}' requires one of the following metrics: {options}")


def _resolve_sequences(source: Optional[SequenceRef], config: ComparisonConfig | OneToManyConfig) -> SequenceBatch:
    """Resolve one sequence source to a padded sequence batch."""
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
