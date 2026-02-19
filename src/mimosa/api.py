"""High-level public API for motif comparison."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from mimosa.comparison import ComparatorConfig, compare, create_comparator_config
from mimosa.io import read_fasta
from mimosa.models import GenericModel, read_model
from mimosa.ragged import RaggedData, ragged_from_list

ModelRef = Union[GenericModel, str, Path]
SequenceRef = Union[RaggedData, str, Path]

_STRATEGY_ALIASES = {
    "motif": "universal",
    "profile": "universal",
    "universal": "universal",
    "tomtom": "tomtom",
    "tomtom-like": "tomtom",
    "motali": "motali",
}


@dataclass
class ComparisonConfig:
    """Unified configuration object for library usage."""

    model1: ModelRef
    model2: ModelRef
    model1_type: Optional[str] = None
    model2_type: Optional[str] = None
    strategy: str = "universal"
    sequences: Optional[SequenceRef] = None
    promoters: Optional[SequenceRef] = None
    num_sequences: int = 1000
    seq_length: int = 200
    seed: int = 127
    comparator: ComparatorConfig = field(default_factory=create_comparator_config)
    model1_kwargs: Dict[str, Any] = field(default_factory=dict)
    model2_kwargs: Dict[str, Any] = field(default_factory=dict)


def create_config(
    model1: ModelRef,
    model2: ModelRef,
    model1_type: Optional[str] = None,
    model2_type: Optional[str] = None,
    strategy: str = "universal",
    sequences: Optional[SequenceRef] = None,
    promoters: Optional[SequenceRef] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: Optional[ComparatorConfig] = None,
    model1_kwargs: Optional[Dict[str, Any]] = None,
    model2_kwargs: Optional[Dict[str, Any]] = None,
    **comparator_kwargs,
) -> ComparisonConfig:
    """Build a unified comparison config."""

    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    resolved_comparator = comparator or create_comparator_config(**comparator_kwargs)

    return ComparisonConfig(
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
        comparator=resolved_comparator,
        model1_kwargs=model1_kwargs or {},
        model2_kwargs=model2_kwargs or {},
    )


def compare_motifs(
    model1: ModelRef,
    model2: ModelRef,
    model1_type: Optional[str] = None,
    model2_type: Optional[str] = None,
    strategy: str = "universal",
    sequences: Optional[SequenceRef] = None,
    promoters: Optional[SequenceRef] = None,
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


def run_comparison(config: ComparisonConfig) -> dict:
    """Execute a comparison using the unified config."""

    strategy = _normalize_strategy(config.strategy)
    model1 = _resolve_model(config.model1, config.model1_type, config.model1_kwargs)
    model2 = _resolve_model(config.model2, config.model2_type, config.model2_kwargs)

    needs_sequences = _needs_sequences(strategy, config.comparator, model1, model2)
    sequences = _resolve_sequences(config.sequences, config) if needs_sequences else None

    promoters = None
    if config.promoters is not None:
        promoters = _resolve_sequences(config.promoters, config)

    if strategy == "motali" and sequences is None:
        sequences = promoters

    return compare(
        model1=model1,
        model2=model2,
        strategy=strategy,
        config=config.comparator,
        sequences=sequences,
        promoters=promoters,
    )


def _normalize_strategy(strategy: str) -> str:
    """Normalize strategy aliases to internal names."""

    resolved = _STRATEGY_ALIASES.get(strategy.lower())
    if resolved is None:
        available = ", ".join(sorted(_STRATEGY_ALIASES.keys()))
        raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")
    return resolved


def _resolve_model(model: ModelRef, model_type: Optional[str], kwargs: Dict[str, Any]) -> GenericModel:
    """Convert a model reference to GenericModel."""

    if isinstance(model, GenericModel):
        return model

    if isinstance(model, (str, Path)):
        if model_type is None:
            raise ValueError("model_type is required when model is provided as a file path.")
        return read_model(str(model), model_type, **kwargs)

    raise TypeError(f"Unsupported model reference type: {type(model)!r}")


def _needs_sequences(strategy: str, comparator: ComparatorConfig, model1: GenericModel, model2: GenericModel) -> bool:
    """Return True if selected strategy requires sequence input."""

    if strategy in {"universal", "motali"}:
        return True
    if strategy == "tomtom" and (comparator.pfm_mode or model1.type_key != model2.type_key):
        return True
    return False


def _resolve_sequences(source: Optional[SequenceRef], config: ComparisonConfig) -> Optional[RaggedData]:
    """Resolve a sequence source to RaggedData."""

    if source is None:
        return _generate_random_sequences(config.num_sequences, config.seq_length, config.seed)
    if isinstance(source, RaggedData):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")
        return read_fasta(path)
    raise TypeError(f"Unsupported sequence source type: {type(source)!r}")


def _generate_random_sequences(num_sequences: int, seq_length: int, seed: int) -> RaggedData:
    """Generate random A/C/G/T integer-encoded sequences."""

    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")

    rng = np.random.default_rng(seed)
    sequences = [rng.integers(0, 4, size=seq_length, dtype=np.int8) for _ in range(num_sequences)]
    return ragged_from_list(sequences, dtype=np.int8)
