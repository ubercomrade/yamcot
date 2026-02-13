"""
Functional Pipeline Module
========================

This module provides a functional programming approach to pipeline orchestration,
replacing class-based pipelines with pure functions and declarative composition.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from mimosa.comparison import ComparatorConfig, compare, create_comparator_config
from mimosa.io import read_fasta
from mimosa.models import read_model
from mimosa.ragged import RaggedData, ragged_from_list


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for pipeline execution."""

    model1_path: Optional[Path] = None
    model2_path: Optional[Path] = None
    model1_type: Optional[str] = None
    model2_type: Optional[str] = None
    comparison_type: str = "motif"
    seq_source1: Optional[Path] = None
    seq_source2: Optional[Path] = None
    output_path: Optional[Path] = None
    comparator_config: Optional[ComparatorConfig] = None
    num_sequences: int = 1000
    seq_length: int = 200
    seed: int = 127


class WorkflowRegistry:
    """Registry for workflow patterns using decorator pattern."""

    def __init__(self):
        self._workflows: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(fn):
            self._workflows[name] = fn
            return fn

        return decorator

    def get(self, name: str):
        return self._workflows.get(name)


workflow_registry = WorkflowRegistry()


def load_sequences(seq_source: Optional[Path], config: PipelineConfig) -> RaggedData:
    """Load or generate sequences for pipeline."""
    if seq_source and Path(seq_source).exists():
        return read_fasta(seq_source)
    else:
        rng = np.random.default_rng(config.seed)
        sequences = []
        for _ in range(config.num_sequences):
            seq = rng.integers(0, 4, size=config.seq_length, dtype=np.int8)
            sequences.append(seq)
        return ragged_from_list(sequences, dtype=np.int8)


def load_profile(profile_path: Union[str, Path], config: PipelineConfig) -> RaggedData:
    """Load profile from text file."""
    path = Path(profile_path)
    scores_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            if "," in line:
                scores = [float(x) for x in line.split(",")]
            elif "	" in line:
                scores = [float(x) for x in line.split("	")]
            else:
                scores = [float(x) for x in line.split()]
            scores_list.append(np.array(scores, dtype=np.float32))

    return (
        ragged_from_list(scores_list, dtype=np.float32)
        if scores_list
        else RaggedData(np.empty(0, dtype=np.float32), np.zeros(1, dtype=np.int64))
    )


def validate_pipeline_inputs(
    model1_path: Optional[Path],
    model2_path: Optional[Path],
    model1_type: Optional[str],
    model2_type: Optional[str],
    comparison_type: str,
) -> None:
    """Validate pipeline inputs."""
    if model1_path and not Path(model1_path).exists():
        raise FileNotFoundError(f"Model 1 file not found: {model1_path}")
    if model2_path and not Path(model2_path).exists():
        raise FileNotFoundError(f"Model 2 file not found: {model2_path}")
    valid_comparison_types = ["profile", "motif", "motali", "tomtom-like"]
    if comparison_type not in valid_comparison_types:
        raise ValueError(f"Invalid comparison type: {comparison_type}")


def run_pipeline(
    model1_path: Union[str, Path],
    model2_path: Union[str, Path],
    model1_type: str,
    model2_type: str,
    comparison_type: str = "motif",
    seq_source1: Optional[Union[str, Path]] = None,
    seq_source2: Optional[Union[str, Path]] = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    **kwargs,
) -> dict:
    """Run the pipeline using functional approach."""
    model1_path = Path(model1_path)
    model2_path = Path(model2_path)
    seq_source1 = Path(seq_source1) if seq_source1 else None
    seq_source2 = Path(seq_source2) if seq_source2 else None

    config = PipelineConfig(
        model1_path=model1_path,
        model2_path=model2_path,
        model1_type=model1_type,
        model2_type=model2_type,
        seq_source1=seq_source1,
        seq_source2=seq_source2,
        num_sequences=num_sequences,
        seq_length=seq_length,
        comparison_type=comparison_type,
        seed=seed,
    )

    comparator_config = create_comparator_config(**kwargs)
    validate_pipeline_inputs(model1_path, model2_path, model1_type, model2_type, comparison_type)

    model1 = read_model(str(model1_path), model1_type)
    model2 = read_model(str(model2_path), model2_type)

    sequences = load_sequences(seq_source1, config)
    promoters = load_sequences(seq_source2, config)

    # Map comparison type to strategy
    strategy_map = {"motif": "universal", "motali": "motali", "tomtom-like": "tomtom", "profile": "universal"}

    strategy = strategy_map.get(comparison_type, "universal")

    result = compare(
        model1=model1,
        model2=model2,
        strategy=strategy,
        config=comparator_config,
        sequences=sequences,
        promoters=promoters,
    )

    return result
