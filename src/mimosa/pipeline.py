"""
Unified pipeline for motif comparison operations.
This module provides a unified interface for score-based, sequence-based, and tomtom-like comparisons.
"""

import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from mimosa.comparison import DataComparator, MotaliComparator, TomtomComparator, UniversalMotifComparator
from mimosa.functions import scores_to_frequencies
from mimosa.io import read_fasta
from mimosa.models import MotifModel
from mimosa.ragged import RaggedData, ragged_from_list


class Pipeline:
    """
    Unified pipeline for motif comparison operations.

    This class handles score-based, sequence-based, and tomtom-like comparison paths,
    supporting various model types (BAMM, PWM, Sitega) and comparison methods.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_scores(self, profile_path: Union[str, Path]) -> RaggedData:
        """
        Load pre-calculated scores from text-based profile files in FASTA-like format.
        Each entry consists of a header line starting with '>' (containing metadata)
        followed by a line containing numerical scores separated by commas or tabs.
        The method also handles files without header lines (lines starting with '>').

        Args:
            profile_path: Path to the profile file containing scores in text format

        Returns:
            RaggedData object containing loaded scores
        """

        path = Path(profile_path)
        scores_list = []

        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                if line.startswith(">"):
                    # Header line, skip it
                    continue
                else:
                    # Parse scores from this line
                    # Try to split by comma first, then tab, then space
                    if "," in line:
                        scores = [float(x) for x in line.split(",")]
                    elif "\t" in line:
                        scores = [float(x) for x in line.split("\t")]
                    else:
                        # Default to splitting by whitespace
                        scores = [float(x) for x in line.split()]
                    scores_list.append(np.array(scores, dtype=np.float32))

        # Convert the list of score arrays to RaggedData
        if not scores_list:
            # Return empty RaggedData if no scores were found
            return RaggedData(np.empty(0, dtype=np.float32), np.zeros(1, dtype=np.int64))

        return ragged_from_list(scores_list, dtype=np.float32)

    def normalize_scores(self, scores: RaggedData) -> RaggedData:
        """
        Normalize scores using scores_to_frequencies function.

        Args:
            scores: RaggedData containing raw scores

        Returns:
            Normalized frequency data as RaggedData
        """
        return scores_to_frequencies(scores)

    def execute_score_comparison(
        self, profile1_path: Union[str, Path], profile2_path: Union[str, Path], **kwargs
    ) -> Any:
        """
        Execute score-based comparison using DataComparator.

        Args:
            profile1_path: Path to first profile file
            profile2_path: Path to second profile file
            **kwargs: Additional arguments for comparison

        Returns:
            Comparison results
        """
        # Load scores from both profiles
        scores1 = self.load_scores(profile1_path)
        scores2 = self.load_scores(profile2_path)

        # Normalize the scores
        freq1 = self.normalize_scores(scores1)
        freq2 = self.normalize_scores(scores2)

        # Sanitize kwargs for DataComparator
        data_kwargs = {}
        for param in [
            "name",
            "metric",
            "n_permutations",
            "distortion_level",
            "n_jobs",
            "seed",
            "filter_type",
            "filter_threshold",
            "search_range",
            "min_kernel_size",
            "max_kernel_size",
        ]:
            if param in kwargs:
                data_kwargs[param] = kwargs[param]
        comparator = DataComparator(**data_kwargs)
        return comparator.compare(freq1, freq2)

    def load_sequences(
        self, seq_source: Union[str, Path, None], num_sequences: int = 1000, seq_length: int = 200
    ) -> RaggedData:
        """
        Load sequences from source or generate them randomly.

        Args:
            seq_source: Path to sequence file or None to generate randomly
            num_sequences: Number of sequences to generate if needed
            seq_length: Length of sequences to generate if needed

        Returns:
            RaggedData object containing sequences
        """
        if seq_source is not None:
            # Load sequences from file
            return read_fasta(str(seq_source))
        else:
            # Generate random sequences
            sequences = []
            for _ in range(num_sequences):
                seq = self._generate_random_sequence(seq_length)
                sequences.append(self._encode_sequence(seq))

            return ragged_from_list(sequences, dtype=np.int8)

    def _encode_sequence(self, seq: str) -> np.ndarray:
        """Encode a DNA sequence string to integer representation."""
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        return np.array([base_map.get(base.upper(), 4) for base in seq], dtype=np.int8)

    def _generate_random_sequence(self, length: int) -> str:
        """Generate a random DNA sequence of specified length."""
        bases = ["A", "C", "G", "T"]
        return "".join(random.choice(bases) for _ in range(length))

    def execute_tomtom_comparison(
        self, model1: MotifModel, model2: MotifModel, sequences: Optional[RaggedData], **kwargs
    ) -> Any:
        """
        Execute TomTom-like comparison using TomtomComparator.

        Args:
            model1: First model
            model2: Second model
            sequences: Set of sequences as RaggedData (for model conversion if needed)
            **kwargs: Additional arguments for comparison

        Returns:
            Comparison results
        """
        # Sanitize kwargs for TomtomComparator - remove unsupported parameters
        tom_kwargs = {}
        for param in ["metric", "n_permutations", "permute_rows", "n_jobs", "seed", "pfm_mode"]:
            if param in kwargs:
                tom_kwargs[param] = kwargs[param]
        comparator = TomtomComparator(**tom_kwargs)
        return comparator.compare(model1, model2, sequences)

    def execute_motif_comparison(
        self,
        model1: MotifModel,
        model2: MotifModel,
        sequences: RaggedData,
        promoters: RaggedData,
        comparison_type: str = "motif",
        **kwargs,
    ) -> Any:
        """
        Execute sequence-based comparison using appropriate comparator.

        Args:
            model1: First model
            model2: Second model
            sequences: Set of sequences as RaggedData
            promoters: Set of promoter sequences as RaggedData
            comparison_type: Type of comparison ('motif', 'motali')
            **kwargs: Additional arguments for comparison

        Returns:
            Comparison results
        """
        if comparison_type.lower() == "motif":
            # Sanitize kwargs for MotifComparator
            motif_kwargs = {}
            for param in [
                "name",
                "metric",
                "n_permutations",
                "distortion_level",
                "n_jobs",
                "seed",
                "filter_type",
                "filter_threshold",
                "search_range",
                "min_kernel_size",
                "max_kernel_size",
            ]:
                if param in kwargs:
                    motif_kwargs[param] = kwargs[param]

            comparator = UniversalMotifComparator(**motif_kwargs)
            return comparator.compare(model1, model2, sequences)

        elif comparison_type.lower() == "motali":
            # Ensure threshold table is calculated for both models
            # Always use the promoters argument for threshold calculation
            if not hasattr(model1, "_threshold_table") or model1._threshold_table is None:
                model1.get_threshold_table(promoters)
            if not hasattr(model2, "_threshold_table") or model2._threshold_table is None:
                model2.get_threshold_table(promoters)

            # MotaliComparator needs a fasta file path
            motali_kwargs = {}
            for param in ["fasta_path", "threshold", "tmp_directory"]:
                if param in kwargs:
                    motali_kwargs[param] = kwargs[param]

            comparator = MotaliComparator(**motali_kwargs)
            return comparator.compare(model1, model2, sequences)

        else:
            raise ValueError(f"Unknown comparison type: {comparison_type}")

    def run_pipeline(
        self,
        model1_path: Union[str, Path],
        model2_path: Union[str, Path],
        model1_type: str,
        model2_type: str,
        comparison_type: str = "motif",
        seq_source1: Optional[Union[str, Path]] = None,
        seq_source2: Optional[Union[str, Path]] = None,
        num_sequences: int = 1000,
        seq_length: int = 200,
        **kwargs,
    ) -> Any:
        """
        Main entry point for the unified pipeline.

        Args:
            model1_path: Path to first model/profile
            model2_path: Path to second model/profile
            model1_type: Type of first model ('pwm', 'bamm', 'sitega')
            model2_type: Type of second model ('pwm', 'bamm', 'sitega')
            comparison_type: Type of comparison ('profile', 'motif', 'motali', 'tomtom-like')
            seq_source1: Path to first sequence file (for sequence-based)
            seq_source2: Path to second sequence file (for sequence-based)
            num_sequences: Number of sequences to generate if needed
            seq_length: Length of sequences to generate if needed
            **kwargs: Additional arguments for comparison

        Returns:
            Comparison results
        """
        self.logger.info(f"Starting pipeline with comparison_type='{comparison_type}'")

        if comparison_type.lower() == "profile":
            # Score-based comparison path
            self.logger.info("Executing score-based comparison")
            return self.execute_score_comparison(model1_path, model2_path, **kwargs)

        elif comparison_type.lower() in ["motif", "motali"]:
            # Sequence-based comparison path
            self.logger.info("Executing scan-based comparison")

            # Load models
            self.logger.info(
                f"Loading models from {model1_path} (type: {model1_type}) and {model2_path} (type: {model2_type})"
            )
            model1 = MotifModel.create_from_file(str(model1_path), model1_type)
            model2 = MotifModel.create_from_file(str(model2_path), model2_type)

            if model1 is None or model2 is None:
                raise ValueError("Failed to load one or both models")

            # Load or generate sequences and promoters
            self.logger.info(f"Loading sequences from source: {seq_source1}")
            sequences = self.load_sequences(seq_source1, num_sequences, seq_length)
            self.logger.info(f"Loading promoters from source: {seq_source2}")
            promoters = self.load_sequences(seq_source2, num_sequences, seq_length)

            # Execute appropriate comparison
            self.logger.info(f"Running {comparison_type} comparison")
            result = self.execute_motif_comparison(
                model1, model2, sequences=sequences, promoters=promoters, comparison_type=comparison_type, **kwargs
            )

            self.logger.info("Pipeline completed successfully")
            return result

        elif comparison_type.lower() == "tomtom-like":
            # TomTom-like comparison path
            self.logger.info("Executing TomTom-like comparison")

            # Load models
            self.logger.info(
                f"Loading models from {model1_path} (type: {model1_type}) and {model2_path} (type: {model2_type})"
            )
            model1 = MotifModel.create_from_file(str(model1_path), model1_type)
            model2 = MotifModel.create_from_file(str(model2_path), model2_type)

            if model1 is None or model2 is None:
                raise ValueError("Failed to load one or both models")

            if model1.model_type != model2.model_type:
                # Load or generate sequences for potential model conversion
                self.logger.info("WARNING! models have different origin, switch to `pfm_mode`")
                self.logger.info("Generation sequences for model conversion")
                sequences = self.load_sequences(None, num_sequences, seq_length)
                kwargs["pfm_mode"] = True
            elif kwargs.get("pfm_mode"):
                # Load or generate sequences for potential model conversion
                self.logger.info("Generation sequences for model conversion")
                sequences = self.load_sequences(None, num_sequences, seq_length)
            else:
                sequences = None

            # Execute TomTom comparison
            self.logger.info(f"Running {comparison_type} comparison")
            result = self.execute_tomtom_comparison(model1, model2, sequences, **kwargs)
            self.logger.info("Pipeline completed successfully")
            return result

        else:
            raise ValueError(
                f"Unknown comparison type: {comparison_type}. Expected 'profile', 'motif', 'motali', or 'tomtom-like'."
            )


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
    **kwargs,
) -> Any:
    """
    Module-level function to run the pipeline.

    Args:
        model1_path: Path to first model/profile
        model2_path: Path to second model/profile
        model1_type: Type of first model ('pwm', 'bamm', 'sitega')
        model2_type: Type of second model ('pwm', 'bamm', 'sitega')
        comparison_type: Type of comparison ('profile', 'motif', 'motali', 'tomtom-like')
        seq_source1: Path to first sequence file (for sequence-based)
        seq_source2: Path to second sequence file (for sequence-based)
        num_sequences: Number of sequences to generate if needed
        seq_length: Length of sequences to generate if needed
        **kwargs: Additional arguments for comparison

    Returns:
        Comparison results
    """
    pipeline = Pipeline()
    return pipeline.run_pipeline(
        model1_path=model1_path,
        model2_path=model2_path,
        model1_type=model1_type,
        model2_type=model2_type,
        comparison_type=comparison_type,
        seq_source1=seq_source1,
        seq_source2=seq_source2,
        num_sequences=num_sequences,
        seq_length=seq_length,
        **kwargs,
    )
