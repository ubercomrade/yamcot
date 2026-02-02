"""
models
======

This module defines abstractions for motif models.  A motif is
represented internally as a numeric array and exposes methods for
scoring sequences and computing the best match score across all
positions and strand orientations.  Concrete subclasses implement
different motif scoring models such as PWMs or Bayesian Markov
models (BaMMs).
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from .functions import batch_all_scores, batch_best_scores, pfm_to_pwm, scores_to_frequencies
from .io import parse_file_content, read_bamm, read_meme, read_pfm, read_sitega, write_sitega
from .ragged import RaggedData

StrandMode = Literal["best", "+", "-", "both"]

@dataclass(frozen=True)
class RaggedScores:
    """Container for ragged arrays of scores with variable-length sequences.
    
    Attributes
    ----------
    values : np.ndarray
        Float32 array of shape (n_seq, Lmax) containing the score values.
    lengths : np.ndarray
        Int32 array of shape (n_seq,) containing the length of each sequence.
    """
    values: np.ndarray      # float32, shape (n_seq, Lmax)
    lengths: np.ndarray     # int32, shape (n_seq,)

    @classmethod
    def from_numba(cls, rs_numba: RaggedData) -> RaggedScores:
        """Convert a RaggedData object from Numba to RaggedScores.
        
        Parameters
        ----------
        rs_numba : RaggedData
            Input RaggedData object from Numba computations.
            
        Returns
        -------
        RaggedScores
            Converted RaggedScores object.
        """
        n = rs_numba.num_sequences
        lengths = np.zeros(n, dtype=np.int32)
        for i in range(n):
            lengths[i] = rs_numba.get_length(i)
        
        l_max = int(lengths.max()) if n > 0 else 0
        values = np.zeros((n, l_max), dtype=np.float32)
        for i in range(n):
            row = rs_numba.get_slice(i)
            values[i, :len(row)] = row
        return cls(values, lengths)

@dataclass
class MotifModel:
    """
    Abstract base class for motif models.

    Subclasses must implement the :meth:`scan` method which returns
    per-position scores for the forward and reverse complement of a
    sequence. The :meth:`best_score` helper uses :meth:`scan` to
    compute the maximal score across all positions and strands.

    Attributes
    ----------
    matrix : np.ndarray
        Numeric representation of the motif.
    name : str
        Human readable identifier of the motif.
    length : int
        Length of the motif (number of positions).
    strand_mode : {"best", "+", "-"}
        Default strand mode for scores/frequencies properties.
    """

    _registry: ClassVar[Dict[str, type[MotifModel]]] = {}

    matrix: np.ndarray
    name: str
    length: int

    strand_mode: StrandMode = field(default="best", init=True)

    statistics: Optional[Dict[str, float]] = field(default=None, init=False)
    _threshold_table: Optional[np.ndarray] = field(default=None, init=False)
    _pfm: Optional[np.ndarray] = field(default=None, init=False)

    _freq_cache: Dict[StrandMode, RaggedData] = field(
        default_factory=dict,
        init=False,
    )
    @classmethod
    def register_subclass(cls, model_type: str, subclass: type[MotifModel]):
        """Register a subclass for a specific model type."""
        cls._registry[model_type.lower()] = subclass

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    @property
    def threshold_table(self) -> Optional[np.ndarray]:
        """
        Lazy property for threshold table.

        Returns
        -------
        Optional[np.ndarray]
            Precomputed threshold values for scoring.
        """
        if self._threshold_table is None:
            logger = logging.getLogger(__name__)
            logger.warning(f"There is no threshold table for model: {self.name}")
        return self._threshold_table


    def get_threshold_table(self, promoters) -> np.ndarray:
        """Compute the threshold table based on the matrix.

        Returns
        -------
        np.ndarray
            The computed threshold table.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Computing threshold table for model: {self.name}")

        if self._threshold_table is None:
            self._threshold_table = self._calculate_threshold_table(promoters)

        return self._threshold_table


    def _calculate_threshold_table(self, promoters: RaggedData) -> np.ndarray:
        """Calculate the threshold table using all sites in all sequences.
        
        Computes a lookup table mapping motif scores to false positive rates (FPR)
        by evaluating the motif against all possible positions in the promoter sequences.
        
        Parameters
        ----------
        promoters : RaggedData
            Collection of promoter sequences to calculate thresholds against.
        
        Returns
        -------
        np.ndarray, shape (N, 2)
            Array where each row contains [score, -log10(fpr)], sorted by descending score.
        """
        # Get scores for all positions
        ragged_scores = self.get_scores(promoters, strand=self.strand_mode)

        # Flatten - collect all scores in one array directly from data (since there's no padding)
        flat_scores = ragged_scores.data

        if flat_scores.size == 0:
            # Fallback: table with one "bad" value
            return np.array([[0.0, 0.0]], dtype=np.float64)

        # Sort in descending order
        scores_sorted = np.sort(flat_scores)[::-1]
        n_total = flat_scores.size

        # Calculate FPR (-log10)
        # FPR = rank / N_total
        # Use unique to ensure identical scores have the same FPR (worst case)
        unique_scores, inverse, counts = np.unique(scores_sorted, return_inverse=True, return_counts=True)

        # np.unique sorts in ascending order. We need descending.
        unique_scores = unique_scores[::-1]  # Score desc
        counts = counts[::-1]

        # Cumulative sum (how many sites have score >= current)
        cum_counts = np.cumsum(counts)

        # FPR = cum_counts / n_total
        fpr_values = cum_counts / n_total

        # Log transform: -log10(FPR)
        # Avoid log(0) (though here fpr > 0 always, since n >= 1)
        log_fpr_values = -np.log10(fpr_values)

        # Table: [Score, -log10(FPR)]
        table = np.column_stack([unique_scores, log_fpr_values])

        return table.astype(np.float64)

    def _score_to_frequency(self, score: float) -> float:
        """Return -log10(FPR) for a given score using the threshold table.
        
        Parameters
        ----------
        score : float
            Motif score to convert to frequency.
        
        Returns
        -------
        float
            The -log10(false positive rate) corresponding to the score.
        """
        if self._threshold_table is None:
            return np.nan

        scores_col = self._threshold_table[:, 0]
        logfpr_col = self._threshold_table[:, 1]  # <-- this is -log10(FPR)

        if score >= scores_col[0]:
            return float(logfpr_col[0])
        if score <= scores_col[-1]:
            return float(logfpr_col[-1])

        idx = np.searchsorted(-scores_col, -score, side="left")
        if idx >= len(logfpr_col):
            return float(logfpr_col[-1])
        return float(logfpr_col[idx])

    def _frequency_to_score(self, frequency: float, background_data: Optional[RaggedData] = None) -> float:
        """Convert frequency (FPR) to threshold score.
        
        If background_data is provided, calculation is done across ALL positions (sites)
        in these sequences using scores_to_frequencies.
        
        Parameters
        ----------
        frequency : float
            False positive rate to convert to a score threshold.
        background_data : RaggedData, optional
            Background sequences to calculate threshold from, if not using precomputed table.
        
        Returns
        -------
        float
            Score threshold corresponding to the given frequency.
        """
        if background_data is not None:
            # 1. Get scores for ALL positions in background data
            all_sites_scores = self.scan(background_data, strand="best")
            
            if all_sites_scores.data.size == 0:
                return 0.0

            # 2. Use scores_to_frequencies to get -log10(FPR) for each position
            # Function internally performs np.unique and np.cumsum on all .data values
            freq_ragged = scores_to_frequencies(all_sites_scores)
            log_p_values = freq_ragged.data
            
            # 3. Determine target -log10(FPR) value
            target_log_p = -np.log10(frequency) if frequency > 0 else 100.0
            
            # 4. Find minimum score among those whose frequency <= target (i.e. -log_p >= target)
            # This precisely corresponds to the _threshold_table construction logic
            mask = log_p_values >= target_log_p
            if not np.any(mask):
                return float(all_sites_scores.data.min())
                
            return float(all_sites_scores.data[mask].min())

        # Fallback via precomputed table
        if self._threshold_table is None:
            raise ValueError("Threshold table not computed. Call get_threshold_table() or provide background_data.")
        
        if frequency <= 0:
            return float(self._threshold_table[0, 0])  # infinitely strict threshold -> maximum score

        target_logfpr = -np.log10(frequency)

        scores_col = self._threshold_table[:, 0]
        logfpr_col = self._threshold_table[:, 1]  # -log10(FPR)

        mask = logfpr_col >= target_logfpr
        if not np.any(mask):
            # even the weakest score doesn't give such FPR
            return float(scores_col[-1])

        last_valid = np.where(mask)[0][-1]
        return float(scores_col[last_valid])


    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def scan(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """
        Perform batch scanning of sequences for motif matches.

        Parameters
        ----------
        sequences : RaggedData
            Encoded sequences (int8).
        strand : {"+", "-", "best"}, optional
            Strand selection mode:
            - None: use self.strand_mode.
            - "+": forward strand only.
            - "-": reverse complement only.
            - "best": maximum score between strands at each position.

        Returns
        -------
        RaggedData
            Scan results.
        """
        raise NotImplementedError("MotifModel subclasses must implement scan()")


    @classmethod
    def from_file(cls, path: str, **kwargs) -> 'MotifModel':
        """
        Abstract class method to create a motif model from a file.
        
        Parameters
        ----------
        path : str
            Path to the motif file.
        **kwargs : dict
            Additional arguments for specific model types.
            
        Returns
        -------
        MotifModel
            A motif model instance.
        """
        raise NotImplementedError("Subclasses must implement from_file()")

    @classmethod
    def create_from_file(cls, path: str, model_type: str, **kwargs) -> 'MotifModel':
        """
        Factory method to create a motif model from a file based on model_type.
        
        Parameters
        ----------
        path : str
            Path to the motif file.
        model_type : str
            Type of the model ('pwm', 'bamm', 'sitega').
        **kwargs : dict
            Additional arguments for specific model types.
            
        Returns
        -------
        MotifModel
            A motif model instance.
        """
        model_type = model_type.lower()
        if model_type not in cls._registry:
            raise ValueError(f"Unsupported model type: {model_type}. "
                             f"Registered types: {list(cls._registry.keys())}")
        
        subclass = cls._registry[model_type]
        return subclass.from_file(path, **kwargs)


    def get_sites(
        self,
        sequences: RaggedData,
        mode: str = "best",
        fpr_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Find motif binding sites in sequences.
        
        The method operates in two modes:
        - "best": finds the single best site in each sequence
        - "threshold": finds all sites with false positive rate ≤ fpr_threshold
        
        Parameters
        ----------
        sequences : RaggedData
            Encoded sequences to search for motif sites.
        mode : {"best", "threshold"}, optional
            Site finding mode (default "best").
        fpr_threshold : float, optional
            False positive rate threshold for "threshold" mode (e.g., 0.001).
            Required for mode="threshold".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - seq_index: sequence index
            - start: start position of site (0-based)
            - end: end position of site (exclusive)
            - strand: DNA strand ("+" or "-")
            - score: recognition function value
            - frequency: false positive rate (FPR) from threshold_table
            - site: site as string representation (ACGT)
        """
        # Validate parameters
        if mode not in ["best", "threshold"]:
            raise ValueError(f"mode must be 'best' or 'threshold', got {mode!r}")
        if mode == "threshold" and fpr_threshold is None:
            raise ValueError("fpr_threshold is required for mode='threshold'")

        # Determine threshold score
        score_threshold = (
            self._frequency_to_score(fpr_threshold)
            if mode == "threshold" and fpr_threshold is not None
            else None
        )
        if score_threshold is not None:
            logger = logging.getLogger(__name__)
            logger.info(f"FPR threshold: {fpr_threshold} → Score threshold: {score_threshold:.4f}")

        # Helper function to add a single site
        def add_site(seq_idx: int, seq: np.ndarray, pos: int, strand_idx: int, score: float):
            """Add a site to results."""
            if pos + self.length > len(seq):
                return

            site_seq = seq[pos : pos + self.length]
            strand = "+" if strand_idx == 0 else "-"

            # Reverse complement for minus strand
            if strand_idx == 1:
                site_seq = self._get_rc_sequence(site_seq)

            results.append({
                'seq_index': seq_idx,
                'start': int(pos),
                'end': int(pos + self.length),
                'strand': strand,
                'score': score,
                'frequency': self._score_to_frequency(score),
                'site': self._int_to_seq(site_seq)
            })

        # Collect results
        results = []

        # Batch scanning
        s_fwd_ragged = self.scan(sequences, strand="+")
        s_rev_ragged = self.scan(sequences, strand="-")
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

            else:  # mode == "threshold"
                # Forward strand
                f_pos = np.where(s_fwd >= score_threshold)[0]
                for pos in f_pos:
                    add_site(seq_idx, seq, int(pos), 0, float(s_fwd[pos]))

                # Reverse strand
                r_pos = np.where(s_rev >= score_threshold)[0]
                for pos in r_pos:
                    add_site(seq_idx, seq, int(pos), 1, float(s_rev[pos]))

        # Create and sort DataFrame
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values(['seq_index', 'score'], ascending=[True, False]).reset_index(drop=True)

        logger = logging.getLogger(__name__)
        logger.info(f"Found {len(df)} site(s) in {sequences.num_sequences} sequence(s)")
        return df

    @staticmethod
    def _int_to_seq(seq_int: np.ndarray) -> str:
        """Convert integer-encoded sequence to ACGT string.
        
        Parameters
        ----------
        seq_int : np.ndarray
            Integer-encoded sequence (0=A, 1=C, 2=G, 3=T, 4=N).

        Returns
        -------
        str
            Sequence as string.
        """
        decoder = np.array(['A', 'C', 'G', 'T', 'N'], dtype='U1')
        safe_seq = np.clip(seq_int, 0, 4)
        return ''.join(decoder[safe_seq])

    @staticmethod
    def _get_rc_sequence(seq_int: np.ndarray) -> np.ndarray:
        """Return reverse complement of sequence.
        
        Parameters
        ----------
        seq_int : np.ndarray
            Integer-encoded sequence.

        Returns
        -------
        np.ndarray
            Reverse complement of sequence.
        """
        RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
        return RC_TABLE[seq_int[::-1]]

    def write_pfm(self, path: str) -> None:
        """Write the motif to a PFM formatted file.

        Parameters
        ----------
        path : str
            Path of the output file.
        """
        if self.pfm is not None:
            with open(path, "w") as fname:
                header = f">{self.name}"
                np.savetxt(fname,
                            self.pfm[:4, :].T,
                            fmt='%.8f',
                            delimiter='\t',
                            newline='\n',
                            header=header,
                            footer='',
                            comments='',
                            encoding=None)

    def write_dist(self, path: str) -> None:
        """Write the threshold table of motif to a DIST formatted file.

        Parameters
        ----------
        path : str
            Path of the output file.
        """
        table = self.threshold_table
        if table is None:
            logger = logging.getLogger(__name__)
            logger.error(f"Cannot write DIST file: threshold table not computed for {self.name}")
            return

        table = copy.deepcopy(table)
        max_score = self.matrix.max(axis=0).sum()
        min_score = self.matrix.min(axis=0).sum()

        table[:, 0] = (table[:, 0] - min_score) / (max_score - min_score)
        with open(path, "w") as fname:
            np.savetxt(fname,
                          table,
                          fmt='%.18f',
                          delimiter='\t',
                          newline='\n',
                          footer='',
                          comments='',
                          encoding=None)

    @property
    def pfm(self) -> Optional[np.ndarray]:
        """
        Lazy property for position frequency matrix.

        Returns
        -------
        Optional[np.ndarray]
            Cached PFM if available, otherwise None.
            Use get_pfm() to compute and cache.
        """
        if self._pfm is None:
            logger = logging.getLogger(__name__)
            logger.warning(f"PFM not computed for model: {self.name}. Use get_pfm(sequences) to compute.")
        return self._pfm

    def get_pfm(
        self,
        sequences: RaggedData,
        mode: str = "best",
        fpr_threshold: Optional[float] = None,
        top_fraction: Optional[float] = None,
        pseudocount: float = 0.25,
        force_recompute: bool = False
    ) -> np.ndarray:
        """Construct Position Frequency Matrix (PFM) from binding sites.
        
        Result is cached in the _pfm attribute for reuse.

        Parameters
        ----------
        sequences : RaggedData
            Encoded sequences to extract binding sites from.
        mode : {"best", "threshold"}, optional
            Site finding mode (default "best").
        fpr_threshold : float, optional
            Frequency threshold for "threshold" mode.
        top_fraction : float, optional
            Selects only top N% of sites by score.
        pseudocount : float, optional
            Pseudocount for smoothing (default 0.25).
        force_recompute : bool, optional
            If True, recomputes PFM even if it's already cached.

        Returns
        -------
        np.ndarray
            Normalized PFM of shape (4, motif_length).
        """
        # Return cached version if available
        if self._pfm is not None and not force_recompute:
            logger = logging.getLogger(__name__)
            logger.info(f"Returning cached PFM for model: {self.name}")
            return self._pfm

        logger = logging.getLogger(__name__)
        logger.info(f"Computing PFM for model: {self.name}")

        # Get sites
        sites_df = self.get_sites(sequences, mode=mode, fpr_threshold=fpr_threshold)
        if len(sites_df) == 0:
            raise ValueError("No sites found")

        sites_df = sites_df.sort_values(by=['score'], axis=0, ascending=False)

        # Select top N% if specified
        if top_fraction is not None:
            n_keep = max(1, int(len(sites_df) * top_fraction))
            sites_df = sites_df.nlargest(n_keep, 'score')
            logger = logging.getLogger(__name__)
            logger.info(f"Selected top {top_fraction*100:.1f}%: {n_keep} sites")

        # Initialize PFM with pseudocounts
        pfm = np.full((4, self.length), pseudocount, dtype=np.float32)
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Fill counters
        for site_str in sites_df['site']:
            for pos, nuc in enumerate(site_str):
                if nuc in nuc_map:
                    pfm[nuc_map[nuc], pos] += 1.0

        # Normalize to probabilities
        pfm = pfm / pfm.sum(axis=0, keepdims=True)

        # Cache the result
        self._pfm = pfm

        return pfm

    # ------------------------------------------------------------------
    # Helper methods for strand modes
    # ------------------------------------------------------------------
    @staticmethod
    def _reduce_strand(seq_scores: np.ndarray, strand: StrandMode) -> np.ndarray:
        """Transform (2, N) → (N,) depending on strand mode.

        Parameters
        ----------
        seq_scores : np.ndarray
            Scores with shape (2, N_positions).
        strand : {"best", "+", "-"}

        Returns
        -------
        np.ndarray
            One-dimensional scores per position (N_positions,).
        """
        if strand == "best":
            # maximum across both strands for each position
            return np.max(seq_scores, axis=0)
        if strand == "+":
            return seq_scores[0]
        if strand == "-":
            return seq_scores[1]
        raise ValueError(f"Unknown strand={strand!r}. Use '+', '-', or 'best'.")

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    def get_scores(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """Calculate motif scores for each position in the sequences using batch processing.

        Parameters
        ----------
        sequences : RaggedData
            Encoded sequences (int8).
        strand : {"best", "+", "-", "both"}, optional
            Strand to score. Default is self.strand_mode.

        Returns
        -------
        RaggedData
            Ragged array of scores (float32).
        """
        return self.scan(sequences, strand=strand)

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------
    def get_frequencies(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """Calculate per-position hit frequencies (probability maps) using batch processing.

        Parameters
        ----------
        sequences : RaggedData
            Encoded nucleotide sequences.
        strand : {"best", "+", "-", "both"}, optional
            Strand to evaluate.

        Returns
        -------
        RaggedData
            Per-position hit frequencies.
        """

        return scores_to_frequencies(self.scan(sequences, strand))

    # ------------------------------------------------------------------
    # Best score
    # ------------------------------------------------------------------
    def best_scores(self, sequences: RaggedData, strand: StrandMode = "best") -> np.ndarray:
        """Return the maximum raw score using batch processing.
        
        Parameters
        ----------
        sequences : RaggedData
            Encoded sequences to score.
        strand : {"best", "+", "-"}, optional
            Strand mode for scoring (default "best").

        Returns
        -------
        np.ndarray
            Maximum scores for each sequence.
        """
        kmer = getattr(self, 'kmer', 1)
        matrix = self.matrix.astype(np.float32)

        is_revcomp = (strand == "-")
        both_strands = (strand == "best")

        return batch_best_scores(sequences, matrix, kmer=kmer, is_revcomp=is_revcomp, both_strands=both_strands)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear all cached scores, frequencies and normalization range."""
        self._freq_cache.clear()
        self._threshold_table = None
        if not getattr(self, '_pfm_is_required', False):
            self._pfm = None

    # ------------------------------------------------------------------
    # Serialization (Joblib)
    # ------------------------------------------------------------------
    def save(self, filepath: str, clear_cache: bool = True) -> None:
        """Save the motif model to a file using joblib.

        Parameters
        ----------
        filepath : str
            Full path to the output file (e.g., 'motifs/M1.pkl').
        clear_cache : bool
            If True, clears caches before saving to minimize file size.
        """
        if clear_cache:
            self.clear_cache()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> MotifModel:
        """Load a motif model from a .pkl file.

        Parameters
        ----------
        filepath : str
            Path to the .pkl file.

        Returns
        -------
        MotifModel
            Loaded motif model instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Motif file not found: {filepath}")

        model = joblib.load(filepath)
        return model

    # ------------------------------------------------------------------
    # Polymorphism support
    # ------------------------------------------------------------------
    @property
    def model_type(self) -> str:
        """Abstract property to return the type of model."""
        raise NotImplementedError("Subclasses must implement model_type property")
    
    def write(self, path: str) -> None:
        """Abstract method to write the motif to a file in its native format."""
        raise NotImplementedError("Subclasses must implement write() method")


class PwmMotif(MotifModel):
    """Position Weight Matrix motif model.

    This class wraps a PWM (log odds scores) and provides efficient
    scoring using the precompiled :func:`yamcot.functions.batch_all_scores`
    Numba kernel. The input matrix is expected to have shape (5, L)
    where the final row contains column minima. The scoring logic
    treats any encoded nucleotide equal to 4 as an ambiguous 'N' and
    assigns a zero contribution.

    Parameters
    ----------
    matrix : np.ndarray
        PWM matrix with shape (5, L) where the 5th row contains column minima.
    name : str
        Name of the motif.
    length : int
        Length of the motif.
    pfm : np.ndarray
        Position Frequency Matrix of shape (4, L). Required attribute for PWM.
    kmer : int, optional
        Size of k-mer for scanning (default is 1).
    """

    def __init__(
        self,
        matrix: np.ndarray,
        name: str,
        length: int,
        pfm: np.ndarray,
        kmer: int = 1,
    ) -> None:
        super().__init__(
            matrix=matrix,
            name=name,
            length=length,
        )
        self.kmer = kmer
        # Assign to private attribute so property works correctly
        self._pfm = pfm
        self._pfm_is_required = True

    def scan(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """Score sequences with the PWM.

        Parameters
        ----------
        sequences : RaggedData
            Encoded nucleotide sequences (int8).
        strand : {"best", "+", "-", "both"}, optional
            Strand selection mode. If None, uses self.strand_mode.

        Returns
        -------
        RaggedData
            Scanning results with scores for each position.
        """
        strand = strand or self.strand_mode
        matrix = self.matrix.astype(np.float32)

        if strand == "+":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False)
        elif strand == "-":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False)
            sr = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"Unknown strand mode: {strand}")
            sys.exit(1)


    @classmethod
    def from_file(cls, path: str, index: int = 0, **kwargs) -> PwmMotif:
        """Create a PwmMotif from a file.
        
        Supports MEME, ProSampler, and PFM formats.
        
        Parameters
        ----------
        path : str
            Path to the motif file.
        index : int, optional
            Index of motif to read from file if multiple motifs are present (default 0).
        **kwargs : dict
            Additional arguments.
            
        Returns
        -------
        PwmMotif
            A PwmMotif object created from the file.
        """
        # Determine file format based on extension
        _, ext = os.path.splitext(path.lower())
        
        if ext == '.pkl':
            return joblib.load(path)
        elif ext == '.meme':
            # MEME format
            matrix, info, _number_of_motifs = read_meme(path, index=index)
            pfm = matrix
            name, length = info
        elif ext == '.pfm':
            # PFM format
            pwm, length, _minimum, _maximum = read_pfm(path)
            # Extract PFM from PWM (first 4 rows, excluding the 5th row of minimums)
            pfm = pwm[:4, :]  # Get the original PFM from the extended PWM
            name = os.path.splitext(os.path.basename(path))[0]
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"Wrong format pf PWM model: {path}")
            sys.exit(1)
        
        # Convert PFM to PWM
        pwm = pfm_to_pwm(pfm)
        # Add the 5th row for 'N' characters (minimum values at each position)
        pwm_ext = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0)
        return cls(
            matrix=pwm_ext,
            name=name,
            length=int(length),
            pfm=pfm
        )

    @property
    def model_type(self) -> str:
        """Return the type of model ('pwm')."""
        return 'pwm'

    def write(self, path: str) -> None:
        """Write the PWM motif to a file in PFM format."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.write_pfm(path)



class BammMotif(MotifModel):
    """Bayesian Markov Model motif.

    BaMMs extend PWMs by modelling dependencies between neighbouring
    nucleotides.  The representation and scanning procedure differ
    substantially from PWMs and require specialised software.  This
    class provides an interface compatible with the :class:`MotifModel`
    API but does not implement the scoring logic.  Users must supply
    an external scoring function via :meth:`scan`.

    Parameters
    ----------
    matrix : np.ndarray
        BaMM matrix representation.
    name : str
        Name of the motif.
    length : int
        Length of the motif.
    kmer : int, optional
        Size of k-mer for scanning (default is 2).
    """

    def __init__(
        self,
        matrix: np.ndarray,
        name: str,
        length: int,
        kmer: int = 2,
    ) -> None:
        super().__init__(
            matrix=matrix,
            name=name,
            length=length,
        )
        self.kmer = kmer

    def scan(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """Score sequences with the BaMM.

        Parameters
        ----------
        sequences : RaggedData
            Encoded nucleotide sequences (int8).
        strand : {"best", "+", "-"}, optional
            Strand selection mode. If None, uses self.strand_mode.

        Returns
        -------
        RaggedData
            Scanning results with scores for each position.
        """
        strand = strand or self.strand_mode

        matrix = self.matrix.astype(np.float32)

        if strand == "+":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False, is_bamm=True)
        elif strand == "-":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True, is_bamm=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False, is_bamm=True)
            sr = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True, is_bamm=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"Unknown strand mode: {strand}")
            sys.exit(1)



    @classmethod
    def from_file(cls, path: str, bg_path: str | None = None, order: int = 2, **kwargs) -> 'BammMotif':
        """Create a BammMotif from a file.
        
        Parameters
        ----------
        path : str
            Path to the BaMM motif file (.ihbcp format or base path).
        bg_path : str, optional
            Path to the background BaMM file (.hbcp format). If not provided,
            it attempts to find a background file in the same directory.
        order : int, optional
            Order of the BaMM model (default is 2).
        **kwargs : dict
            Additional arguments.
            
        Returns
        -------
        BammMotif
            A BammMotif object created from the file.
        """
        # Handle case where path is provided without extension (as in pipeline.py)
        if not path.endswith('.ihbcp') and not os.path.exists(path):
            ihbcp_path = f"{path}.ihbcp"
            hbcp_path = f"{path}.hbcp"
            if os.path.exists(ihbcp_path):
                path = ihbcp_path
                if bg_path is None and os.path.exists(hbcp_path):
                    bg_path = hbcp_path

        # If no background path is provided, try to find it
        if bg_path is None:
            # Look for background file in the same directory
            dir_path = os.path.dirname(path)
            basename = os.path.basename(path)
            # Try to find background file by replacing extension or using common names
            possible_bg_names = ['bamm.hbcp', 'background.hbcp', basename.replace('.ihbcp', '.hbcp')]
            for bg_name in possible_bg_names:
                possible_bg_path = os.path.join(dir_path, bg_name)
                if os.path.exists(possible_bg_path):
                    bg_path = possible_bg_path
                    break
        
        if bg_path is None:
            raise ValueError(f"Background file not found for {path}. Please provide bg_path parameter.")
        
        # If order is not provided, try to determine it from the file (like in pipeline.py)
        if 'order' not in kwargs and order == 2:
             _, max_order, num_positions = parse_file_content(path)
             order = max_order
             length = num_positions
        else:
             matrix_tmp = read_bamm(path, bg_path, order)
             length = matrix_tmp.shape[-1]

        # Read the BaMM motif and background files
        matrix = read_bamm(path, bg_path, order)
        name = os.path.splitext(os.path.basename(path))[0]
        return cls(
            matrix=matrix,
            name=name,
            length=length,
            kmer=order + 1  # BaMM kmer is typically order + 1
        )

    @property
    def model_type(self) -> str:
        """Return the type of model ('bamm')."""
        return 'bamm'

    def write(self, path: str) -> None:
        """Write the BaMM motif to a file."""
        # BaMM writing functionality would go here
        # For now, we'll implement a basic version that saves as joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.save(path)



class SitegaMotif(MotifModel):
    """SiteGA motif model.

    SiteGA motifs are based on a linear probabilistic grammar and
    represent dinucleotide dependencies across multiple positions.  The
    scoring logic can be derived from the original SiteGA
    implementation but is non‑trivial.  This class acts as a
    placeholder for future SiteGA integration.  A custom scanning
    function can be provided via :meth:`scan`.

    Parameters
    ----------
    matrix : np.ndarray
        SiteGA matrix representation.
    name : str
        Name of the motif.
    length : int
        Length of the motif.
    kmer : int, optional
        Size of k-mer for scanning (default is 2).
    """

    def __init__(
        self,
        matrix: np.ndarray,
        name: str,
        length: int,
        minimum: float = 0.0,
        maximum: float = 0.0,
        kmer: int = 2,
    ) -> None:
        super().__init__(
            matrix=matrix,
            name=name,
            length=length,
        )
        self.minimum = minimum
        self.maximum = maximum
        self.kmer = kmer

    def scan(self, sequences: RaggedData, strand: Optional[StrandMode] = None) -> RaggedData:
        """Score sequences with SiteGA model.

        Parameters
        ----------
        sequences : RaggedData
            Encoded nucleotide sequences (int8).
        strand : {"best", "+", "-"}, optional
            Strand selection mode. If None, uses self.strand_mode.

        Returns
        -------
        RaggedData
            Scanning results with scores for each position.
        """
        strand = strand or self.strand_mode
        matrix = self.matrix.astype(np.float32)

        if strand == "+":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False)
        elif strand == "-":
            return batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True)
        elif strand == "best":
            sf = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=False)
            sr = batch_all_scores(sequences, matrix, kmer=self.kmer, is_revcomp=True)
            return RaggedData(np.maximum(sf.data, sr.data), sf.offsets)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"Unknown strand mode: {strand}")
            sys.exit(1)


    @classmethod
    def from_file(cls, path: str, **kwargs) -> SitegaMotif:
        """Parse SiteGA output file to create a SitegaMotif object.
        
        Parameters
        ----------
        path : str
            Path to the SiteGA output file (typically ends with '.mat').
        **kwargs : dict
            Additional arguments.
            
        Returns
        -------
        SitegaMotif
            A SitegaMotif object created from the parsed data.
        """
        # Parse the SiteGA output file
        matrix, length, minimum, maximum = read_sitega(path)
        
        # Extract motif name from the file path
        name = os.path.splitext(os.path.basename(path))[0]
        
        # Create and return the SitegaMotif instance
        return cls(
            matrix=matrix,
            name=name,
            length=length,
            minimum=minimum,
            maximum=maximum,
            kmer=2  # SiteGA typically uses dinucleotide dependencies
        )

    @property
    def model_type(self) -> str:
        """Return the type of model ('sitega')."""
        return 'sitega'

    def write(self, path: str) -> None:
        """Write the SiteGA motif to a file in .mat format."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_sitega(self, path)


# Register subclasses for polymorphic loading
MotifModel.register_subclass('pwm', PwmMotif)
MotifModel.register_subclass('bamm', BammMotif)
MotifModel.register_subclass('sitega', SitegaMotif)
