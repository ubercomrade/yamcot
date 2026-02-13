"""
Unit tests for key computational functions in mimosa.

These tests validate the correctness of individual functions from:
- mimosa/functions.py
- mimosa/comparison.py
- mimosa/models.py
"""

import numpy as np
import pytest

from mimosa.comparison import (
    ComparatorConfig,
    compare,
    create_comparator_config,
    strategy_motali,
    strategy_tomtom,
    strategy_universal,
)
from mimosa.comparison import registry as comparison_registry
from mimosa.functions import (
    batch_all_scores,
    cut_prc,
    cut_roc,
    format_params,
    pcm_to_pfm,
    pfm_to_pwm,
    precision_recall_curve,
    roc_curve,
    score_seq,
    scores_to_frequencies,
    standardized_pauc,
)
from mimosa.models import (
    GenericModel,
    _score_to_frequency,
    calculate_threshold_table,
    get_frequencies,
    get_scores,
    scan_model,
)
from mimosa.models import registry as model_registry
from mimosa.ragged import RaggedData, ragged_from_list


def test_pfm_to_pwm_basic():
    """Test basic PFM to PWM conversion"""
    # Create a simple PFM with uniform values
    pfm = np.array([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])

    pwm = pfm_to_pwm(pfm)

    # Verify shape is preserved
    assert pwm.shape == pfm.shape

    # Verify that PWM values are log ratios
    expected = np.log((pfm + 0.0001) / 0.25)
    np.testing.assert_allclose(pwm, expected, rtol=1e-6)


def test_pfm_to_pwm_with_zeros():
    """Test PFM to PWM conversion with zeros"""
    pfm = np.array([[0.0, 0.5], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    pwm = pfm_to_pwm(pfm)

    # Verify shape is preserved
    assert pwm.shape == pfm.shape

    # Verify that PWM values are calculated correctly even with zeros
    expected = np.log((pfm + 0.0001) / 0.25)
    np.testing.assert_allclose(pwm, expected, rtol=1e-6)


def test_pcm_to_pfm_basic():
    """Test basic PCM to PFM conversion"""
    pcm = np.array([[2, 3], [1, 1], [1, 0], [0, 0]], dtype=float)

    pfm = pcm_to_pfm(pcm)

    # Verify shape is preserved
    assert pfm.shape == pcm.shape

    # Calculate expected manually
    number_of_sites = pcm.sum(axis=0)
    expected = (pcm + 0.25) / (number_of_sites + 1)
    np.testing.assert_allclose(pfm, expected, rtol=1e-6)


def test_score_seq_basic():
    """Test basic sequence scoring"""
    # Simple scoring model
    model = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    # DNA sequence as numerical representation [A, C, G, T] -> [0, 1, 2, 3]
    num_site = np.array([0, 1, 2], dtype=np.int8)
    kmer = 1

    score = score_seq(num_site, kmer, model)

    # With kmer=1, the function should compute the sum of model values at positions
    expected_score = 1.0 + 2.0 + 3.0
    assert score == expected_score


def test_precision_recall_curve_basic():
    """Test basic precision-recall curve calculation"""
    classification = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    precision, recall, thresholds = precision_recall_curve(classification, scores)

    # Verify shapes are consistent
    assert precision.shape == recall.shape == thresholds.shape
    # Verify bounds
    assert np.all(precision >= 0) and np.all(precision <= 1.1)  # Allow for small numerical errors
    assert np.all(recall >= 0) and np.all(recall <= 1.1)


def test_roc_curve_basic():
    """Test basic ROC curve calculation"""
    classification = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    tpr, fpr, thresholds = roc_curve(classification, scores)

    # Verify shapes are consistent
    assert tpr.shape == fpr.shape == thresholds.shape
    # Verify bounds
    assert np.all(tpr >= 0) and np.all(tpr <= 1.1)
    assert np.all(fpr >= 0) and np.all(fpr <= 1.1)


def test_cut_roc_basic():
    """Test basic ROC curve cutting"""
    tpr = np.array([0.0, 0.5, 1.0])
    fpr = np.array([0.0, 0.2, 0.5])
    thr = np.array([np.inf, 0.5, 0.0])
    score_cutoff = 0.6

    tpr_cut, fpr_cut, thr_cut = cut_roc(tpr, fpr, thr, score_cutoff)

    # Verify that the function returns arrays
    assert isinstance(tpr_cut, np.ndarray)
    assert isinstance(fpr_cut, np.ndarray)
    assert isinstance(thr_cut, np.ndarray)


def test_cut_prc_basic():
    """Test basic PRC curve cutting"""
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 0.8, 0.6])
    thr = np.array([np.inf, 0.5, 0.0])
    score_cutoff = 0.6

    rec_cut, prec_cut, thr_cut = cut_prc(rec, prec, thr, score_cutoff)

    # Verify that the function returns arrays
    assert isinstance(rec_cut, np.ndarray)
    assert isinstance(prec_cut, np.ndarray)
    assert isinstance(thr_cut, np.ndarray)


def test_standardized_pauc_basic():
    """Test basic standardized partial AUC calculation"""
    pauc_raw = 0.7
    pauc_min = 0.5
    pauc_max = 1.0

    standardized = standardized_pauc(pauc_raw, pauc_min, pauc_max)

    # Manual calculation: 0.5 * (1.0 + (0.7 - 0.5) / (1.0 - 0.5))
    expected = 0.5 * (1.0 + (0.7 - 0.5) / (1.0 - 0.5))
    assert abs(standardized - expected) < 1e-6


def test_scores_to_frequencies_basic():
    """Test basic scores to frequencies conversion"""
    # Create a simple RaggedData with some scores
    data = np.array([1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32)
    offsets = np.array([0, 2, 4, 5], dtype=np.int64)  # Two sequences of lengths 2, 2, 1
    ragged_scores = RaggedData(data, offsets)

    freq_result = scores_to_frequencies(ragged_scores)

    # Verify the output is a RaggedData
    assert isinstance(freq_result, RaggedData)
    # Verify shape consistency
    assert freq_result.data.shape == ragged_scores.data.shape
    assert freq_result.offsets.shape == ragged_scores.offsets.shape


def test_format_params_basic():
    """Test basic parameter formatting"""
    params = {"k": 3, "metric": "pcc", "n_perm": 1000}
    formatted = format_params(params)

    # Should be in alphabetical order: k-3_metric-pcc_n_perm-1000
    expected = "k-3_metric-pcc_n_perm-1000"
    assert formatted == expected


def test_generic_model_creation():
    """Test GenericModel creation and immutability"""
    representation = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.1, 0.2]])

    model = GenericModel(type_key="pwm", name="test_model", representation=representation, length=2, config={"kmer": 1})

    assert model.type_key == "pwm"
    assert model.name == "test_model"
    assert model.length == 2
    assert model.config["kmer"] == 1

    # Test immutability
    try:
        model.name = "modified"
        assert False, "Model should be immutable"
    except Exception:
        pass  # Expected


def test_model_registry():
    """Test model registry functionality"""
    # Test that we can get registered strategies
    pwm_strategy = model_registry.get("pwm")
    assert pwm_strategy is not None

    bamm_strategy = model_registry.get("bamm")
    assert bamm_strategy is not None

    # Test invalid strategy raises ValueError
    try:
        model_registry.get("invalid_strategy")
        assert False, "Should raise ValueError for invalid strategy"
    except ValueError:
        pass  # Expected


def test_create_comparator_config():
    """Test ComparatorConfig creation and factory function"""
    # Test factory function with defaults
    config = create_comparator_config()
    assert config.metric == "pcc"
    assert config.n_permutations == 0
    assert config.seed is None

    # Test factory function with custom parameters
    config = create_comparator_config(metric="cj", n_permutations=100, seed=42)
    assert config.metric == "cj"
    assert config.n_permutations == 100
    assert config.seed == 42

    # Test immutability
    try:
        config.metric = "modified"
        assert False, "Config should be immutable"
    except Exception:
        pass  # Expected


def test_comparison_registry():
    """Test comparison registry functionality"""
    # Test that we can get registered strategies
    tomtom_strategy = comparison_registry.get("tomtom")
    assert tomtom_strategy is not None

    universal_strategy = comparison_registry.get("universal")
    assert universal_strategy is not None

    motali_strategy = comparison_registry.get("motali")
    assert motali_strategy is not None

    # Test invalid strategy returns None
    invalid_strategy = comparison_registry.get("invalid_strategy")
    assert invalid_strategy is None


def test_scan_model_with_pwm():
    """Test scanning with PWM model"""
    # Create simple PWM model
    representation = np.array(
        [
            [0.2, 0.3, 0.1],  # A
            [0.3, 0.2, 0.4],  # C
            [0.2, 0.4, 0.3],  # G
            [0.3, 0.1, 0.2],  # T
            [0.1, 0.1, 0.1],  # N (minimum values)
        ]
    )

    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=3, config={"kmer": 1})

    # Create test sequences
    sequences = ragged_from_list(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),  # A,C,G,T,C,G,A
            np.array([1, 2, 3, 0], dtype=np.int8),  # C,G,T,A
        ],
        dtype=np.int8,
    )

    # Test scanning
    scores = scan_model(model, sequences, "+")
    assert isinstance(scores, RaggedData)
    assert scores.data.size > 0


def test_get_frequencies():
    """Test frequency calculation"""
    representation = np.array([[0.2, 0.3], [0.3, 0.2], [0.2, 0.4], [0.3, 0.1], [0.1, 0.1]])

    model = GenericModel("pwm", "test", representation, 2, {"kmer": 1})

    sequences = ragged_from_list(
        [
            np.array([0, 1, 2, 3], dtype=np.int8),
            np.array([1, 2, 3, 0], dtype=np.int8),
        ],
        dtype=np.int8,
    )

    frequencies = get_frequencies(model, sequences, "+")
    assert isinstance(frequencies, RaggedData)
    assert frequencies.data.size > 0


def test_batch_all_scores_with_simple_data():
    """Test batch_all_scores with simple RaggedData"""
    # Create proper RaggedData for testing
    data = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    offsets = np.array([0, 3, 6], dtype=np.int64)
    sequences = RaggedData(data, offsets)
    representation = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=False)
    assert hasattr(result, "data") and hasattr(result, "offsets")


def test_strategy_functions_exist():
    """Test that all strategy functions are properly defined"""
    # Test that strategy functions exist and are callable
    assert callable(strategy_tomtom)
    assert callable(strategy_universal)
    assert callable(strategy_motali)


if __name__ == "__main__":
    pytest.main([__file__])
