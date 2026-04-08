"""
Unit tests for key computational functions in mimosa.

These tests validate the correctness of individual functions from:
- mimosa/functions.py
- mimosa/comparison.py
- mimosa/models.py
"""

from pathlib import Path

import numpy as np
import pytest

from mimosa.api import compare_motifs, create_config, run_comparison
from mimosa.cache import clear_cache
from mimosa.comparison import (
    create_comparator_config,
    strategy_motali,
    strategy_motif,
    strategy_profile,
)
from mimosa.comparison import registry as comparison_registry
from mimosa.functions import (
    batch_all_scores,
    cut_prc,
    cut_roc,
    fast_profile_score,
    format_params,
    pcm_to_pfm,
    pfm_to_pwm,
    precision_recall_curve,
    roc_curve,
    score_seq,
    scores_to_frequencies,
    standardized_pauc,
)
from mimosa.io import read_scores
from mimosa.models import (
    GenericModel,
    calculate_threshold_table,
    get_frequencies,
    read_model,
    scan_model,
    scores_to_log_fpr,
)
from mimosa.models import registry as model_registry
from mimosa.ragged import RaggedData, ragged_from_list

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "models"
_DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode an ACGT string as the project's integer alphabet."""
    return np.array([_DNA_TO_INT[symbol] for symbol in sequence], dtype=np.int8)


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


def test_read_scores_basic(tmp_path):
    """Numerical score profiles should be parsed from FASTA-like input."""
    path = tmp_path / "scores.fasta"
    path.write_text(">seq1\n0.1 0.2 0.3\n>seq2\n1.0 2.0\n", encoding="utf-8")

    result = read_scores(path)

    assert result.num_sequences == 2
    np.testing.assert_allclose(result.get_slice(0), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(result.get_slice(1), np.array([1.0, 2.0], dtype=np.float32))


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


def test_model_registry():
    """Test model registry functionality"""
    # Test that we can get registered strategies
    pwm_strategy = model_registry.get("pwm")
    assert pwm_strategy is not None

    bamm_strategy = model_registry.get("bamm")
    assert bamm_strategy is not None

    dimont_strategy = model_registry.get("dimont")
    assert dimont_strategy is not None

    slim_strategy = model_registry.get("slim")
    assert slim_strategy is not None


def test_read_model_supports_dimont_xml_and_matches_example_score():
    """Dimont XML models should load and reproduce exact strand-aware site scores."""
    model = read_model(str(FIXTURES_ROOT / "dimont" / "exampleD-model-1.xml"), "dimont")
    plus_sequence = ragged_from_list([_encode_sequence("TTCCAGGGAACCC")], dtype=np.int8)
    minus_sequence = ragged_from_list([_encode_sequence("GGGTTCCCTGGAA")], dtype=np.int8)

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 13
    assert model.config["kmer"] == 1
    assert model.representation.shape == (5, 13)
    assert plus_scores.get_slice(0)[0] == pytest.approx(-11.370793931934642)
    assert minus_scores.get_slice(0)[0] == pytest.approx(-11.370793931934642)


def test_read_model_supports_higher_order_dimont_xml():
    """Higher-order Dimont XML models should scan via the shared context kernel."""
    model = read_model(str(FIXTURES_ROOT / "dimont" / "stat_dimont-model-1.xml"), "dimont")
    plus_sequence = ragged_from_list([_encode_sequence("AACCC")], dtype=np.int8)
    minus_sequence = ragged_from_list([_encode_sequence("GGGTT")], dtype=np.int8)

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 5
    assert model.config["kmer"] == 4
    assert model.representation.shape == (5, 5, 5, 5, 5)
    assert plus_scores.get_slice(0)[0] == pytest.approx(-7.622432197747987)
    assert minus_scores.get_slice(0)[0] == pytest.approx(-7.622432197747987)


def test_read_model_supports_slim_xml_and_matches_example_score():
    """Slim XML models should reproduce the exact published site scores."""
    model = read_model(str(FIXTURES_ROOT / "slim" / "example-model-1.xml"), "slim")
    plus_sequence = ragged_from_list([_encode_sequence("TTCCTCGGAACTGAG")], dtype=np.int8)
    minus_sequence = ragged_from_list([_encode_sequence("CTCAGTTCCGAGGAA")], dtype=np.int8)

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "slim"
    assert model.length == 15
    assert model.config["kmer"] == 6
    assert model.representation.shape == (5, 5, 5, 5, 5, 5, 15)
    assert plus_scores.get_slice(0)[0] == pytest.approx(-14.550483500732756)
    assert minus_scores.get_slice(0)[0] == pytest.approx(-14.550483500732756)


def test_create_comparator_config():
    """Test ComparatorConfig creation and factory function"""
    # Test factory function with defaults
    config = create_comparator_config()
    assert config.metric == "pcc"
    assert config.n_permutations == 0
    assert config.seed is None

    # Test factory function with custom parameters
    config = create_comparator_config(metric="co", n_permutations=100, seed=42)
    assert config.metric == "co"
    assert config.n_permutations == 100
    assert config.seed == 42


def test_create_comparator_config_validates_kernel_range():
    """Kernel-size range should be valid for centered surrogate kernels."""
    with pytest.raises(ValueError, match="min_kernel_size"):
        create_comparator_config(min_kernel_size=7, max_kernel_size=5)

    with pytest.raises(ValueError, match="at least one odd value"):
        create_comparator_config(min_kernel_size=4, max_kernel_size=4)

    config = create_comparator_config(min_kernel_size=4, max_kernel_size=6)
    assert config.min_kernel_size == 4
    assert config.max_kernel_size == 6


def test_create_comparator_config_validates_min_logfpr():
    """Profile floor should reject negative logFPR thresholds."""
    with pytest.raises(ValueError, match="min_logfpr"):
        create_comparator_config(min_logfpr=-0.1)


def test_create_comparator_config_validates_cache_mode():
    """Profile cache mode should accept only explicit on/off values."""
    with pytest.raises(ValueError, match="cache_mode"):
        create_comparator_config(cache_mode="targets")

    config = create_comparator_config(cache_mode="on")
    assert config.cache_mode == "on"


def test_comparison_registry():
    """Test comparison registry functionality"""
    # Test that we can get registered strategies
    motif_strategy = comparison_registry.get("motif")
    assert motif_strategy is not None

    profile_strategy = comparison_registry.get("profile")
    assert profile_strategy is not None

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


def test_scores_to_log_fpr_uses_threshold_table_scale():
    """Threshold-table conversion should match the cached score-to-logFPR lookup."""
    representation = np.array(
        [
            [1.0, 0.2],
            [0.2, 1.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=2, config={"kmer": 1})
    promoters = ragged_from_list(
        [
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
            np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
        ],
        dtype=np.int8,
    )

    scores = scan_model(model, promoters, "+")
    threshold_table = calculate_threshold_table(model, promoters, strand="+")

    transformed = scores_to_log_fpr(model, scores, promoters, strand="+")

    idx = np.searchsorted(-threshold_table[:, 0], -scores.data, side="left")
    idx = np.clip(idx, 0, len(threshold_table) - 1)
    expected = threshold_table[:, 1][idx].astype(np.float32)

    np.testing.assert_allclose(transformed.data, expected)
    np.testing.assert_array_equal(transformed.offsets, scores.offsets)


def test_batch_all_scores_with_simple_data():
    """Test batch_all_scores with simple RaggedData"""
    # Create proper RaggedData for testing
    data = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    offsets = np.array([0, 3, 6], dtype=np.int64)
    sequences = RaggedData(data, offsets)
    representation = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=False)
    assert hasattr(result, "data") and hasattr(result, "offsets")


def _manual_reverse_scores(seq: np.ndarray, matrix: np.ndarray, kmer: int, with_context: bool) -> np.ndarray:
    """Reproduce reverse-complement scoring with the legacy per-window logic."""
    motif_len = matrix.shape[-1]
    context_len = kmer - 1
    window_size = motif_len + context_len
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    n_scores = max(0, seq.size - motif_len + 1)
    scores = np.zeros(n_scores, dtype=np.float32)

    for pos in range(n_scores):
        if with_context:
            window = np.full(window_size, 4, dtype=np.int8)
            for t in range(window_size):
                data_idx = pos + (window_size - 1 - t)
                if 0 <= data_idx < seq.size:
                    window[t] = rc_table[seq[data_idx]]
        else:
            window = rc_table[seq[pos : pos + motif_len][::-1]]
        scores[pos] = score_seq(window, kmer, matrix)

    return scores


def test_batch_all_scores_reverse_complement_preserves_positions():
    """Reverse-complement PWM scan should remain aligned to forward coordinates."""
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = ragged_from_list([seq], dtype=np.int8)
    representation = np.array(
        [
            [1.0, 0.1, 0.3],
            [0.2, 1.1, 0.2],
            [0.3, 0.2, 1.2],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=True)
    expected = _manual_reverse_scores(seq, representation, kmer=1, with_context=False)

    np.testing.assert_allclose(result.get_slice(0), expected)


def test_batch_all_scores_reverse_complement_with_context_preserves_positions():
    """Reverse-complement BaMM scan should keep the same coordinate convention."""
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = ragged_from_list([seq], dtype=np.int8)
    representation = np.arange(25 * 3, dtype=np.float32).reshape(25, 3) / 10.0

    result = batch_all_scores(sequences, representation, kmer=2, is_revcomp=True, with_context=True)
    expected = _manual_reverse_scores(seq, representation, kmer=2, with_context=True)

    np.testing.assert_allclose(result.get_slice(0), expected)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("co", 1.0),
        ("dice", 4.0 / 7.0),
    ],
)
def test_profile_pairwise_threshold_mask_keeps_real_values(metric, expected):
    """Profile metrics should ignore only dual-below-threshold pairs."""
    data1 = np.array([2.0], dtype=np.float32)
    data2 = np.array([0.8], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.int64)

    score, offset = fast_profile_score(data1, offsets, data2, offsets, 0, 1.0, metric=metric)

    assert score == pytest.approx(expected)
    assert offset == 0


def test_fast_profile_score_rejects_unknown_metric():
    """Shared profile scorer should reject unsupported metric names."""
    data = np.array([1.0], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="'co', 'dice'"):
        fast_profile_score(data, offsets, data, offsets, 0, metric="corr")


def test_strategy_functions_exist():
    """Test that all strategy functions are properly defined"""
    # Test that strategy functions exist and are callable
    assert callable(strategy_motif)
    assert callable(strategy_profile)
    assert callable(strategy_motali)


def test_strategy_profile_uses_motif_offset_convention():
    """Profile strategy should report offsets in the same direction as motif mode."""

    def make_pwm_model(name: str, pwm_core: np.ndarray) -> GenericModel:
        n_row = np.min(pwm_core, axis=0, keepdims=True)
        pwm_ext = np.vstack([pwm_core, n_row]).astype(np.float32)
        return GenericModel(
            type_key="pwm", name=name, representation=pwm_ext, length=pwm_core.shape[1], config={"kmer": 1}
        )

    rng = np.random.default_rng(0)
    core = rng.normal(size=(4, 8)).astype(np.float32)
    model1 = make_pwm_model("m1", core)

    plus_shifted = core[:, 2:7].copy()
    model2_plus = make_pwm_model("m2_plus", plus_shifted)

    rc_source = core[:, 1:7]
    rc_index = np.array([3, 2, 1, 0])
    minus_shifted = rc_source[rc_index][:, ::-1]
    model2_minus = make_pwm_model("m2_minus", minus_shifted)

    seq_rng = np.random.default_rng(1)
    seqs = [seq_rng.integers(0, 4, size=60, dtype=np.int8) for _ in range(200)]
    sequences = ragged_from_list(seqs, dtype=np.int8)

    universal_cfg = create_comparator_config(metric="co", search_range=8, n_permutations=0, seed=1)
    tomtom_cfg = create_comparator_config(metric="pcc", n_permutations=0, seed=1)

    res_profile_plus = strategy_profile(model1, model2_plus, sequences, universal_cfg)
    res_motif_plus = strategy_motif(model1, model2_plus, sequences, tomtom_cfg)
    assert res_profile_plus["orientation"] == "++"
    assert res_motif_plus["orientation"] == "++"
    assert res_profile_plus["offset"] == res_motif_plus["offset"] == 2

    res_profile_minus = strategy_profile(model1, model2_minus, sequences, universal_cfg)
    res_motif_minus = strategy_motif(model1, model2_minus, sequences, tomtom_cfg)
    assert res_profile_minus["orientation"] == "+-"
    assert res_motif_minus["orientation"] == "+-"
    assert res_profile_minus["offset"] == res_motif_minus["offset"] == 1


def test_strategy_profile_uses_disk_cache_for_target_and_query(tmp_path, monkeypatch):
    """Cached query and target profiles should be reused across repeated comparisons."""

    def make_model(name: str) -> GenericModel:
        representation = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.2, 0.8, 0.3],
                [0.1, 0.3, 0.9],
                [0.3, 0.2, 0.1],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return GenericModel(type_key="pwm", name=name, representation=representation, length=3, config={"kmer": 1})

    sequences = ragged_from_list(
        [
            np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
        ],
        dtype=np.int8,
    )
    query = make_model("query")
    target = make_model("target")
    cfg = create_comparator_config(metric="co", cache_mode="on", cache_dir=str(tmp_path), n_permutations=0)

    first = strategy_profile(query, target, sequences, cfg)
    assert first["target"] == "target"
    assert len(list(tmp_path.rglob("*.npz"))) == 3

    fresh_query = make_model("query")
    fresh_target = make_model("target")
    original_scan = strategy_profile.__globals__["scan_model"]

    def guarded_scan(model, current_sequences, strand):
        if model.name == "query":
            raise AssertionError("query scan should be served from disk cache")
        if model.name == "target":
            raise AssertionError("target scan should be served from disk cache")
        return original_scan(model, current_sequences, strand)

    monkeypatch.setitem(strategy_profile.__globals__, "scan_model", guarded_scan)
    second = strategy_profile(fresh_query, fresh_target, sequences, cfg)

    assert second["target"] == "target"
    assert second["score"] == pytest.approx(first["score"])
    assert second["orientation"] == first["orientation"]


def test_clear_cache_removes_cached_profiles(tmp_path):
    """Cache cleanup helper should remove stored profile artifacts."""
    representation = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.2, 0.8, 0.3],
            [0.1, 0.3, 0.9],
            [0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sequences = ragged_from_list([np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8)], dtype=np.int8)
    query = GenericModel(type_key="pwm", name="query", representation=representation, length=3, config={"kmer": 1})
    target = GenericModel(type_key="pwm", name="target", representation=representation, length=3, config={"kmer": 1})
    cfg = create_comparator_config(metric="co", cache_mode="on", cache_dir=str(tmp_path), n_permutations=0)

    strategy_profile(query, target, sequences, cfg)

    removed = clear_cache(str(tmp_path))

    assert removed > 0
    assert not tmp_path.exists()


def test_create_config_builds_unified_config():
    """Unified config builder should create comparator config from kwargs."""
    config = create_config(
        model1="a.meme",
        model2="b.pfm",
        model1_type="pwm",
        model2_type="pwm",
        strategy="profile",
        metric="co",
        n_permutations=10,
        seed=99,
    )

    assert config.strategy == "profile"
    assert config.comparator.metric == "co"
    assert config.comparator.n_permutations == 10
    assert config.seed == 99


def test_run_comparison_with_unified_config_and_models():
    """run_comparison should work with preloaded GenericModel objects."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = ragged_from_list(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2], dtype=np.int8),
        ],
        dtype=np.int8,
    )

    config = create_config(
        model1=model1,
        model2=model2,
        strategy="profile",
        sequences=sequences,
        metric="co",
        n_permutations=0,
        seed=7,
    )
    result = run_comparison(config)

    assert "score" in result
    assert "offset" in result
    assert "orientation" in result


@pytest.mark.parametrize("metric", ["corr", "cj", "l1sim"])
def test_run_comparison_rejects_removed_profile_metrics(metric):
    """Profile mode should reject unsupported legacy metrics."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = ragged_from_list([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)], dtype=np.int8)

    config = create_config(
        model1=model1,
        model2=model2,
        strategy="profile",
        sequences=sequences,
        metric=metric,
        n_permutations=0,
        seed=7,
    )

    with pytest.raises(ValueError, match="co, dice"):
        run_comparison(config)


def test_strategy_profile_rejects_promoters_with_scores_inputs():
    """Calibrated profile mode should reject precomputed numerical profiles."""
    scores_1 = RaggedData(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = RaggedData(np.array([0.2, 0.3, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    promoters = ragged_from_list([np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)], dtype=np.int8)
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric="co", promoters=promoters)

    with pytest.raises(ValueError, match="requires motif"):
        strategy_profile(model1, model2, None, cfg)


@pytest.mark.parametrize("metric", ["co", "dice"])
def test_strategy_profile_handles_all_positions_masked_by_threshold(metric):
    """Pairwise threshold masking should not crash when every aligned pair is filtered out."""
    scores_1 = RaggedData(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = RaggedData(np.array([0.1, 0.2, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric=metric, min_logfpr=10.0, n_permutations=0)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] == pytest.approx(0.0)
    assert result["offset"] == 0
    assert result["orientation"] == "++"


def test_run_comparison_supports_dice_for_profile():
    """Unified API should expose the Dice profile metric."""
    scores_1 = RaggedData(np.array([0.1, 0.5, 1.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = RaggedData(np.array([0.1, 0.5, 0.9], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    config = create_config(model1=model1, model2=model2, strategy="profile", metric="dice", n_permutations=0, seed=7)
    result = run_comparison(config)

    assert result["metric"] == "dice"
    assert 0.0 <= result["score"] <= 1.0


def test_compare_motifs_shortcut_works_with_single_import_api():
    """compare_motifs should provide one-call high-level API."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = ragged_from_list([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)], dtype=np.int8)

    result = compare_motifs(
        model1=model1,
        model2=model2,
        strategy="profile",
        sequences=sequences,
        metric="co",
        n_permutations=0,
        seed=13,
    )

    assert result["query"] == "m1"
    assert result["target"] == "m2"


if __name__ == "__main__":
    pytest.main([__file__])
