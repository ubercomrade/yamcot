"""
Unit tests for key computational functions in mimosa.

These tests validate the correctness of individual functions from:
- mimosa/functions.py
- mimosa/comparison.py
- mimosa/models.py
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pytest
from numba import get_num_threads

from mimosa.api import (
    compare_motifs,
    compare_one_to_many,
    create_config,
    create_many_config,
    run_comparison,
    run_one_to_many,
)
from mimosa.batches import (
    MINUS_STRAND,
    PLUS_STRAND,
    flatten_profile_bundle,
    flatten_valid,
    make_score_batch,
    make_sequence_batch,
    make_strand_bundle,
    profile_row_values,
    row_values,
)
from mimosa.cache import clear_cache
from mimosa.comparison import (
    compare,
    create_comparator_config,
    strategy_motali,
    strategy_motif,
    strategy_profile,
)
from mimosa.comparison import registry as comparison_registry
from mimosa.functions import (
    apply_score_log_tail_table,
    batch_all_scores,
    batch_all_scores_strands,
    build_profile_score_options,
    build_score_log_tail_table,
    cut_prc,
    cut_roc,
    fast_profile_score,
    fast_profile_score_orientations,
    format_params,
    normalize_empirical_log_tail_pair,
    pcm_to_pfm,
    pfm_to_pwm,
    precision_recall_curve,
    roc_curve,
    score_seq,
    scores_to_empirical_log_tail,
    standardized_pauc,
)
from mimosa.io import parse_file_content, read_pfm, read_scores
from mimosa.models import (
    GenericModel,
    calculate_threshold_table,
    get_frequencies,
    get_pfm,
    get_sites,
    read_model,
    scan_model,
    scan_model_strands,
    write_model,
)
from mimosa.models import registry as model_registry

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "models"
EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"
_DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
_JSTACS_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
_LOG_UNIFORM_BASE = float(np.log(4.0))


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode an ACGT string as the project's integer alphabet."""
    return np.array([_DNA_TO_INT[symbol] for symbol in sequence], dtype=np.int8)


def _reverse_complement_encoded(sequence: np.ndarray) -> np.ndarray:
    """Return the reverse complement in the project's integer alphabet."""
    return _RC_TABLE[sequence[::-1]]


def _score_batch_from_flat(data: np.ndarray, offsets: np.ndarray):
    """Build one score batch from flattened values and ragged offsets."""
    rows = [np.asarray(data[offsets[index] : offsets[index + 1]]) for index in range(len(offsets) - 1)]
    return make_score_batch(rows)


def _xml_numeric_value_reference(elem: ET.Element | None) -> float | None:
    """Extract the last numeric scalar from a Jstacs XML element."""
    if elem is None:
        return None

    texts = [text.strip() for text in elem.itertext() if text and text.strip()]
    for text in reversed(texts):
        if _JSTACS_NUMERIC_RE.fullmatch(text):
            return float(text)

    return None


def _xml_array_reference(elem: ET.Element):
    """Recursively convert Jstacs <pos>-based arrays to Python lists."""
    pos_children = [child for child in elem if child.tag == "pos"]
    if not pos_children:
        return _xml_numeric_value_reference(elem)

    return [_xml_array_reference(child) for child in pos_children]


def _logsumexp_reference(values: np.ndarray) -> float:
    """Compute log-sum-exp in float64 for Java-reference scoring."""
    shifted = values - np.max(values)
    return float(np.max(values) + np.log(np.sum(np.exp(shifted))))


def _parse_dimont_tree_reference(elem: ET.Element) -> dict:
    """Parse one Dimont parameter tree using only XML state."""
    pars = elem.find("pars")
    pars_pos = [child for child in pars if child.tag == "pos"] if pars is not None else []
    if pars_pos:
        values = np.asarray(
            [_xml_numeric_value_reference(pos.find("parameter/value")) for pos in pars_pos],
            dtype=np.float64,
        )
        log_z = np.asarray(
            [(_xml_numeric_value_reference(pos.find("parameter/z")) or 0.0) for pos in pars_pos],
            dtype=np.float64,
        )
        return {"scores": values, "log_z": log_z}

    children = elem.find("children")
    assert children is not None
    return {
        "context_pos": int(_xml_numeric_value_reference(elem.find("contextPos"))),
        "children": [_parse_dimont_tree_reference(pos.find("treeElement")) for pos in children if pos.tag == "pos"],
    }


@lru_cache(maxsize=None)
def _load_dimont_reference(path: str) -> tuple[tuple[dict, ...], np.ndarray]:
    """Load the Java-equivalent Dimont tree structure from XML."""
    root = ET.parse(path).getroot()
    model = root.find(".//ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM")
    assert model is not None
    trees = model.find("bayesianNetworkSF/trees")
    assert trees is not None
    parsed_trees = []
    root_offsets = []
    for pos in trees:
        if pos.tag != "pos":
            continue
        node = _parse_dimont_tree_reference(pos.find("parameterTree/root/treeElement"))
        parsed_trees.append(node)

        context_pos_elem = pos.find("parameterTree/contextPoss")
        parents = [int(_xml_numeric_value_reference(child)) for child in context_pos_elem if child.tag == "pos"]
        if parents:
            root_offsets.append(0.0)
        else:
            assert "scores" in node
            root_offsets.append(_logsumexp_reference(node["scores"] + node["log_z"]))

    return tuple(parsed_trees), np.asarray(root_offsets, dtype=np.float64)


def _reference_dimont_site_score(path: Path, sequence: np.ndarray) -> float:
    """Evaluate one site as Dimont log-odds against a uniform single-base background."""
    trees, root_offsets = _load_dimont_reference(str(path))
    total = 0.0
    for position, tree in enumerate(trees):
        node = tree
        while "scores" not in node:
            node = node["children"][int(sequence[node["context_pos"]])]
        total += float(node["scores"][int(sequence[position])]) - float(root_offsets[position])
    return total + len(sequence) * _LOG_UNIFORM_BASE


@lru_cache(maxsize=None)
def _load_slim_reference(path: str) -> tuple[list, list, list]:
    """Load the raw Java SLIM parameter arrays from XML."""
    root = ET.parse(path).getroot()
    slim = root.find(".//SLIM")
    assert slim is not None
    component = _xml_array_reference(slim.find("componentMixtureParameters"))
    ancestor = _xml_array_reference(slim.find("ancestorMixtureParameters"))
    dependency = _xml_array_reference(slim.find("dependencyParameters"))
    return component, ancestor, dependency


def _reference_slim_site_score(path: Path, sequence: np.ndarray) -> float:
    """Evaluate one site as Slim log-odds against a uniform single-base background."""
    component, ancestor, dependency = _load_slim_reference(str(path))
    score = 0.0
    alphabet_size = 4

    def get_offset(start: int, order: int) -> int:
        offset = 0
        current_order = 1
        while current_order < order:
            offset = offset * alphabet_size + int(sequence[start - current_order])
            current_order += 1
        return offset

    def next_context(context: int, position: int, component_index: int, ancestor_index: int) -> int:
        width = len(dependency[position][component_index][0])
        return (context * width + int(sequence[position - component_index - ancestor_index])) % len(
            dependency[position][component_index]
        )

    for position in range(len(component)):
        current_nt = int(sequence[position])
        component_logits = np.asarray(component[position], dtype=np.float64)
        component_log_norm = _logsumexp_reference(component_logits)
        local_scores = []

        independent_logits = np.asarray(dependency[position][0][0], dtype=np.float64)
        independent_log_norm = _logsumexp_reference(independent_logits)
        local_scores.append(
            component_logits[0] - component_log_norm + independent_logits[current_nt] - independent_log_norm
        )

        for component_index in range(1, len(component[position])):
            ancestor_logits = np.asarray(ancestor[position][component_index], dtype=np.float64)
            ancestor_log_norm = _logsumexp_reference(ancestor_logits)
            context = get_offset(position, component_index)
            ancestor_scores = []

            for ancestor_index in range(len(ancestor[position][component_index])):
                context = next_context(context, position, component_index, ancestor_index)
                dependency_logits = np.asarray(dependency[position][component_index][context], dtype=np.float64)
                dependency_log_norm = _logsumexp_reference(dependency_logits)
                ancestor_scores.append(
                    ancestor_logits[ancestor_index]
                    - ancestor_log_norm
                    + dependency_logits[current_nt]
                    - dependency_log_norm
                )

            local_scores.append(
                component_logits[component_index]
                - component_log_norm
                + _logsumexp_reference(np.asarray(ancestor_scores))
            )

        score += _logsumexp_reference(np.asarray(local_scores))

    return score + len(sequence) * _LOG_UNIFORM_BASE


def _reference_fast_profile_score(data1, offsets1, data2, offsets2, search_range, min_value=0.0, metric="co"):
    """Reference implementation of profile scoring for validation tests."""
    inters = np.zeros(2 * search_range + 1, dtype=np.float64)
    sum1s = np.zeros(2 * search_range + 1, dtype=np.float64)
    sum2s = np.zeros(2 * search_range + 1, dtype=np.float64)

    for i in range(len(offsets1) - 1):
        seq1 = data1[offsets1[i] : offsets1[i + 1]]
        seq2 = data2[offsets2[i] : offsets2[i + 1]]

        for k, offset in enumerate(range(-search_range, search_range + 1)):
            idx1_start = 0 if offset < 0 else offset
            idx2_start = -offset if offset < 0 else 0

            if idx1_start >= seq1.size or idx2_start >= seq2.size:
                continue

            overlap = min(seq1.size - idx1_start, seq2.size - idx2_start)
            if overlap <= 0:
                continue

            for j in range(overlap):
                v1 = float(seq1[idx1_start + j])
                v2 = float(seq2[idx2_start + j])
                if min_value > 0.0 and (v1 < min_value or v2 < min_value):
                    continue
                sum1s[k] += v1
                sum2s[k] += v2
                inters[k] += min(v1, v2)

    best_score = -1.0
    best_offset = 0

    for k, offset in enumerate(range(-search_range, search_range + 1)):
        if metric == "co":
            denom = min(sum1s[k], sum2s[k])
            score = inters[k] / denom if denom > 1e-6 else -1.0
        elif metric == "dice":
            denom = sum1s[k] + sum2s[k]
            score = (2.0 * inters[k]) / denom if denom > 1e-6 else -1.0
        else:
            raise ValueError(metric)

        if score > best_score:
            best_score = score
            best_offset = offset

    if best_score < 0.0:
        return 0.0, 0

    return float(best_score), int(best_offset)


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


def test_scores_to_empirical_log_tail_basic():
    """Empirical score normalization should preserve dense masked layout."""
    data = np.array([1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32)
    offsets = np.array([0, 2, 4, 5], dtype=np.int64)
    score_batch = _score_batch_from_flat(data, offsets)

    transformed = scores_to_empirical_log_tail(score_batch)

    assert transformed["values"].shape == score_batch["values"].shape
    assert transformed["mask"].shape == score_batch["mask"].shape
    np.testing.assert_array_equal(transformed["lengths"], score_batch["lengths"])


def test_build_score_log_tail_table_returns_float32():
    """Score log-tail tables should use float32 for faster downstream lookup."""
    table = build_score_log_tail_table(np.array([0.1, 0.3, 0.2, 0.3], dtype=np.float64))
    assert table.dtype == np.float32


def test_apply_score_log_tail_table_preserves_padding_for_empty_rows():
    """Lookup normalization should keep padding intact when no valid scores are present."""
    score_batch = make_score_batch([np.array([], dtype=np.float32)])
    table = build_score_log_tail_table(np.array([0.1, 0.3, 0.2], dtype=np.float32))

    transformed = apply_score_log_tail_table(score_batch, table)

    np.testing.assert_array_equal(transformed["mask"], score_batch["mask"])
    np.testing.assert_array_equal(transformed["values"], score_batch["values"])


def test_read_scores_basic(tmp_path):
    """Numerical score profiles should be parsed from FASTA-like input."""
    path = tmp_path / "scores.fasta"
    path.write_text(">seq1\n0.1 0.2 0.3\n>seq2\n1.0 2.0\n", encoding="utf-8")

    result = read_scores(path)

    assert len(result["lengths"]) == 2
    np.testing.assert_allclose(row_values(result, 0), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(row_values(result, 1), np.array([1.0, 2.0], dtype=np.float32))


def test_format_params_basic():
    """Test basic parameter formatting"""
    params = {"k": 3, "metric": "pcc", "n_perm": 1000}
    formatted = format_params(params)

    # Should be in alphabetical order: k-3_metric-pcc_n_perm-1000
    expected = "k-3_metric-pcc_n_perm-1000"
    assert formatted == expected


def test_generic_model_creation():
    """Test GenericModel creation and basic field wiring."""
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
    """Dimont XML models should load and reproduce exact uniform-background log-odds site scores."""
    model = read_model(str(FIXTURES_ROOT / "dimont" / "exampleD-model-1.xml"), "dimont")
    plus_sequence = make_sequence_batch([_encode_sequence("TTCCAGGGAACCC")])
    minus_sequence = make_sequence_batch([_encode_sequence("GGGTTCCCTGGAA")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 13
    assert model.config["kmer"] == 1
    assert model.representation.shape == (5, 13)
    assert row_values(plus_scores, 0)[0] == pytest.approx(6.65103275172514)
    assert row_values(minus_scores, 0)[0] == pytest.approx(6.65103275172514)


def test_read_model_supports_higher_order_dimont_xml():
    """Higher-order Dimont XML models should scan via the shared context kernel in log-odds space."""
    path = FIXTURES_ROOT / "dimont" / "stat_dimont-model-1.xml"
    model = read_model(str(path), "dimont")
    plus_sequence = make_sequence_batch([_encode_sequence("AACCC")])
    minus_sequence = make_sequence_batch([_encode_sequence("GGGTT")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 5
    assert model.config["kmer"] == 4
    assert model.representation.shape == (5, 5, 5, 5, 5)
    assert row_values(plus_scores, 0)[0] == pytest.approx(_reference_dimont_site_score(path, _encode_sequence("AACCC")))
    assert row_values(minus_scores, 0)[0] == pytest.approx(
        _reference_dimont_site_score(path, _encode_sequence("AACCC"))
    )


def test_read_model_supports_slim_xml_and_matches_example_score():
    """Slim XML models should reproduce the exact uniform-background log-odds site scores."""
    model = read_model(str(FIXTURES_ROOT / "slim" / "example-model-1.xml"), "slim")
    plus_sequence = make_sequence_batch([_encode_sequence("TTCCTCGGAACTGAG")])
    minus_sequence = make_sequence_batch([_encode_sequence("CTCAGTTCCGAGGAA")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "slim"
    assert model.length == 15
    assert model.config["kmer"] == 6
    assert model.representation.shape == (5, 5, 5, 5, 5, 5, 15)
    assert row_values(plus_scores, 0)[0] == pytest.approx(6.243921739188615, abs=2e-5)
    assert row_values(minus_scores, 0)[0] == pytest.approx(6.243921739188615, abs=2e-5)


@pytest.mark.parametrize("path", sorted((FIXTURES_ROOT / "dimont").glob("*.xml")), ids=lambda path: path.name)
def test_dimont_fixture_site_scores_match_java_reference(path: Path):
    """Dimont fixture scores should match the Java parameter-tree semantics on both strands."""
    model = read_model(str(path), "dimont")
    rng = np.random.default_rng(sum(path.name.encode("utf-8")))

    for _ in range(20):
        sequence = rng.integers(0, 4, size=model.length, dtype=np.int8)
        site = make_sequence_batch([sequence])

        plus_score = float(row_values(scan_model(model, site, "+"), 0)[0])
        minus_score = float(row_values(scan_model(model, site, "-"), 0)[0])

        expected_plus = _reference_dimont_site_score(path, sequence)
        expected_minus = _reference_dimont_site_score(path, _reverse_complement_encoded(sequence))

        assert plus_score == pytest.approx(expected_plus, abs=1e-5)
        assert minus_score == pytest.approx(expected_minus, abs=1e-5)


@pytest.mark.parametrize("path", sorted((FIXTURES_ROOT / "slim").glob("*.xml")), ids=lambda path: path.name)
def test_slim_fixture_site_scores_match_java_reference(path: Path):
    """Slim fixture scores should match the Java higher-order mixture formula on both strands."""
    model = read_model(str(path), "slim")
    rng = np.random.default_rng(sum(path.name.encode("utf-8")))

    for _ in range(20):
        sequence = rng.integers(0, 4, size=model.length, dtype=np.int8)
        site = make_sequence_batch([sequence])

        plus_score = float(row_values(scan_model(model, site, "+"), 0)[0])
        minus_score = float(row_values(scan_model(model, site, "-"), 0)[0])

        expected_plus = _reference_slim_site_score(path, sequence)
        expected_minus = _reference_slim_site_score(path, _reverse_complement_encoded(sequence))

        assert plus_score == pytest.approx(expected_plus, abs=1e-5)
        assert minus_score == pytest.approx(expected_minus, abs=1e-5)


def test_read_model_bamm_defaults_to_max_order_and_derives_kmer():
    """BaMM loading should preserve the highest-order tensor unless an explicit order is requested."""
    myog_path = EXAMPLES_ROOT / "myog.ihbcp"
    _, myog_max_order, _ = parse_file_content(str(myog_path))

    default_order = read_model(str(myog_path), "bamm")
    second_order = read_model(str(myog_path), "bamm", order=2)
    foxa1 = read_model(str(FIXTURES_ROOT / "bamm" / "PEAKS036274_FOXA1_P35582_MACS2_motif_1.ihbcp"), "bamm")

    assert default_order.config["order"] == myog_max_order
    assert default_order.config["kmer"] == myog_max_order + 1
    assert default_order.representation.ndim == myog_max_order + 2
    assert second_order.config["kmer"] == 3
    assert second_order.config["order"] == 2
    assert second_order.representation.shape == (5, 5, 5, 14)
    assert default_order.representation.shape != second_order.representation.shape

    assert foxa1.config["kmer"] == foxa1.config["order"] + 1
    assert foxa1.config["order"] > 0
    assert foxa1.representation.ndim == foxa1.config["order"] + 2


def test_read_model_bamm_ignores_missing_background_and_uses_uniform_log_odds(tmp_path):
    """BaMM loading should no longer require a background file when using a uniform background."""
    motif_path = tmp_path / "toy.ihbcp"
    motif_path.write_text("0.25 0.25 0.25 0.25\n", encoding="utf-8")

    model = read_model(str(motif_path), "bamm")

    assert model.config["kmer"] == 1
    np.testing.assert_allclose(model.representation[:4, 0], np.zeros(4, dtype=np.float32), atol=1e-6)


def test_create_comparator_config():
    """Test ComparatorConfig creation and factory function"""
    # Test factory function with defaults
    config = create_comparator_config()
    assert config["metric"] == "pcc"
    assert config["n_permutations"] == 0
    assert config["seed"] is None
    assert config["pfm_top_fraction"] == pytest.approx(0.05)
    assert config["profile_normalization"] == "empirical_log_tail"
    assert config["numba_threads"] is None

    # Test factory function with custom parameters
    config = create_comparator_config(metric="co", n_permutations=100, seed=42, pfm_top_fraction=0.2)
    assert config["metric"] == "co"
    assert config["n_permutations"] == 100
    assert config["seed"] == 42
    assert config["pfm_top_fraction"] == pytest.approx(0.2)


def test_create_comparator_config_resolves_numba_threads():
    """Explicit numba_threads should drive the effective parallel setting."""
    config = create_comparator_config(numba_threads=4)
    assert config["numba_threads"] == 4


@pytest.mark.parametrize("kwargs", [{"numba_threads": 0}, {"numba_threads": -2}])
def test_create_comparator_config_validates_thread_counts(kwargs):
    """Thread-count settings should accept only positive values or -1."""
    with pytest.raises(ValueError, match="positive or -1"):
        create_comparator_config(**kwargs)


def test_compare_applies_numba_threads_temporarily(monkeypatch):
    """Execution boundary should apply and restore numba thread settings."""
    observed = []
    original_threads = get_num_threads()

    def fake_strategy(model1, model2, sequences, cfg):
        observed.append((cfg["numba_threads"], get_num_threads()))
        return {"score": 1.0}

    monkeypatch.setitem(comparison_registry, "thread_test", fake_strategy)

    result = compare(
        model1=None,
        model2=None,
        strategy="thread_test",
        config=create_comparator_config(numba_threads=1),
    )

    assert result == {"score": 1.0}
    assert observed == [(1, 1)]
    assert get_num_threads() == original_threads


def test_create_comparator_config_validates_kernel_range():
    """Kernel-size range should be valid for centered surrogate kernels."""
    with pytest.raises(ValueError, match="min_kernel_size"):
        create_comparator_config(min_kernel_size=7, max_kernel_size=5)

    with pytest.raises(ValueError, match="at least one odd value"):
        create_comparator_config(min_kernel_size=4, max_kernel_size=4)

    config = create_comparator_config(min_kernel_size=4, max_kernel_size=6)
    assert config["min_kernel_size"] == 4
    assert config["max_kernel_size"] == 6


def test_create_comparator_config_validates_min_logfpr():
    """Profile floor should reject negative logFPR thresholds."""
    with pytest.raises(ValueError, match="min_logfpr"):
        create_comparator_config(min_logfpr=-0.1)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_create_comparator_config_validates_pfm_top_fraction(value):
    """PFM site-selection fraction should stay in the open-closed unit interval."""
    with pytest.raises(ValueError, match="pfm_top_fraction"):
        create_comparator_config(pfm_top_fraction=value)


def test_create_comparator_config_validates_cache_mode():
    """Profile cache mode should accept only explicit on/off values."""
    with pytest.raises(ValueError, match="cache_mode"):
        create_comparator_config(cache_mode="targets")

    config = create_comparator_config(cache_mode="on")
    assert config["cache_mode"] == "on"


def test_create_comparator_config_validates_profile_normalization():
    """Profile normalization mode should accept only known strategies."""
    with pytest.raises(ValueError, match="profile_normalization"):
        create_comparator_config(profile_normalization="zscore")

    config = create_comparator_config(profile_normalization="empirical_log_tail")
    assert config["profile_normalization"] == "empirical_log_tail"


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
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),  # A,C,G,T,C,G,A
            np.array([1, 2, 3, 0], dtype=np.int8),  # C,G,T,A
        ]
    )

    # Test scanning
    scores = scan_model(model, sequences, "+")
    assert flatten_valid(scores).size > 0


def test_scan_model_strands_returns_strand_bundle():
    """Two-strand scanning should return one bundle with shape strands x rows x cols."""
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
    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=3, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),
            np.array([1, 2, 3, 0], dtype=np.int8),
        ]
    )

    strand_bundle = scan_model_strands(model, sequences)
    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")

    assert strand_bundle["values"].shape[0] == 2
    np.testing.assert_array_equal(strand_bundle["lengths"], plus_scores["lengths"])
    np.testing.assert_allclose(strand_bundle["values"][PLUS_STRAND], plus_scores["values"])
    np.testing.assert_allclose(strand_bundle["values"][MINUS_STRAND], minus_scores["values"])


def test_get_frequencies():
    """Test frequency calculation"""
    representation = np.array([[0.2, 0.3], [0.3, 0.2], [0.2, 0.4], [0.3, 0.1], [0.1, 0.1]])

    model = GenericModel("pwm", "test", representation, 2, {"kmer": 1})

    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3], dtype=np.int8),
            np.array([1, 2, 3, 0], dtype=np.int8),
        ]
    )

    frequencies = get_frequencies(model, sequences, "+")
    assert flatten_valid(frequencies).size > 0


def test_calculate_threshold_table_does_not_mutate_model_config():
    """Threshold-table calculation should stay pure and avoid storing runtime lookup tables on the model."""
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
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
            np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
        ]
    )

    table = calculate_threshold_table(model, sequences, strand="best")

    assert table.shape[1] == 2
    assert "_threshold_table" not in model.config
    assert "_threshold_tables" not in model.config


def test_get_pfm_reconstructs_pwm_from_sites_with_single_pseudocount():
    """PFM reconstruction should ignore source PFM caches and add the pseudocount only once."""
    representation = np.array(
        [
            [5.0, 0.2],
            [0.1, 4.5],
            [0.1, 0.1],
            [0.1, 1.5],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    source_pfm = np.full((4, 2), 0.25, dtype=np.float32)
    model = GenericModel(
        type_key="pwm",
        name="test_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_source_pfm": source_pfm},
    )
    sequences = make_sequence_batch(
        [
            _encode_sequence("AC"),
            _encode_sequence("AC"),
            _encode_sequence("AT"),
        ]
    )

    pfm = get_pfm(model, sequences, pseudocount=0.25)
    expected_pcm = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expected = pcm_to_pfm(expected_pcm, pseudocount=0.25).astype(np.float32)

    np.testing.assert_allclose(pfm, expected)
    assert not np.allclose(pfm, source_pfm)


def test_write_model_serializes_pwm_source_pfm(tmp_path):
    """PWM serialization should write the stored source PFM via the shared PFM writer."""
    source_pfm = np.array([[0.7, 0.1], [0.1, 0.7], [0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    pwm = pfm_to_pwm(source_pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    model = GenericModel(
        type_key="pwm",
        name="serialized_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_source_pfm": source_pfm},
    )

    path = tmp_path / "serialized.pfm"
    write_model(model, str(path))

    assert path.exists()
    written_pfm, length = read_pfm(str(path))
    assert length == 2
    np.testing.assert_allclose(written_pfm, source_pfm, atol=1e-6)


def test_read_model_rejects_pwm_pickle_without_source_pfm(tmp_path):
    """Legacy PWM pickles without a source PFM should no longer be accepted."""
    source_pfm = np.full((4, 2), 0.25, dtype=np.float32)
    pwm = pfm_to_pwm(source_pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    legacy_model = GenericModel(
        type_key="pwm",
        name="legacy_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_pfm": source_pfm},
    )
    path = tmp_path / "legacy_pwm.pkl"
    joblib.dump(legacy_model, path)

    with pytest.raises(ValueError, match="_source_pfm"):
        read_model(str(path), "pwm")


def test_get_sites_threshold_uses_current_sequences_by_default():
    """Thresholded site extraction should use the current sequences unless an external background is passed."""
    representation = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(
        type_key="pwm",
        name="threshold_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1},
    )
    sequences = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")]
    )
    background = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG"), _encode_sequence("TG")]
    )

    default_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5)
    background_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5, background_sequences=background)

    assert default_sites["site"].tolist() == ["AC", "AC"]
    assert background_sites["site"].tolist() == ["AC", "AC", "AT", "AT"]


def test_get_pfm_threshold_accepts_external_background_sequences():
    """PFM reconstruction should switch to external calibration only when requested."""
    representation = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(
        type_key="pwm",
        name="threshold_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1},
    )
    sequences = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")]
    )
    background = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG"), _encode_sequence("TG")]
    )

    default_pfm = get_pfm(model, sequences, mode="threshold", fpr_threshold=0.5, pseudocount=0.25)
    background_pfm = get_pfm(
        model,
        sequences,
        mode="threshold",
        fpr_threshold=0.5,
        background_sequences=background,
        pseudocount=0.25,
    )

    expected_default_pcm = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    expected_background_pcm = np.array([[4.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(default_pfm, pcm_to_pfm(expected_default_pcm, pseudocount=0.25).astype(np.float32))
    np.testing.assert_allclose(background_pfm, pcm_to_pfm(expected_background_pcm, pseudocount=0.25).astype(np.float32))


def test_get_sites_best_skips_sequences_shorter_than_motif():
    """Best-site extraction should return no hits instead of crashing on short sequences."""
    representation = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="short_test", representation=representation, length=4, config={"kmer": 1})
    sequences = make_sequence_batch([_encode_sequence("ACG")])

    result = get_sites(model, sequences, mode="best")

    assert result.empty


def test_batch_all_scores_with_simple_data():
    """Test batch_all_scores with a simple dense masked batch."""
    data = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = make_sequence_batch([data[:3], data[3:]])
    representation = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=False)
    assert result["values"].shape == (2, 2)
    np.testing.assert_array_equal(result["lengths"], np.array([2, 2], dtype=np.int64))


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
    sequences = make_sequence_batch([seq])
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

    np.testing.assert_allclose(row_values(result, 0), expected)


def test_batch_all_scores_reverse_complement_with_context_preserves_positions():
    """Reverse-complement BaMM scan should keep the same coordinate convention."""
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = make_sequence_batch([seq])
    representation = np.arange(25 * 3, dtype=np.float32).reshape(25, 3) / 10.0

    result = batch_all_scores(sequences, representation, kmer=2, is_revcomp=True, with_context=True)
    expected = _manual_reverse_scores(seq, representation, kmer=2, with_context=True)

    np.testing.assert_allclose(row_values(result, 0), expected)


@pytest.mark.parametrize(
    ("representation", "kmer", "with_context"),
    [
        (
            np.array(
                [
                    [1.0, 0.1, 0.3],
                    [0.2, 1.1, 0.2],
                    [0.3, 0.2, 1.2],
                    [0.4, 0.5, 0.6],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            1,
            False,
        ),
        (np.arange(25 * 3, dtype=np.float32).reshape(25, 3) / 10.0, 2, True),
    ],
)
def test_batch_all_scores_strands_matches_separate_calls(representation, kmer, with_context):
    """Two-strand batch scanning should match separate forward and reverse passes."""
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 0, 1], dtype=np.int8),
            np.array([3, 2, 1, 0, 3], dtype=np.int8),
        ]
    )

    expected_plus = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
    expected_minus = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
    plus_batch, minus_batch = batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)

    np.testing.assert_allclose(plus_batch["values"], expected_plus["values"])
    np.testing.assert_array_equal(plus_batch["mask"], expected_plus["mask"])
    np.testing.assert_array_equal(plus_batch["lengths"], expected_plus["lengths"])
    np.testing.assert_allclose(minus_batch["values"], expected_minus["values"])
    np.testing.assert_array_equal(minus_batch["mask"], expected_minus["mask"])
    np.testing.assert_array_equal(minus_batch["lengths"], expected_minus["lengths"])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("co", 0.0),
        ("dice", 0.0),
    ],
)
def test_profile_pairwise_threshold_mask_requires_both_values_above_threshold(metric, expected):
    """Profile metrics should keep pairs only when both values satisfy the threshold."""
    data1 = np.array([2.0], dtype=np.float32)
    data2 = np.array([0.8], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.int64)

    score, offset = fast_profile_score(
        _score_batch_from_flat(data1, offsets),
        _score_batch_from_flat(data2, offsets),
        build_profile_score_options(search_range=0, min_value=1.0, metric=metric),
    )

    assert score == pytest.approx(expected)
    assert offset == 0


def test_fast_profile_score_rejects_unknown_metric():
    """Shared profile scorer should reject unsupported metric names."""
    data = np.array([1.0], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="'co', 'dice'"):
        fast_profile_score(
            _score_batch_from_flat(data, offsets),
            _score_batch_from_flat(data, offsets),
            build_profile_score_options(search_range=0, metric="corr"),
        )


@pytest.mark.parametrize("metric", ["co", "dice"])
@pytest.mark.parametrize("min_value", [0.0, 1.0])
def test_fast_profile_score_matches_reference_for_ragged_inputs(metric, min_value):
    """Optimized profile scorer should match the dense reference on masked batches."""
    rng = np.random.default_rng(42)
    lengths = np.array([2, 5, 7, 4], dtype=np.int64)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    data1 = rng.gamma(shape=1.5, scale=1.0, size=int(offsets[-1])).astype(np.float32)
    data2 = rng.gamma(shape=1.7, scale=0.9, size=int(offsets[-1])).astype(np.float32)

    score, offset = fast_profile_score(
        _score_batch_from_flat(data1, offsets),
        _score_batch_from_flat(data2, offsets),
        build_profile_score_options(search_range=3, min_value=min_value, metric=metric),
    )
    expected_score, expected_offset = _reference_fast_profile_score(
        data1, offsets, data2, offsets, 3, min_value=min_value, metric=metric
    )

    assert score == pytest.approx(expected_score)
    assert offset == expected_offset


@pytest.mark.parametrize("metric", ["co", "dice"])
def test_fast_profile_score_normalizes_non_contiguous_inputs(metric):
    """Profile scorer should normalize dtype and contiguity before dispatch."""
    base1 = np.array([0.0, 0.6, 0.0, 1.2, 0.0, 0.8, 0.0, 0.4], dtype=np.float64)
    base2 = np.array([0.0, 0.5, 0.0, 0.7, 0.0, 1.1, 0.0, 0.3], dtype=np.float64)
    data1 = base1[1::2]
    data2 = base2[1::2]
    offsets = np.array([0, 2, 4], dtype=np.int32)

    score, offset = fast_profile_score(
        _score_batch_from_flat(data1, offsets),
        _score_batch_from_flat(data2, offsets),
        build_profile_score_options(search_range=1, min_value=0.5, metric=metric),
    )
    expected_score, expected_offset = _reference_fast_profile_score(
        np.ascontiguousarray(data1, dtype=np.float32),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(data2, dtype=np.float32),
        np.ascontiguousarray(offsets, dtype=np.int64),
        1,
        min_value=0.5,
        metric=metric,
    )

    assert score == pytest.approx(expected_score)
    assert offset == expected_offset


@pytest.mark.parametrize("metric", ["co", "dice"])
def test_fast_profile_score_thresholded_mask_matches_reference(metric):
    """Profile scorer should match the reference under explicit threshold masking."""
    data1 = np.array([0.2, 1.3, 0.4, 1.1, 0.1, 0.9], dtype=np.float32)
    data2 = np.array([0.3, 0.7, 1.4, 0.2, 1.2, 0.1], dtype=np.float32)
    offsets = np.array([0, 3, 6], dtype=np.int64)

    score, offset = fast_profile_score(
        _score_batch_from_flat(data1, offsets),
        _score_batch_from_flat(data2, offsets),
        build_profile_score_options(search_range=1, min_value=1.0, metric=metric),
    )
    expected_score, expected_offset = _reference_fast_profile_score(
        data1, offsets, data2, offsets, 1, min_value=1.0, metric=metric
    )

    assert score == pytest.approx(expected_score)
    assert offset == expected_offset


@pytest.mark.parametrize("metric", ["co", "dice"])
def test_fast_profile_score_orientations_matches_pairwise(metric):
    """Batch orientation scorer should match pairwise profile scoring."""
    rng = np.random.default_rng(13)
    lengths = np.array([4, 6, 5], dtype=np.int64)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    query_plus = _score_batch_from_flat(rng.uniform(0.0, 3.0, size=int(offsets[-1])).astype(np.float32), offsets)
    query_minus = _score_batch_from_flat(rng.uniform(0.0, 3.0, size=int(offsets[-1])).astype(np.float32), offsets)
    target_plus = _score_batch_from_flat(rng.uniform(0.0, 3.0, size=int(offsets[-1])).astype(np.float32), offsets)
    target_minus = _score_batch_from_flat(rng.uniform(0.0, 3.0, size=int(offsets[-1])).astype(np.float32), offsets)
    options = build_profile_score_options(search_range=2, min_value=0.5, metric=metric)
    query_bundle = make_strand_bundle(query_plus, query_minus)
    target_bundle = make_strand_bundle(target_plus, target_minus)
    strand_pairs = [
        (PLUS_STRAND, PLUS_STRAND),
        (MINUS_STRAND, MINUS_STRAND),
        (PLUS_STRAND, MINUS_STRAND),
        (MINUS_STRAND, PLUS_STRAND),
    ]

    orientation_pairs = [
        (query_plus, target_plus),
        (query_minus, target_minus),
        (query_plus, target_minus),
        (query_minus, target_plus),
    ]
    scores, orientation_offsets = fast_profile_score_orientations(query_bundle, target_bundle, strand_pairs, options)

    assert scores.shape == (4,)
    assert orientation_offsets.shape == (4,)

    for pair_index, (query_profile, target_profile) in enumerate(orientation_pairs):
        expected_score, expected_offset = fast_profile_score(query_profile, target_profile, options)
        assert scores[pair_index] == pytest.approx(expected_score)
        assert int(orientation_offsets[pair_index]) == expected_offset


def test_profile_orientation_search_requires_minus_plus_candidate():
    """Restricting profile search to ++ and +- should miss valid -+ optima."""
    query_plus = _score_batch_from_flat(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    query_minus = _score_batch_from_flat(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    target_plus = _score_batch_from_flat(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    target_minus = _score_batch_from_flat(np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    options = build_profile_score_options(search_range=0, min_value=0.0, metric="co")
    orientation_labels = ("++", "--", "+-", "-+")
    query_bundle = make_strand_bundle(query_plus, query_minus)
    target_bundle = make_strand_bundle(target_plus, target_minus)
    scores, orientation_offsets = fast_profile_score_orientations(
        query_bundle,
        target_bundle,
        [
            (PLUS_STRAND, PLUS_STRAND),
            (MINUS_STRAND, MINUS_STRAND),
            (PLUS_STRAND, MINUS_STRAND),
            (MINUS_STRAND, PLUS_STRAND),
        ],
        options,
    )

    assert int(np.argmax(scores)) == 3
    assert orientation_labels[int(np.argmax(scores))] == "-+"
    assert scores[3] == pytest.approx(1.0)
    assert scores[3] > max(float(scores[0]), float(scores[2]))
    np.testing.assert_array_equal(orientation_offsets, np.zeros(4, dtype=np.int32))


def test_profile_orientation_search_requires_all_four_candidates_for_pwm_models():
    """PFM-derived PWM profile comparison can prefer -+ over the reduced ++ / +- search space."""

    def make_pwm_from_pfm(name: str, pfm: np.ndarray) -> GenericModel:
        pwm = pfm_to_pwm(pfm).astype(np.float32)
        representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0)
        return GenericModel(
            type_key="pwm",
            name=name,
            representation=representation,
            length=pfm.shape[1],
            config={"kmer": 1, "_source_pfm": pfm.astype(np.float32)},
        )

    pfm_1 = np.array(
        [
            [0.03364587, 0.00353458, 0.345236, 0.09329311, 0.15702443, 0.01077098, 0.00142371, 0.13933925, 0.00131438,
             0.0057433],
            [0.78910834, 0.37919682, 0.2486873, 0.00466091, 0.1745861, 0.08962669, 0.83672076, 0.62271661, 0.11432868,
             0.14217305],
            [0.12944466, 0.03434311, 0.01618065, 0.89675272, 0.03784276, 0.55884331, 0.05381118, 0.2102953, 0.00582364,
             0.66432488],
            [0.04780111, 0.5829255, 0.38989606, 0.00529327, 0.63054669, 0.34075904, 0.10804431, 0.02764886, 0.8785333,
             0.18775877],
        ],
        dtype=np.float32,
    )
    pfm_2 = np.array(
        [
            [0.00277317, 0.05549242, 0.42488962, 0.90949905, 0.06935914, 0.00925176, 0.85040897, 0.04507855, 0.9808197,
             0.03197747],
            [0.21828328, 0.02460412, 0.00311564, 0.05960011, 0.65439123, 0.98875666, 0.00032324, 0.19157799, 0.01060591,
             0.95191884],
            [0.05031024, 0.87492639, 0.01184713, 0.00106371, 0.27624866, 0.00150903, 0.14537251, 0.76326233, 0.00749461,
             0.01601587],
            [0.72863328, 0.04497707, 0.56014758, 0.02983714, 0.00000094, 0.00048257, 0.00389528, 0.00008116, 0.00107979,
             0.00008784],
        ],
        dtype=np.float32,
    )
    pwm_1 = make_pwm_from_pfm(
        "pwm_1",
        pfm_1,
    )
    pwm_2 = make_pwm_from_pfm(
        "pwm_2",
        pfm_2,
    )
    sequence_strings = [
        "CAGATCTGAGGCCACAATCGGGTTCCTCCCAA",
        "CACATGCCCAATCTCCACACCGCAAGTTCTG",
        "ACGTACGGTTTGTTCATAGACATCGCTG",
        "TGATTTGTATAGGGGGCGAAATCG",
        "AACGTCACACTTCTCAGAACTTCGATAGCCCTTG",
        "AACGGTGCACCAAGTGTGCGCACTAT",
        "GGGGGACAGCATTCGCGGATTATGG",
        "ACACAGGATTCATTATTAGTATCT",
        "ATCTTCTGTAACGTTCTTAAAGAGTAT",
        "TGTCAATGGACCATATGGCCTCA",
        "CTACACTGTCTGTATGAACACTATTGACTCACC",
        "CGTGAGGACCAACTATGAGGCCC",
        "AAGGTATCAACTTCCCGCGTCATTCTTC",
        "TGATGTTCGTCCAAAGCGTTCGGCC",
        "CCCTATCCTGTGGCCGTCATGCGCACCGTTGACC",
        "TCATGGGCAGATGTTTACGAGGC",
        "ATATACGACCTCTCAGTGTGGAGA",
        "GCGACAGGTCCAATCTGACGAGTG",
        "AGCTGTCTCCCGACCGTGATGAAACGCGGAGATC",
        "CGAACGCGGCCTAAGTGTCGTCTCAGTTGAGTTA",
        "CGCATAGAGAATAGGTTATCGTT",
        "TCAGATCTTCTCACATGTAGAAGCTTGCTCCGGCA",
        "CGGTGCGTTACTTCCCCCAGG",
        "ACCCCGTGGCTCCGCATAATCA",
        "GGATAATATGCGGTCCTTCATAC",
        "GGGTTGTATATGTGCTGCAG",
        "AGAGGGGGCAAATTGATACACCGAGTC",
        "CAAGGCGTATGCAGCTGATGGGTTAGCAAGACTGT",
        "GCCGAACTCACTCTCTTACCACCAACGCGG",
        "ACACATGCGCTCCAGGTATG",
        "ATGTGAACCGACTGTAAAGTG",
        "CTGGGCTACCTATGTTGAGA",
        "AGCGCTCATATGCAACGGCTAGTC",
        "CGGGCAATAGCACGGCGTAGGT",
        "TATTACTCGCGCCATACTCTTCCCCAGGGTA",
        "CGTCGTCTAGCCGATTCTTTTGCGGCGCCAACTGG",
        "CATTGCTCCATCATCTACGGACTTCTACGTACAA",
        "ATACCGCAATCGTGTGAGCTTGGTCCACAA",
        "GGGTCCATACCGAGCGGCACCTGTAACCTCCAGGG",
        "TACTCGCTATTAGGAATCACGGGG",
        "GGAAGACAACTAAGGCACCCTATCCAGTG",
        "TTCGCATCATCTGACGAAGTGCCAGGTCT",
        "ACAGAGATAATGCCCGGGCCCAGCAGGTTG",
        "TTACCTAGTTGTTTACCCGTGTTGG",
        "TTCGTCCATTTTACTACTCCTAAGGGCGGGCT",
        "CCAACAAATAGGCAGATCTCTCTTC",
        "CTCAGCTCGGACACTTCACATGGCG",
        "TAGCAGACGCCTTTCTACCTCCGA",
    ]
    sequences = make_sequence_batch(
        [_encode_sequence(sequence) for sequence in sequence_strings]
    )
    cfg = create_comparator_config(metric="co", n_permutations=0, search_range=16)
    calibration_sequences = strategy_profile.__globals__["_get_profile_calibration_sequences"](sequences, cfg)
    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    prepare_profile_bundle = strategy_profile.__globals__["_prepare_profile_bundle_for_scoring"]

    bundle_1 = resolve_profile_bundle(pwm_1, sequences, calibration_sequences, cfg)
    bundle_2 = resolve_profile_bundle(pwm_2, sequences, calibration_sequences, cfg)
    prepared_1 = prepare_profile_bundle(bundle_1, 0.0)
    prepared_2 = prepare_profile_bundle(bundle_2, 0.0)
    orientation_pairs = [
        ("++", PLUS_STRAND, PLUS_STRAND),
        ("--", MINUS_STRAND, MINUS_STRAND),
        ("+-", PLUS_STRAND, MINUS_STRAND),
        ("-+", MINUS_STRAND, PLUS_STRAND),
    ]
    min_length = min(pwm_1.length, pwm_2.length)
    expected_offset = min_length // 2

    scores, offsets = fast_profile_score_orientations(
        prepared_1,
        prepared_2,
        [(query_strand, target_strand) for _, query_strand, target_strand in orientation_pairs],
        build_profile_score_options(search_range=cfg["search_range"], min_value=0.0, metric="co"),
    )
    score_by_orientation = {
        orientation: float(score)
        for (orientation, _query_strand, _target_strand), score in zip(orientation_pairs, scores, strict=True)
    }
    offset_by_orientation = {
        orientation: int(offset)
        for (orientation, _query_strand, _target_strand), offset in zip(orientation_pairs, offsets, strict=True)
    }
    reduced_best = max(("++", "+-"), key=score_by_orientation.get)
    full_best = max(score_by_orientation, key=score_by_orientation.get)
    result = strategy_profile(pwm_1, pwm_2, sequences, cfg)
    overlap = min_length - abs(result["offset"])

    assert reduced_best == "+-"
    assert full_best == "-+"
    assert score_by_orientation["-+"] > score_by_orientation["+-"]
    assert offset_by_orientation["-+"] == expected_offset
    assert overlap == min_length // 2
    assert result["orientation"] == "-+"
    assert result["score"] == pytest.approx(score_by_orientation["-+"])
    assert result["offset"] == -expected_offset


def test_strategy_functions_exist():
    """Test that all strategy functions are properly defined"""
    # Test that strategy functions exist and are callable
    assert callable(strategy_motif)
    assert callable(strategy_profile)
    assert callable(strategy_motali)


def test_strategy_profile_uses_target_relative_offset_convention():
    """Profile strategy should report offsets using the same target-relative convention as pairwise scoring."""
    scores_1 = _score_batch_from_flat(
        np.array([0.0, 1.0, 4.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0, 7], dtype=np.int64),
    )
    scores_2 = _score_batch_from_flat(
        np.array([0.0, 0.0, 0.0, 1.0, 4.0, 2.0, 0.0], dtype=np.float32),
        np.array([0, 7], dtype=np.int64),
    )
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric="co", search_range=4, min_logfpr=0.1, n_permutations=0)
    options = build_profile_score_options(search_range=4, min_value=0.1, metric="co")

    forward = strategy_profile(model1, model2, None, cfg)
    reverse = strategy_profile(model2, model1, None, cfg)
    expected_forward_score, expected_forward_offset = fast_profile_score(scores_1, scores_2, options)
    expected_reverse_score, expected_reverse_offset = fast_profile_score(scores_2, scores_1, options)

    assert forward["orientation"] == "++"
    assert reverse["orientation"] == "++"
    assert forward["score"] == pytest.approx(expected_forward_score)
    assert reverse["score"] == pytest.approx(expected_reverse_score)
    assert forward["offset"] == -expected_forward_offset
    assert reverse["offset"] == -expected_reverse_offset


def test_strategy_profile_empirical_uses_combined_strand_table():
    """Empirical profile normalization should use one combined +/- calibration table."""
    site = "CAGTAAACAG"
    rng = np.random.default_rng(0)
    encoded_site = _encode_sequence(site)
    sequences = []

    for _ in range(300):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co", n_permutations=0)

    plus_scores = scan_model(dimont, ragged_sequences, "+")
    minus_scores = scan_model(dimont, ragged_sequences, "-")
    combined_table = build_score_log_tail_table(
        np.concatenate((flatten_valid(plus_scores), flatten_valid(minus_scores)))
    )

    expected_plus = apply_score_log_tail_table(plus_scores, combined_table)
    expected_minus = apply_score_log_tail_table(minus_scores, combined_table)

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(dimont, ragged_sequences, ragged_sequences, cfg)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, MINUS_STRAND),
        flatten_valid(expected_minus),
        atol=1e-6,
    )

    plus_slice = profile_row_values(resolved, PLUS_STRAND, 0)
    minus_slice = profile_row_values(resolved, MINUS_STRAND, 0)
    assert plus_slice[30] > minus_slice[29]


def test_strategy_profile_uses_promoters_for_empirical_calibration():
    """Profile normalization should use promoter scans when explicit calibration sequences are provided."""
    representation = np.array(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="m1", representation=representation, length=2, config={"kmer": 1})
    sequences = make_sequence_batch([_encode_sequence("ACACAC"), _encode_sequence("CAAAAA")])
    promoters = make_sequence_batch([_encode_sequence("AAAAAA"), _encode_sequence("CCCCCC")])
    cfg = create_comparator_config(metric="co", promoters=promoters, n_permutations=0)

    plus_scores = scan_model(model, sequences, "+")
    promoter_plus = scan_model(model, promoters, "+")
    promoter_minus = scan_model(model, promoters, "-")
    table = build_score_log_tail_table(np.concatenate((flatten_valid(promoter_plus), flatten_valid(promoter_minus))))
    expected_plus = apply_score_log_tail_table(plus_scores, table)

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(model, sequences, promoters, cfg)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )


def test_resolve_profile_bundle_matches_direct_two_strand_normalization():
    """Resolved profile bundle should match direct two-strand normalization exactly."""
    representation = np.array(
        [
            [1.2, -0.3],
            [-0.4, 1.0],
            [-0.5, -0.4],
            [-0.3, -0.5],
            [-0.5, -0.5],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="joint", representation=representation, length=2, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            _encode_sequence("ACGTAC"),
            _encode_sequence("TGCATG"),
            _encode_sequence("AAAAAC"),
        ]
    )
    cfg = create_comparator_config(metric="co", n_permutations=0, cache_mode="off")

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(model, sequences, sequences, cfg)

    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")
    expected_plus, expected_minus = normalize_empirical_log_tail_pair(plus_scores, minus_scores)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, MINUS_STRAND),
        flatten_valid(expected_minus),
        atol=1e-6,
    )


def test_strategy_motif_handles_reverse_complement_for_higher_order_tensors():
    """Higher-order tensor comparison should reverse both nucleotide values and context-axis order."""
    rng = np.random.default_rng(7)
    core = rng.normal(size=(4, 4, 6)).astype(np.float32)
    rep1 = np.full((5, 5, 6), -3.0, dtype=np.float32)
    rep1[:4, :4, :] = core

    core_rc = np.transpose(core, (1, 0, 2))[::-1, ::-1, ::-1]
    rep2 = np.full((5, 5, 6), -3.0, dtype=np.float32)
    rep2[:4, :4, :] = core_rc

    model1 = GenericModel(type_key="sitega", name="m1", representation=rep1, length=6, config={"kmer": 2})
    model2 = GenericModel(type_key="sitega", name="m2", representation=rep2, length=6, config={"kmer": 2})

    result = strategy_motif(model1, model2, None, create_comparator_config(metric="cosine", n_permutations=0))

    assert result["orientation"] == "+-"
    assert result["score"] == pytest.approx(1.0, abs=1e-6)


def test_foxa1_cross_type_motif_comparison_recovers_dimont_and_slim_similarity():
    """FOXA1 higher-order models should stay aligned with PWM/BaMM after PFM reconstruction."""
    rng = np.random.default_rng(1)
    sequences = make_sequence_batch([rng.integers(0, 4, size=40, dtype=np.int8) for _ in range(2000)])

    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    bamm = read_model(str(FIXTURES_ROOT / "bamm" / "PEAKS036274_FOXA1_P35582_MACS2_motif_1.ihbcp"), "bamm", order=2)
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    slim = read_model(str(FIXTURES_ROOT / "slim" / "PEAKS036274_FOXA1_P35582_MACS2-model-2.xml"), "slim")
    cfg = create_comparator_config(metric="pcc", n_permutations=0)

    pwm_dimont = strategy_motif(pwm, dimont, sequences, cfg)
    pwm_slim = strategy_motif(pwm, slim, sequences, cfg)
    bamm_dimont = strategy_motif(bamm, dimont, sequences, cfg)
    bamm_slim = strategy_motif(bamm, slim, sequences, cfg)
    bamm_pwm = strategy_motif(bamm, pwm, sequences, cfg)

    assert pwm_dimont["orientation"] == "+-"
    assert pwm_dimont["score"] > 0.60

    assert pwm_slim["orientation"] == "++"
    assert pwm_slim["score"] > 0.68

    assert bamm_dimont["orientation"] == "+-"
    assert bamm_dimont["score"] > 0.54

    assert bamm_slim["orientation"] == "++"
    assert bamm_slim["score"] > 0.65

    assert bamm_pwm["orientation"] == "--"
    assert bamm_pwm["score"] > 0.85


def test_strategy_profile_co_uses_sparse_signal_on_foxa1_dimont_pwm_sites():
    """CO profile comparison should stay stable for FOXA1 profiles after strand normalization."""
    site = "CCAGAGTAAACAG"
    dna_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = np.array([dna_to_int[base] for base in site], dtype=np.int8)

    for _ in range(500):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")

    result = strategy_profile(dimont, pwm, ragged_sequences, create_comparator_config(metric="co", n_permutations=0))

    assert result["orientation"] == "--"
    assert result["offset"] == -2
    assert result["score"] > 0.55


def test_profile_orientation_search_requires_minus_minus_candidate_on_real_profiles():
    """Real profile bundles can prefer -- over the reduced ++ / +- search space."""
    site = "CCAGAGTAAACAG"
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = _encode_sequence(site)

    for _ in range(500):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co", n_permutations=0)
    calibration_sequences = strategy_profile.__globals__["_get_profile_calibration_sequences"](ragged_sequences, cfg)
    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    prepare_profile_bundle = strategy_profile.__globals__["_prepare_profile_bundle_for_scoring"]

    pwm_bundle = resolve_profile_bundle(pwm, ragged_sequences, calibration_sequences, cfg)
    dimont_bundle = resolve_profile_bundle(dimont, ragged_sequences, calibration_sequences, cfg)
    prepared_pwm = prepare_profile_bundle(pwm_bundle, 0.0)
    prepared_dimont = prepare_profile_bundle(dimont_bundle, 0.0)
    orientation_pairs = [
        ("++", PLUS_STRAND, PLUS_STRAND),
        ("--", MINUS_STRAND, MINUS_STRAND),
        ("+-", PLUS_STRAND, MINUS_STRAND),
        ("-+", MINUS_STRAND, PLUS_STRAND),
    ]

    scores, _ = fast_profile_score_orientations(
        prepared_pwm,
        prepared_dimont,
        [(query_strand, target_strand) for _, query_strand, target_strand in orientation_pairs],
        build_profile_score_options(search_range=cfg["search_range"], min_value=0.0, metric=cfg["metric"]),
    )
    score_by_orientation = {
        orientation: float(score)
        for (orientation, _query_strand, _target_strand), score in zip(orientation_pairs, scores, strict=True)
    }

    assert max(score_by_orientation, key=score_by_orientation.get) == "--"
    assert score_by_orientation["--"] > max(score_by_orientation["++"], score_by_orientation["+-"])


def test_strategy_profile_is_symmetric_when_models_peak_on_different_strands():
    """Profile comparison should consider both query and target strands before choosing orientation."""
    site = "CAGTAAACAG"
    dna_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = np.array([dna_to_int[base] for base in site], dtype=np.int8)

    for _ in range(300):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co", n_permutations=0)

    pwm_vs_dimont = strategy_profile(pwm, dimont, ragged_sequences, cfg)
    dimont_vs_pwm = strategy_profile(dimont, pwm, ragged_sequences, cfg)

    assert pwm_vs_dimont["score"] == pytest.approx(dimont_vs_pwm["score"], abs=1e-6)
    assert pwm_vs_dimont["orientation"] == dimont_vs_pwm["orientation"] == "--"
    assert pwm_vs_dimont["offset"] == -dimont_vs_pwm["offset"]
    assert pwm_vs_dimont["score"] > 0.6


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

    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
        ]
    )
    query = make_model("query")
    target = make_model("target")
    cfg = create_comparator_config(metric="co", cache_mode="on", cache_dir=str(tmp_path), n_permutations=0)

    first = strategy_profile(query, target, sequences, cfg)
    assert first["target"] == "target"
    assert len(list(tmp_path.rglob("*.npz"))) == 2

    fresh_query = make_model("query")
    fresh_target = make_model("target")
    original_scan_strands = strategy_profile.__globals__["scan_model_strands"]

    def guarded_scan_strands(model, current_sequences):
        if model.name == "query":
            raise AssertionError("query scan should be served from disk cache")
        if model.name == "target":
            raise AssertionError("target scan should be served from disk cache")
        return original_scan_strands(model, current_sequences)

    monkeypatch.setitem(strategy_profile.__globals__, "scan_model_strands", guarded_scan_strands)
    second = strategy_profile(fresh_query, fresh_target, sequences, cfg)

    assert second["target"] == "target"
    assert second["score"] == pytest.approx(first["score"])
    assert second["orientation"] == first["orientation"]


def test_strategy_profile_permutations_use_batched_surrogate_scoring(monkeypatch):
    """Profile permutations should score plus/minus surrogates in one batched call."""
    scores_1 = _score_batch_from_flat(
        np.array([0.0, 0.8, 1.7, 0.2, 0.0, 1.1, 0.4, 0.0], dtype=np.float32),
        np.array([0, 4, 8], dtype=np.int64),
    )
    scores_2 = _score_batch_from_flat(
        np.array([0.0, 1.0, 1.6, 0.3, 0.0, 0.9, 0.5, 0.0], dtype=np.float32),
        np.array([0, 4, 8], dtype=np.int64),
    )
    model1 = GenericModel(
        type_key="scores",
        name="q",
        representation=None,
        length=0,
        config={"scores_data": scores_1},
    )
    model2 = GenericModel(
        type_key="scores",
        name="t",
        representation=None,
        length=0,
        config={"scores_data": scores_2},
    )
    cfg = create_comparator_config(metric="co", n_permutations=1, numba_threads=1, search_range=2)

    call_sizes = []
    original_orientations = strategy_profile.__globals__["_score_prepared_profile_orientations"]

    def recording_orientations(left_bundle, right_bundle, strand_pairs, options):
        call_sizes.append(len(strand_pairs))
        return original_orientations(left_bundle, right_bundle, strand_pairs, options)

    def fake_run_montecarlo(obs_score_func, surrogate_generator_func, n_permutations, seed, *args):
        value = surrogate_generator_func(np.random.default_rng(0), *args)
        null_scores = np.array([value], dtype=np.float32)
        return null_scores, float(null_scores.mean()), float(null_scores.std())

    monkeypatch.setitem(
        strategy_profile.__globals__,
        "_score_prepared_profile_orientations",
        recording_orientations,
    )
    monkeypatch.setitem(strategy_profile.__globals__, "run_montecarlo", fake_run_montecarlo)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] >= 0.0
    assert 4 in call_sizes
    assert 2 in call_sizes


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
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8)])
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

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_permutations"] == 10
    assert config["seed"] == 99


def test_create_many_config_builds_unified_config():
    """Unified one-vs-many config builder should create comparator config from kwargs."""
    config = create_many_config(
        query="query.meme",
        targets=["target_a.pfm", "target_b.pfm"],
        query_type="pwm",
        target_type="pwm",
        strategy="profile",
        metric="co",
        n_permutations=5,
        seed=11,
    )

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_permutations"] == 5
    assert config["targets"] == ["target_a.pfm", "target_b.pfm"]
    assert config["seed"] == 11


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
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2], dtype=np.int8),
        ]
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


def test_run_one_to_many_matches_pairwise_profile_results():
    """One-vs-many profile API should match repeated pairwise comparisons."""
    query_scores = _score_batch_from_flat(np.array([0.2, 0.5, 0.8], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    target_a_scores = _score_batch_from_flat(
        np.array([0.2, 0.4, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_b_scores = _score_batch_from_flat(
        np.array([0.8, 0.5, 0.2], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel(
        type_key="scores",
        name="query",
        representation=None,
        length=0,
        config={"scores_data": query_scores},
    )
    target_a = GenericModel(
        type_key="scores",
        name="target_a",
        representation=None,
        length=0,
        config={"scores_data": target_a_scores},
    )
    target_b = GenericModel(
        type_key="scores",
        name="target_b",
        representation=None,
        length=0,
        config={"scores_data": target_b_scores},
    )

    config = create_many_config(
        query=query,
        targets=[target_a, target_b],
        strategy="profile",
        metric="co",
        n_permutations=0,
    )
    results = run_one_to_many(config)

    expected_a = run_comparison(
        create_config(model1=query, model2=target_a, strategy="profile", metric="co", n_permutations=0)
    )
    expected_b = run_comparison(
        create_config(model1=query, model2=target_b, strategy="profile", metric="co", n_permutations=0)
    )

    assert [result["target"] for result in results] == ["target_a", "target_b"]
    for result, expected in zip(results, [expected_a, expected_b], strict=False):
        assert result["query"] == expected["query"]
        assert result["target"] == expected["target"]
        assert result["orientation"] == expected["orientation"]
        assert result["offset"] == expected["offset"]
        np.testing.assert_allclose(result["score"], expected["score"])


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
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)])

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


def test_strategy_profile_accepts_promoters_configuration():
    """Profile strategy should accept external promoter calibration sequences."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.3, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    promoters = make_sequence_batch([np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)])
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric="co", promoters=promoters)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["metric"] == "co"
    assert "score" in result


def test_strategy_profile_co_has_no_default_floor():
    """CO should not apply an implicit logFPR floor when min_logfpr is omitted."""
    scores_1 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    result = strategy_profile(model1, model2, None, create_comparator_config(metric="co", n_permutations=0))

    assert result["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("metric", ["co", "dice"])
def test_strategy_profile_handles_all_positions_masked_by_threshold(metric):
    """Pairwise threshold masking should not crash when every aligned pair is filtered out."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.1, 0.2, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric=metric, min_logfpr=10.0, n_permutations=0)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] == pytest.approx(0.0)
    assert result["offset"] == 0
    assert result["orientation"] == "++"


def test_run_comparison_supports_dice_for_profile():
    """Unified API should expose the Dice profile metric."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.5, 1.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.1, 0.5, 0.9], dtype=np.float32), np.array([0, 3], dtype=np.int64))
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
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)])

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


def test_compare_one_to_many_matches_pairwise_motif_results():
    """One-vs-many motif API should match repeated pairwise comparisons."""
    query_representation = np.array(
        [
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.6],
            [0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_a_representation = np.array(
        [
            [0.55, 0.15, 0.3],
            [0.25, 0.65, 0.1],
            [0.1, 0.1, 0.55],
            [0.1, 0.1, 0.05],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_b_representation = np.array(
        [
            [0.1, 0.6, 0.1],
            [0.6, 0.1, 0.1],
            [0.1, 0.1, 0.6],
            [0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = GenericModel(
        type_key="pwm",
        name="query",
        representation=query_representation,
        length=3,
        config={"kmer": 1},
    )
    target_a = GenericModel(
        type_key="pwm",
        name="target_a",
        representation=target_a_representation,
        length=3,
        config={"kmer": 1},
    )
    target_b = GenericModel(
        type_key="pwm",
        name="target_b",
        representation=target_b_representation,
        length=3,
        config={"kmer": 1},
    )

    results = compare_one_to_many(
        query=query,
        targets=[target_a, target_b],
        strategy="motif",
        metric="pcc",
        n_permutations=0,
    )

    expected_a = compare_motifs(query, target_a, strategy="motif", metric="pcc", n_permutations=0)
    expected_b = compare_motifs(query, target_b, strategy="motif", metric="pcc", n_permutations=0)

    assert [result["target"] for result in results] == ["target_a", "target_b"]
    for result, expected in zip(results, [expected_a, expected_b], strict=False):
        assert result["orientation"] == expected["orientation"]
        assert result["offset"] == expected["offset"]
        np.testing.assert_allclose(result["score"], expected["score"])


def test_run_one_to_many_rejects_motali():
    """One-vs-many API should reject unsupported motali execution."""
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
    query = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    target = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})

    config = create_many_config(query=query, targets=[target], strategy="motali")

    with pytest.raises(NotImplementedError, match="motali"):
        run_one_to_many(config)


if __name__ == "__main__":
    pytest.main([__file__])
