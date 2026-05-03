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
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import mimosa.api as api_module
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
from mimosa.cli import map_args_to_comparator_kwargs
from mimosa.comparison import (
    compare,
    create_comparator_config,
    strategy_motif,
    strategy_profile,
)
from mimosa.comparison import registry as comparison_registry
from mimosa.functions import (
    apply_score_log_tail_table,
    batch_all_scores,
    batch_all_scores_strands,
    build_score_log_tail_table,
    calc_co,
    calc_dice,
    cut_prc,
    cut_roc,
    format_params,
    normalize_empirical_log_tail_pair,
    pcm_to_pfm,
    pfm_to_pwm,
    precision_recall_curve,
    roc_curve,
    rowwise_co,
    rowwise_cosine,
    rowwise_dice,
    score_seq,
    scores_to_empirical_log_tail,
    standardized_pauc,
)
from mimosa.io import parse_file_content, read_pfm, read_scores, write_dist
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


def _make_scores_model(name: str, rows: list[list[float]] | list[np.ndarray]) -> GenericModel:
    """Build one GenericModel backed by precomputed score rows."""
    batch = make_score_batch([np.asarray(row, dtype=np.float32) for row in rows])
    return GenericModel(type_key="scores", name=name, representation=None, length=0, config={"scores_data": batch})


def _make_shifted_core_pwm_model(
    name: str,
    core_offset: int,
    core: tuple[int, ...] = (0, 1, 2),
    motif_length: int = 7,
) -> GenericModel:
    """Build one PWM with an informative core placed at a requested matrix offset."""
    pfm = np.full((4, motif_length), 0.25, dtype=np.float32)
    for column_delta, base_index in enumerate(core):
        column_index = core_offset + column_delta
        pfm[:, column_index] = 0.001
        pfm[base_index, column_index] = 0.997
        pfm[:, column_index] /= pfm[:, column_index].sum()

    pwm = pfm_to_pwm(pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    return GenericModel("pwm", name, representation, motif_length, {"kmer": 1, "_source_pfm": pfm})


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
    assert config["n_jobs"] is None

    # Test factory function with custom parameters
    config = create_comparator_config(metric="co", n_permutations=100, seed=42, pfm_top_fraction=0.2)
    assert config["metric"] == "co"
    assert config["n_permutations"] == 100
    assert config["seed"] == 42
    assert config["pfm_top_fraction"] == pytest.approx(0.2)


def test_create_comparator_config_resolves_n_jobs():
    """Explicit n_jobs should drive the effective parallel setting."""
    config = create_comparator_config(n_jobs=4)
    assert config["n_jobs"] == 4


@pytest.mark.parametrize("kwargs", [{"n_jobs": 0}, {"n_jobs": -2}])
def test_create_comparator_config_validates_thread_counts(kwargs):
    """Thread-count settings should accept only positive values or -1."""
    with pytest.raises(ValueError, match="positive or -1"):
        create_comparator_config(**kwargs)


def test_compare_passes_n_jobs_to_strategy(monkeypatch):
    """Execution boundary should pass the validated config to the strategy."""
    observed = []

    def fake_strategy(model1, model2, sequences, cfg):
        observed.append(cfg["n_jobs"])
        return {"score": 1.0}

    monkeypatch.setitem(comparison_registry, "thread_test", fake_strategy)

    result = compare(
        model1=None,
        model2=None,
        strategy="thread_test",
        config=create_comparator_config(n_jobs=1),
    )

    assert result == {"score": 1.0}
    assert observed == [1]


def test_cli_maps_jobs_to_n_jobs_for_profile_config():
    """CLI --jobs should reach comparator config as n_jobs."""
    args = SimpleNamespace(
        mode="profile",
        metric="co",
        permutations=7,
        distortion=0.3,
        jobs=3,
        seed=5,
        search_range=2,
        window_radius=4,
        realign_window=1,
        min_kernel_size=3,
        max_kernel_size=5,
        min_logfpr=None,
        cache="off",
        cache_dir=".mimosa-cache",
    )

    kwargs = map_args_to_comparator_kwargs(args)
    config = create_comparator_config(**kwargs)

    assert kwargs["n_jobs"] == 3
    assert config["n_jobs"] == 3


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


def test_create_comparator_config_rejects_unknown_metric():
    """Comparator config should fail fast for unsupported metric names."""
    with pytest.raises(ValueError, match="metric must be one of"):
        create_comparator_config(metric="wrong")


def test_comparison_registry():
    """Test comparison registry functionality"""
    # Test that we can get registered strategies
    motif_strategy = comparison_registry.get("motif")
    assert motif_strategy is not None

    profile_strategy = comparison_registry.get("profile")
    assert profile_strategy is not None

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
    both_scores = scan_model(model, sequences, "both")
    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")

    assert strand_bundle["values"].shape[0] == 2
    np.testing.assert_array_equal(strand_bundle["lengths"], plus_scores["lengths"])
    np.testing.assert_allclose(strand_bundle["values"][PLUS_STRAND], plus_scores["values"])
    np.testing.assert_allclose(strand_bundle["values"][MINUS_STRAND], minus_scores["values"])
    np.testing.assert_allclose(both_scores["values"], strand_bundle["values"])
    np.testing.assert_array_equal(both_scores["lengths"], strand_bundle["lengths"])


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


def test_calculate_threshold_table_both_uses_combined_strand_sample():
    """strand='both' should fit the threshold table on all + and - predictions."""
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
    sequences = make_sequence_batch([_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")])

    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")
    expected = build_score_log_tail_table(
        np.concatenate((flatten_valid(plus_scores), flatten_valid(minus_scores)))
    ).astype(np.float64)

    table = calculate_threshold_table(model, sequences)

    np.testing.assert_allclose(table, expected)


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


@pytest.mark.parametrize("model_type", ["sitega", "dimont", "slim"])
def test_read_model_rejects_non_model_pickle_payload(tmp_path, model_type):
    """Pickled payloads must always deserialize to GenericModel instances."""
    path = tmp_path / f"{model_type}.pkl"
    joblib.dump({"invalid": True}, path)

    with pytest.raises(TypeError, match="expected GenericModel"):
        read_model(str(path), model_type)


def test_write_dist_rejects_zero_score_range(tmp_path):
    """DIST output must reject degenerate score bounds to avoid inf values."""
    table = np.array([[0.5, 1.0], [0.4, 2.0]], dtype=np.float64)
    path = tmp_path / "invalid.dist"

    with pytest.raises(ValueError, match="max_score must be greater than min_score"):
        write_dist(table, max_score=1.0, min_score=1.0, path=str(path))


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

    best_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5, strand="best")
    both_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5)
    background_sites = get_sites(
        model,
        sequences,
        mode="threshold",
        fpr_threshold=0.5,
        strand="best",
        background_sequences=background,
    )

    assert best_sites["site"].tolist() == ["AC", "AC"]
    assert both_sites["site"].tolist() == ["AC", "AC", "AT", "AT"]
    assert both_sites["strand"].tolist() == ["+", "+", "+", "-"]
    assert background_sites["site"].tolist() == ["AC", "AC", "AT"]


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

    default_pfm = get_pfm(model, sequences, mode="threshold", fpr_threshold=0.5, strand="best", pseudocount=0.25)
    background_pfm = get_pfm(
        model,
        sequences,
        mode="threshold",
        fpr_threshold=0.5,
        strand="best",
        background_sequences=background,
        pseudocount=0.25,
    )

    expected_default_pcm = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    expected_background_pcm = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

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


def test_rowwise_cosine_is_averaged_per_window():
    """Profile cosine should be the mean of per-window cosine values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [1.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_cosine(windows_1, windows_2),
        np.array([1.0, 0.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("cosine", windows_1, windows_2) == pytest.approx(0.5)


def test_rowwise_co_is_averaged_per_window():
    """Profile rowwise CO should be the mean of per-window CO values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_co(windows_1, windows_2),
        np.array([1.0, 0.5, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("co_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_rowwise_dice_is_averaged_per_window():
    """Profile rowwise Dice should be the mean of per-window Dice values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_dice(windows_1, windows_2),
        np.array([1.0, 0.5, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("dice_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_overlap_profile_metrics_match_reference_formulas():
    """CO and Dice should use the weighted overlap formulas."""
    windows_1 = np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[2.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    intersection = np.minimum(windows_1, windows_2).sum()

    assert calc_co(windows_1, windows_2) == pytest.approx(intersection / min(windows_1.sum(), windows_2.sum()))
    assert calc_dice(windows_1, windows_2) == pytest.approx((2.0 * intersection) / (windows_1.sum() + windows_2.sum()))


@pytest.mark.parametrize(("metric", "expected"), [("co", 0.5), ("dice", 0.5)])
def test_window_metrics_score_only_selected_windows(metric, expected):
    """CO and Dice should be computed on the selected window collection only."""
    windows_1 = np.array([[2.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 1.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    assert score_window_collection(metric, windows_1, windows_2) == pytest.approx(expected)


def test_dice_rowwise_differs_from_global_dice_when_window_weights_differ():
    """Rowwise Dice should weight windows uniformly, unlike global Dice."""
    windows_1 = np.array([[100.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    windows_2 = np.array([[100.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    assert calc_dice(windows_1, windows_2) == pytest.approx(202.0 / 204.0)
    assert score_window_collection("dice_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_threshold_profile_selection_uses_or_logic():
    """Threshold mode should keep windows when either motif contributes the anchor."""
    compute_alignment = strategy_profile.__globals__["_compute_shifted_window_alignment"]
    collect_anchor_sites = strategy_profile.__globals__["_collect_anchor_sites"]
    lengths = np.array([3], dtype=np.int32)
    scores_1 = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    scores_2 = np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
    query_anchors = collect_anchor_sites(scores_1, lengths, 1.0)
    target_anchors = collect_anchor_sites(scores_2, lengths, 1.0)

    result = compute_alignment(
        scores_1,
        lengths,
        scores_2,
        lengths,
        0,
        np.array([0], dtype=np.int32),
        0,
        0,
        query_anchors,
        target_anchors,
        0,
        "co",
    )

    assert result["n_sites"] == 1
    assert result["score"] == pytest.approx(1.0)


def test_model2_threshold_anchors_are_realigned_on_model1():
    """Anchors from the second motif should be recentered on the best nearby site of the first motif."""
    collect_model2_candidates = strategy_profile.__globals__["_collect_model2_window_candidates"]
    collect_anchor_sites = strategy_profile.__globals__["_collect_anchor_sites"]
    lengths = np.array([5], dtype=np.int32)
    scores_1 = np.array([[0.0, 1.0, 3.0, 0.0, 0.0]], dtype=np.float32)
    scores_2 = np.array([[0.0, 4.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    anchor_rows, anchor_pos2 = collect_anchor_sites(scores_2, lengths, 1.0)

    rows, pos1, pos2 = collect_model2_candidates(
        scores_1,
        lengths,
        lengths,
        anchor_rows,
        anchor_pos2,
        0,
        0,
        0,
        1,
    )

    np.testing.assert_array_equal(rows, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(pos1, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(pos2, np.array([2], dtype=np.int32))


def test_profile_orientation_search_keeps_all_four_candidates():
    """Window-based profile search should still consider all four strand combinations."""
    score_profile_candidates = strategy_profile.__globals__["_score_profile_candidates"]
    query_plus = make_score_batch([np.array([1.0, 0.0, 0.0], dtype=np.float32)])
    query_minus = make_score_batch([np.array([0.0, 1.0, 0.0], dtype=np.float32)])
    target_plus = make_score_batch([np.array([0.0, 1.0, 0.0], dtype=np.float32)])
    target_minus = make_score_batch([np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    query_bundle = make_strand_bundle(query_plus, query_minus)
    target_bundle = make_strand_bundle(target_plus, target_minus)
    cfg = create_comparator_config(metric="co", n_permutations=0, search_range=0, window_radius=0, realign_window=0)

    best = strategy_profile.__globals__["_select_best_orientation"](
        score_profile_candidates(
            query_bundle,
            target_bundle,
            strategy_profile.__globals__["PROFILE_ORIENTATION_PAIRS"],
            cfg,
        )
    )

    assert best["orientation"] == "-+"
    assert best["score"] == pytest.approx(1.0)


def test_strategy_functions_exist():
    """Test that all strategy functions are properly defined"""
    # Test that strategy functions exist and are callable
    assert callable(strategy_motif)
    assert callable(strategy_profile)


def test_strategy_profile_uses_motif_offset_convention_for_score_tracks():
    """Profile strategy should report target position minus query position."""
    model1 = _make_scores_model("s1", [[0.0, 0.0, 9.0, 0.0, 0.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 0.0, 0.0, 0.0, 9.0, 0.0]])
    cfg = create_comparator_config(metric="co", search_range=4, window_radius=0, min_logfpr=0.1, n_permutations=0)

    forward = strategy_profile(model1, model2, None, cfg)
    reverse = strategy_profile(model2, model1, None, cfg)

    assert forward["orientation"] == "++"
    assert reverse["orientation"] == "++"
    assert forward["score"] == pytest.approx(1.0)
    assert reverse["score"] == pytest.approx(1.0)
    assert forward["offset"] == 2
    assert reverse["offset"] == -2
    assert forward["n_sites"] == 1
    assert reverse["n_sites"] == 1


def test_strategy_profile_offset_matches_motif_for_shifted_pwm_core():
    """Profile and motif strategies should use the same offset sign and value."""
    query = _make_shifted_core_pwm_model("query", core_offset=1)
    target = _make_shifted_core_pwm_model("target", core_offset=3)
    sequences = []
    embedded_site = _encode_sequence("ACG")
    for _ in range(64):
        sequence = np.full(60, _DNA_TO_INT["T"], dtype=np.int8)
        sequence[25 : 25 + embedded_site.size] = embedded_site
        sequences.append(sequence)
    sequence_batch = make_sequence_batch(sequences)
    motif_cfg = create_comparator_config(metric="cosine", n_permutations=0, pfm_mode=False)
    profile_cfg = create_comparator_config(
        metric="co",
        n_permutations=0,
        search_range=5,
        window_radius=0,
        realign_window=0,
    )

    motif_result = strategy_motif(query, target, sequence_batch, motif_cfg)
    profile_result = strategy_profile(query, target, sequence_batch, profile_cfg)
    motif_reverse = strategy_motif(target, query, sequence_batch, motif_cfg)
    profile_reverse = strategy_profile(target, query, sequence_batch, profile_cfg)

    assert motif_result["orientation"] == profile_result["orientation"] == "++"
    assert motif_result["offset"] == profile_result["offset"] == -2
    assert profile_result["score"] == pytest.approx(1.0)
    assert motif_reverse["orientation"] == profile_reverse["orientation"] == "++"
    assert motif_reverse["offset"] == profile_reverse["offset"] == 2
    assert profile_reverse["score"] == pytest.approx(1.0)


def test_strategy_profile_offset_matches_motif_for_reverse_complement_pwm_core():
    """Offset convention should also match for reverse-complement motif orientation."""
    core = (0, 1, 2, 3, 1, 0)
    target_core = tuple(int(base) for base in _reverse_complement_encoded(np.asarray(core, dtype=np.int8)))
    query = _make_shifted_core_pwm_model("query", core_offset=2, core=core, motif_length=12)
    target = _make_shifted_core_pwm_model("target", core_offset=3, core=target_core, motif_length=12)
    sequences = []
    embedded_site = np.asarray(core, dtype=np.int8)
    for _ in range(64):
        sequence = np.full(100, _DNA_TO_INT["A"], dtype=np.int8)
        sequence[45 : 45 + embedded_site.size] = embedded_site
        sequences.append(sequence)
    sequence_batch = make_sequence_batch(sequences)
    motif_cfg = create_comparator_config(metric="cosine", n_permutations=0, pfm_mode=False)
    profile_cfg = create_comparator_config(
        metric="co",
        n_permutations=0,
        search_range=8,
        window_radius=5,
        realign_window=0,
    )

    motif_result = strategy_motif(query, target, sequence_batch, motif_cfg)
    profile_result = strategy_profile(query, target, sequence_batch, profile_cfg)

    assert motif_result["orientation"] == profile_result["orientation"] == "+-"
    assert motif_result["offset"] == profile_result["offset"] == -1
    assert profile_result["score"] == pytest.approx(1.0)


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


def test_strategy_profile_uses_background_for_empirical_calibration():
    """Profile normalization should use background scans when explicit calibration sequences are provided."""
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
    background = make_sequence_batch([_encode_sequence("AAAAAA"), _encode_sequence("CCCCCC")])
    cfg = create_comparator_config(metric="co", background=background, n_permutations=0)

    plus_scores = scan_model(model, sequences, "+")
    background_plus = scan_model(model, background, "+")
    background_minus = scan_model(model, background, "-")
    table = build_score_log_tail_table(
        np.concatenate((flatten_valid(background_plus), flatten_valid(background_minus)))
    )
    expected_plus = apply_score_log_tail_table(plus_scores, table)

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(model, sequences, background, cfg)

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
    assert abs(result["offset"]) == 2
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
    calibration_sequences = strategy_profile.__globals__["_get_profile_background_sequences"](ragged_sequences, cfg)
    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    score_profile_candidates = strategy_profile.__globals__["_score_profile_candidates"]
    orientation_pairs = strategy_profile.__globals__["PROFILE_ORIENTATION_PAIRS"]

    pwm_bundle = resolve_profile_bundle(pwm, ragged_sequences, calibration_sequences, cfg)
    dimont_bundle = resolve_profile_bundle(dimont, ragged_sequences, calibration_sequences, cfg)
    candidates = score_profile_candidates(pwm_bundle, dimont_bundle, orientation_pairs, cfg)
    score_by_orientation = {candidate["orientation"]: float(candidate["score"]) for candidate in candidates}

    assert max(score_by_orientation, key=score_by_orientation.get) == "--"
    assert score_by_orientation["--"] > max(score_by_orientation["++"], score_by_orientation["+-"])


def test_strategy_profile_is_symmetric_when_models_peak_on_different_strands():
    """Profile comparison should stay numerically close when query and target are swapped."""
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

    assert pwm_vs_dimont["score"] == pytest.approx(dimont_vs_pwm["score"], abs=1e-3)
    assert pwm_vs_dimont["orientation"] == dimont_vs_pwm["orientation"] == "--"
    assert pwm_vs_dimont["offset"] == -dimont_vs_pwm["offset"]
    assert pwm_vs_dimont["score"] > 0.6


def test_strategy_runtime_cache_keys_use_model_content_not_name():
    """Runtime cache should not mix distinct models sharing the same name."""

    def build_model(seed: int, name: str) -> GenericModel:
        rng = np.random.default_rng(seed)
        pfm = rng.random((4, 6), dtype=np.float32)
        pfm /= pfm.sum(axis=0, keepdims=True)
        pwm = pfm_to_pwm(pfm)
        representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
        return GenericModel("pwm", name, representation, 6, {"kmer": 1, "_source_pfm": pfm})

    sequences = make_sequence_batch(
        [np.random.default_rng(3).integers(0, 4, size=200, dtype=np.int8) for _ in range(20)]
    )
    cfg_profile = create_comparator_config(metric="co", n_permutations=0)
    cfg_motif = create_comparator_config(metric="cosine", n_permutations=0, pfm_mode=False)

    model_a = build_model(1, "a")
    model_b = build_model(2, "b")
    model_same_1 = build_model(1, "same")
    model_same_2 = build_model(2, "same")

    profile_named = strategy_profile(model_a, model_b, sequences, cfg_profile)
    profile_same_name = strategy_profile(model_same_1, model_same_2, sequences, cfg_profile)
    motif_named = strategy_motif(model_a, model_b, sequences, cfg_motif)
    motif_same_name = strategy_motif(model_same_1, model_same_2, sequences, cfg_motif)

    assert profile_same_name["score"] == pytest.approx(profile_named["score"])
    assert motif_same_name["score"] == pytest.approx(motif_named["score"])


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
    """Profile permutations should reuse the new scorer with four observed and two surrogate orientations."""
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
    cfg = create_comparator_config(metric="co", n_permutations=1, n_jobs=1, search_range=2)

    call_sizes = []
    original_candidates = strategy_profile.__globals__["_score_profile_candidates"]

    def recording_candidates(left_bundle, right_bundle, pair_specs, cfg):
        call_sizes.append(len(pair_specs))
        return original_candidates(left_bundle, right_bundle, pair_specs, cfg)

    def fake_run_montecarlo(obs_score_func, surrogate_generator_func, n_permutations, seed, *args):
        value = surrogate_generator_func(np.random.default_rng(0), *args)
        null_scores = np.array([value], dtype=np.float32)
        return null_scores, float(null_scores.mean()), float(null_scores.std())

    monkeypatch.setitem(
        strategy_profile.__globals__,
        "_score_profile_candidates",
        recording_candidates,
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
        n_jobs=2,
        seed=99,
    )

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_permutations"] == 10
    assert config["comparator"]["n_jobs"] == 2
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
        n_jobs=3,
        seed=11,
    )

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_permutations"] == 5
    assert config["comparator"]["n_jobs"] == 3
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


def test_run_one_to_many_passes_targets_lazily(monkeypatch):
    """One-vs-many executor should receive a lazy target iterable instead of a materialized list."""
    query_scores = _score_batch_from_flat(
        np.array([0.2, 0.5, 0.8], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_scores = _score_batch_from_flat(
        np.array([0.2, 0.4, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel(
        type_key="scores",
        name="query",
        representation=None,
        length=0,
        config={"scores_data": query_scores},
    )
    target = GenericModel(
        type_key="scores",
        name="target",
        representation=None,
        length=0,
        config={"scores_data": target_scores},
    )
    config = create_many_config(query=query, targets=[target], strategy="profile", metric="co", n_permutations=0)
    observed = {}

    def fake_compare_one_to_many_models(query_model, target_models, strategy, config, sequences=None, background=None):
        observed["is_list"] = isinstance(target_models, list)
        materialized = list(target_models)
        observed["count"] = len(materialized)
        return [{"query": query_model.name, "target": materialized[0].name, "score": 1.0}]

    monkeypatch.setattr(api_module, "compare_one_to_many_models", fake_compare_one_to_many_models)

    results = run_one_to_many(config)

    assert observed == {"is_list": False, "count": 1}
    assert results == [{"query": "query", "target": "target", "score": 1.0}]


def test_run_one_to_many_preserves_generator_targets():
    """One-vs-many API should not lose targets when config["targets"] is a generator."""
    query_scores = _score_batch_from_flat(
        np.array([0.2, 0.5, 0.8], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_a_scores = _score_batch_from_flat(
        np.array([0.1, 0.3, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_b_scores = _score_batch_from_flat(
        np.array([0.6, 0.2, 0.4], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel("scores", "query", None, 0, {"scores_data": query_scores})
    target_a = GenericModel("scores", "target_a", None, 0, {"scores_data": target_a_scores})
    target_b = GenericModel("scores", "target_b", None, 0, {"scores_data": target_b_scores})

    config = {
        "query": query,
        "targets": (target for target in (target_a, target_b)),
        "query_type": None,
        "target_type": None,
        "strategy": "profile",
        "sequences": None,
        "background": None,
        "num_sequences": 1000,
        "seq_length": 200,
        "seed": 127,
        "comparator": create_comparator_config(metric="co", n_permutations=0),
        "query_kwargs": {},
        "target_kwargs": {},
    }
    results = run_one_to_many(config)  # type: ignore[arg-type]

    assert [item["target"] for item in results] == ["target_a", "target_b"]


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

    with pytest.raises(ValueError, match="metric"):
        config = create_config(
            model1=model1,
            model2=model2,
            strategy="profile",
            sequences=sequences,
            metric=metric,
            n_permutations=0,
            seed=7,
        )
        run_comparison(config)


def test_strategy_profile_accepts_background_configuration():
    """Profile strategy should accept external background calibration sequences."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.3, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    background = make_sequence_batch([np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)])
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric="co", background=background)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["metric"] == "co"
    assert "score" in result


def test_strategy_profile_co_has_no_default_floor():
    """CO should not apply an implicit logFPR floor when min_logfpr is omitted."""
    scores_1 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    result = strategy_profile(
        model1,
        model2,
        None,
        create_comparator_config(metric="co", window_radius=0, n_permutations=0),
    )

    assert result["score"] == pytest.approx(1.0)


def test_strategy_profile_zero_min_logfpr_uses_best_anchor_mode():
    """min_logfpr=0 should behave like an omitted threshold."""
    scores_1 = _score_batch_from_flat(
        np.array([0.1, 0.9, 0.8, 0.7], dtype=np.float32),
        np.array([0, 4], dtype=np.int64),
    )
    scores_2 = _score_batch_from_flat(
        np.array([0.2, 0.95, 0.1, 0.1], dtype=np.float32),
        np.array([0, 4], dtype=np.int64),
    )
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    kwargs = {"metric": "co", "window_radius": 0, "search_range": 0, "n_permutations": 0}
    omitted = strategy_profile(model1, model2, None, create_comparator_config(**kwargs, min_logfpr=None))
    zero = strategy_profile(model1, model2, None, create_comparator_config(**kwargs, min_logfpr=0))

    assert zero["score"] == pytest.approx(omitted["score"])
    assert zero["n_sites"] == omitted["n_sites"] == 1


@pytest.mark.parametrize("metric", ["co", "co_rowwise", "dice", "dice_rowwise", "cosine"])
def test_strategy_profile_handles_all_positions_masked_by_threshold(metric):
    """Threshold site selection should not crash when no sites survive the cutoff."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.1, 0.2, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric=metric, min_logfpr=10.0, n_permutations=0)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] == pytest.approx(0.0)
    assert result["offset"] == 0
    assert result["orientation"] == "++"
    assert result["n_sites"] == 0


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


def test_run_comparison_supports_dice_rowwise_for_profile():
    """Unified API should expose the window-averaged rowwise Dice profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])

    config = create_config(
        model1=model1,
        model2=model2,
        strategy="profile",
        metric="dice_rowwise",
        n_permutations=0,
        seed=7,
        window_radius=1,
    )
    result = run_comparison(config)

    assert result["metric"] == "dice_rowwise"
    assert 0.0 <= result["score"] <= 1.0


def test_run_comparison_supports_cosine_for_profile():
    """Unified API should expose the window-averaged cosine profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [0.0, 1.0, -1.0]])

    config = create_config(
        model1=model1,
        model2=model2,
        strategy="profile",
        metric="cosine",
        n_permutations=0,
        seed=7,
        window_radius=1,
    )
    result = run_comparison(config)

    assert result["metric"] == "cosine"
    assert 0.0 <= result["score"] <= 1.0


def test_run_comparison_supports_co_rowwise_for_profile():
    """Unified API should expose the window-averaged rowwise CO profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])

    config = create_config(
        model1=model1,
        model2=model2,
        strategy="profile",
        metric="co_rowwise",
        n_permutations=0,
        seed=7,
        window_radius=1,
    )
    result = run_comparison(config)

    assert result["metric"] == "co_rowwise"
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


if __name__ == "__main__":
    pytest.main([__file__])
