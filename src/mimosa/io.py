from __future__ import annotations

import copy
import itertools
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from mimosa.ragged import RaggedData

_JSTACS_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")


def read_fasta(path: str | Path) -> RaggedData:
    """Read a FASTA file and return integer-encoded sequences."""

    trans_table = bytearray([4] * 256)
    for char, code in zip(b"ACGTacgt", [0, 1, 2, 3] * 2, strict=False):
        trans_table[char] = code

    sequences: List[np.ndarray] = []

    with open(path, "r") as handle:
        current_seq_bytes = bytearray()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_bytes:
                    encoded = np.frombuffer(current_seq_bytes.translate(trans_table), dtype=np.int8).copy()
                    sequences.append(encoded)
                    current_seq_bytes.clear()
            else:
                current_seq_bytes.extend(line.encode("ascii", errors="ignore"))

        if current_seq_bytes:
            encoded = np.frombuffer(current_seq_bytes.translate(trans_table), dtype=np.int8).copy()
            sequences.append(encoded)

    if not sequences:
        return RaggedData(np.empty(0, dtype=np.int8), np.zeros(1, dtype=np.int64))

    n = len(sequences)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(seq)

    total_data = np.empty(offsets[-1], dtype=np.int8)
    for i, seq in enumerate(sequences):
        total_data[offsets[i] : offsets[i + 1]] = seq

    return RaggedData(total_data, offsets)


def read_scores(path: str | Path) -> RaggedData:
    """Read FASTA-like numerical score profiles into RaggedData."""

    profiles: List[np.ndarray] = []
    current_values: List[float] = []

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_values:
                    profiles.append(np.asarray(current_values, dtype=np.float32))
                    current_values = []
                continue

            try:
                current_values.extend(float(token) for token in line.replace(",", " ").split())
            except ValueError as exc:
                raise ValueError(f"Invalid score value in {path}: {line}") from exc

    if current_values:
        profiles.append(np.asarray(current_values, dtype=np.float32))

    if not profiles:
        return RaggedData(np.empty(0, dtype=np.float32), np.zeros(1, dtype=np.int64))

    n = len(profiles)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, profile in enumerate(profiles):
        offsets[i + 1] = offsets[i] + len(profile)

    total_data = np.empty(offsets[-1], dtype=np.float32)
    for i, profile in enumerate(profiles):
        total_data[offsets[i] : offsets[i + 1]] = profile

    return RaggedData(total_data, offsets)


def write_fasta(sequences: Union[RaggedData, Iterable[np.ndarray]], path: str) -> None:
    """Write integer-encoded sequences to a FASTA file."""

    decoder = np.array(["A", "C", "G", "T", "N"], dtype="U1")

    with open(path, "w") as out:
        if isinstance(sequences, RaggedData):
            for i in range(sequences.num_sequences):
                seq_int = sequences.get_slice(i)
                safe_seq = np.clip(seq_int, 0, 4)
                chars = decoder[safe_seq]
                seq_str = "".join(chars)
                out.write(f">{i}\n")
                out.write(f"{seq_str}\n")
        else:
            for idx, seq_int in enumerate(sequences):
                safe_seq = np.clip(seq_int, 0, 4)
                chars = decoder[safe_seq]
                seq_str = "".join(chars)
                out.write(f">{idx}\n")
                out.write(f"{seq_str}\n")


def read_meme(path: str, index: int = 0) -> Tuple[np.ndarray, Tuple[str, int], int]:
    """Read a specific motif from a MEME formatted file and return total count."""
    target_motif: np.ndarray | None = None
    target_info: Tuple[str, int] | None = None
    motif_count = 0

    with open(path) as handle:
        line = handle.readline()
        while line:
            if line.startswith("MOTIF"):
                is_target = motif_count == index
                motif_count += 1

                parts = line.strip().split()
                name = parts[1]

                header_line = handle.readline()
                header = header_line.strip().split()

                try:
                    length_idx = header.index("w=") + 1
                    length = int(header[length_idx])
                except (ValueError, IndexError):
                    length = 0

                if is_target:
                    matrix = []
                    for _ in range(length):
                        row_line = handle.readline()
                        row = row_line.strip().split()
                        if not row:
                            continue
                        matrix.append(list(map(float, row)))

                    target_motif = np.array(matrix, dtype=np.float32).T
                    target_info = (name, length)
                else:
                    for _ in range(length):
                        handle.readline()

            line = handle.readline()

    if target_motif is None:
        if motif_count == 0:
            raise ValueError(f"No motifs found in {path}")
        else:
            raise IndexError(f"Motif index {index} out of range. File contains {motif_count} motifs.")

    assert target_info is not None

    return target_motif, target_info, motif_count


def read_sitega(path: str) -> tuple[np.ndarray, str, int, float, float]:
    """Parse SiteGA output file and return the motif matrix with metadata."""
    converter = {"A": 0, "C": 1, "G": 2, "T": 3}
    with open(path) as file:
        name = file.readline().strip()
        _number_of_lpd = int(file.readline().strip().split()[0])
        length = int(file.readline().strip().split()[0])
        minimum = float(file.readline().strip().split()[0])
        maximum = float(file.readline().strip().split()[0])
        sitega = np.zeros((5, 5, length), dtype=np.float32)
        for line in file:
            start, stop, value, _, dinucleotide = line.strip().split()
            dinucleotide = dinucleotide.upper()
            nuc_1, nuc_2 = converter[dinucleotide[0]], converter[dinucleotide[1]]
            number_of_positions = int(stop) - int(start) + 1
            for index in range(int(start), int(stop) + 1):
                sitega[nuc_1][nuc_2][index] += float(value) / number_of_positions
    return np.array(sitega, dtype=np.float32), name, length, minimum, maximum


def parse_file_content(filepath: str) -> tuple[dict[int, list[np.ndarray]], int, int]:
    """Parse BaMM file content, ignoring comments starting with '#'."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    with open(filepath, "r") as f:
        raw_text = f.read()

    raw_blocks = raw_text.strip().split("\n\n")
    clean_blocks_data = []

    for raw_block in raw_blocks:
        lines = raw_block.strip().split("\n")

        valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        if not valid_lines:
            continue

        block_arrays = []
        for line in valid_lines:
            parts = line.split()
            if not parts:
                continue
            arr = np.array([float(x) for x in parts], dtype=np.float32)
            block_arrays.append(arr)

        clean_blocks_data.append(block_arrays)

    if not clean_blocks_data:
        raise ValueError(f"No valid data found in {filepath}")

    num_positions = len(clean_blocks_data)
    max_order = len(clean_blocks_data[0]) - 1

    data_by_order = {}
    for k in range(max_order + 1):
        data_by_order[k] = []
        for pos_idx in range(num_positions):
            if len(clean_blocks_data[pos_idx]) <= k:
                raise ValueError(f"Inconsistent orders in block {pos_idx}")
            data_by_order[k].append(clean_blocks_data[pos_idx][k])

    return data_by_order, max_order, num_positions


def read_bamm(motif_path: str, target_order: int) -> np.ndarray:
    """Read a BaMM motif and convert it to log-odds against a uniform background."""

    motif_raw, max_order_file, motif_length = parse_file_content(motif_path)
    if target_order > max_order_file:
        target_order = max_order_file
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Target order {target_order} exceeds file max order {max_order_file}, target order set as max order"
        )

    acgt_slices = []

    for pos in range(motif_length):
        current_k = min(pos, target_order)

        p_motif = motif_raw[current_k][pos]
        uniform_bg = np.full_like(p_motif, 0.25 ** (current_k + 1), dtype=np.float32)

        epsilon = 1e-10
        log_odds = np.log((p_motif + epsilon) / (uniform_bg + epsilon))

        shape_k = [4] * (current_k + 1)
        tensor_k = log_odds.reshape(shape_k)

        if current_k < target_order:
            missing_dims = target_order - current_k
            expand_shape = [1] * missing_dims + shape_k
            tensor_expanded = tensor_k.reshape(expand_shape)
            target_shape_4 = [4] * (target_order + 1)
            tensor_final = np.broadcast_to(tensor_expanded, target_shape_4).copy()
        else:
            tensor_final = tensor_k

        acgt_slices.append(tensor_final)

    acgt_tensor = np.stack(acgt_slices, axis=-1)

    reduce_axes = tuple(range(target_order + 1))
    min_scores_per_pos = np.min(acgt_tensor, axis=reduce_axes)

    new_shape = [5] * (target_order + 1) + [motif_length]

    final_tensor = np.ones(new_shape, dtype=np.float32) * min_scores_per_pos

    slice_objs = [slice(0, 4)] * (target_order + 1) + [slice(None)]
    final_tensor[tuple(slice_objs)] = acgt_tensor

    return np.array(final_tensor, dtype=np.float32)


def _xml_numeric_value(elem: Optional[ET.Element]) -> Optional[float]:
    """Extract the last numeric scalar from a Jstacs XML element."""
    if elem is None:
        return None

    texts = [text.strip() for text in elem.itertext() if text and text.strip()]
    for text in reversed(texts):
        if _JSTACS_NUMERIC_RE.fullmatch(text):
            return float(text)

    return None


def _xml_array(elem: ET.Element):
    """Recursively convert Jstacs <pos>-based arrays to Python lists."""
    pos_children = [child for child in elem if child.tag == "pos"]
    if not pos_children:
        return _xml_numeric_value(elem)

    return [_xml_array(child) for child in pos_children]


def _log_normalize(values: np.ndarray) -> np.ndarray:
    """Convert unconstrained log-parameters to normalized log-probabilities."""
    return values - _logsumexp(values)


def _logsumexp(values: np.ndarray) -> float:
    """Compute a stable float64 log-sum-exp."""
    shifted = values - np.max(values)
    return float(np.max(values) + np.log(np.sum(np.exp(shifted))))


def _fill_n_axis_with_min(arr: np.ndarray, axis: int) -> None:
    """Assign the N state on one axis to the minimum over concrete nucleotides."""
    index = [slice(None)] * arr.ndim
    index[axis] = 4
    arr[tuple(index)] = np.min(np.take(arr, [0, 1, 2, 3], axis=axis), axis=axis)


def _build_position_tensor(
    context_log_probs: Dict[Tuple[int, ...], np.ndarray],
    context_axes: List[int],
    span: int,
) -> np.ndarray:
    """Build one dense 5-ary context tensor for a single motif position."""
    temp = np.full((5,) * span + (4,), np.inf, dtype=np.float64)

    for context_values, log_probs in context_log_probs.items():
        assignment = {axis: value for axis, value in zip(context_axes, context_values, strict=False)}
        index = []
        for axis in range(span):
            index.append(assignment.get(axis, slice(None)))
        index.append(slice(None))
        temp[tuple(index)] = log_probs

    for axis in context_axes:
        _fill_n_axis_with_min(temp, axis)

    position_tensor = np.empty((5,) * (span + 1), dtype=np.float64)
    position_tensor[..., :4] = temp
    position_tensor[..., 4] = np.min(temp, axis=-1)
    return position_tensor.astype(np.float32, copy=False)


def _context_value(full_context: Tuple[int, ...], span: int, position: int, absolute_position: int) -> int:
    """Map one absolute parent position to the corresponding dense context axis."""
    if absolute_position < 0:
        raise ValueError(
            f"Model references position {absolute_position} before the motif start at position {position}"
        )

    axis = absolute_position - (position - span)
    if axis < 0 or axis >= span:
        raise ValueError(
            f"Context position {absolute_position} at motif position {position} does not fit span {span}"
        )

    return full_context[axis]


def _iter_full_contexts(span: int) -> Iterable[Tuple[int, ...]]:
    """Iterate over all concrete A/C/G/T contexts of the requested span."""
    return itertools.product(range(4), repeat=span)


def read_slim(path: str) -> tuple[np.ndarray, int, int]:
    """Read a Jstacs Slim XML model into a dense log-odds tensor."""
    root = ET.parse(path).getroot()
    slim = root.find(".//SLIM")
    if slim is None:
        raise ValueError(f"Could not find SLIM model in {path}")

    length = int(_xml_numeric_value(slim.find("length")))
    _distance = int(_xml_numeric_value(slim.find("distance")))
    component_params = _xml_array(slim.find("componentMixtureParameters"))
    ancestor_params = _xml_array(slim.find("ancestorMixtureParameters"))
    dependency_params = _xml_array(slim.find("dependencyParameters"))
    uniform_log_odds_offset = np.log(4.0)
    alphabet_size = len(dependency_params[0][0][0])

    component_log_probs = [
        _log_normalize(np.asarray(component_params[position], dtype=np.float64)) for position in range(length)
    ]
    ancestor_log_probs = [
        [_log_normalize(np.asarray(component, dtype=np.float64)) for component in ancestor_params[position]]
        for position in range(length)
    ]
    dependency_log_probs = [
        [
            np.vstack(
                [_log_normalize(np.asarray(row, dtype=np.float64)) for row in dependency_params[position][component]]
            )
            for component in range(len(dependency_params[position]))
        ]
        for position in range(length)
    ]

    span = 0
    for position in range(length):
        for component_index in range(1, len(component_params[position])):
            ancestor_count = len(ancestor_params[position][component_index])
            if ancestor_count <= 0:
                raise ValueError(f"Malformed SLIM model in {path}: empty ancestor mixture at position {position}")
            span = max(span, component_index + ancestor_count - 1)

    tensor = np.empty((5,) * (span + 1) + (length,), dtype=np.float32)
    context_axes = list(range(span))
    full_contexts = list(_iter_full_contexts(span))

    for position in range(length):
        context_log_probs = {}

        for full_context in full_contexts:
            symbol_log_probs = np.empty(4, dtype=np.float64)

            for symbol in range(4):
                local_scores = np.empty(len(component_log_probs[position]), dtype=np.float64)
                local_scores[0] = component_log_probs[position][0] + dependency_log_probs[position][0][0, symbol]

                for component_index in range(1, len(component_log_probs[position])):
                    ancestor_count = len(ancestor_log_probs[position][component_index])
                    context_index = 0

                    for current_order in range(1, component_index):
                        parent_position = position - current_order
                        context_index = (
                            context_index * alphabet_size
                            + _context_value(full_context, span, position, parent_position)
                        )

                    ancestor_scores = np.empty(ancestor_count, dtype=np.float64)
                    total_contexts = dependency_log_probs[position][component_index].shape[0]
                    width = dependency_log_probs[position][component_index].shape[1]

                    for ancestor_index in range(ancestor_count):
                        parent_position = position - component_index - ancestor_index
                        context_index = (
                            context_index * width
                            + _context_value(full_context, span, position, parent_position)
                        ) % total_contexts
                        ancestor_scores[ancestor_index] = (
                            ancestor_log_probs[position][component_index][ancestor_index]
                            + dependency_log_probs[position][component_index][context_index, symbol]
                        )

                    local_scores[component_index] = (
                        component_log_probs[position][component_index] + _logsumexp(ancestor_scores)
                    )

                symbol_log_probs[symbol] = _logsumexp(local_scores) + uniform_log_odds_offset

            context_log_probs[full_context] = symbol_log_probs

        tensor[..., position] = _build_position_tensor(context_log_probs, context_axes, span)

    return tensor, length, span


def _parse_dimont_treeelement(elem: ET.Element) -> dict:
    """Parse one recursive MarkovModelDiffSM tree element."""
    node = {"context_pos": int(_xml_numeric_value(elem.find("contextPos")))}
    pars = elem.find("pars")
    pars_pos = [child for child in pars if child.tag == "pos"] if pars is not None else []

    if pars_pos:
        node["scores"] = np.asarray(
            [_xml_numeric_value(pos.find("parameter/value")) for pos in pars_pos],
            dtype=np.float64,
        )
        node["log_z"] = np.asarray(
            [(_xml_numeric_value(pos.find("parameter/z")) or 0.0) for pos in pars_pos],
            dtype=np.float64,
        )
        return node

    children_elem = elem.find("children")
    if children_elem is None:
        raise ValueError("Malformed Dimont tree: expected children or parameters")

    node["children"] = [_parse_dimont_treeelement(pos.find("treeElement")) for pos in children_elem if pos.tag == "pos"]
    return node


def read_dimont(path: str) -> tuple[np.ndarray, int, int]:
    """Read a Jstacs Dimont XML model into a dense log-odds tensor."""
    root = ET.parse(path).getroot()
    model = root.find(".//ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM")
    if model is None:
        raise ValueError(f"Could not find Dimont motif model in {path}")
    uniform_log_odds_offset = np.log(4.0)

    trees = model.find("bayesianNetworkSF/trees")
    if trees is None:
        raise ValueError(f"Malformed Dimont model in {path}: missing trees")

    tree_elements = [pos for pos in trees if pos.tag == "pos"]
    length = len(tree_elements)
    context_positions: List[List[int]] = []
    nodes = []

    for pos in tree_elements:
        parameter_tree = pos.find("parameterTree")
        if parameter_tree is None:
            raise ValueError(f"Malformed Dimont model in {path}: missing parameter tree")

        context_pos_elem = parameter_tree.find("contextPoss")
        current_context_positions = (
            [int(_xml_numeric_value(child)) for child in context_pos_elem if child.tag == "pos"]
            if context_pos_elem is not None
            else []
        )
        context_positions.append(current_context_positions)
        nodes.append(_parse_dimont_treeelement(parameter_tree.find("root/treeElement")))

    span = 0
    for position, positions in enumerate(context_positions):
        if positions:
            span = max(span, max(position - parent for parent in positions))

    tensor = np.empty((5,) * (span + 1) + (length,), dtype=np.float32)
    context_axes = list(range(span))
    full_contexts = list(_iter_full_contexts(span))
    root_offsets = np.zeros(length, dtype=np.float64)

    for position, positions in enumerate(context_positions):
        if positions:
            continue
        if "scores" not in nodes[position]:
            raise ValueError(f"Malformed Dimont model in {path}: root position {position} is not a leaf")
        root_offsets[position] = _logsumexp(nodes[position]["scores"] + nodes[position]["log_z"])

    for position, node in enumerate(nodes):
        context_log_probs = {}

        for full_context in full_contexts:
            current = node
            while "scores" not in current:
                parent_nt = _context_value(full_context, span, position, current["context_pos"])
                current = current["children"][parent_nt]

            context_log_probs[full_context] = current["scores"] + uniform_log_odds_offset - root_offsets[position]

        tensor[..., position] = _build_position_tensor(context_log_probs, context_axes, span)

    return tensor, length, span


def write_sitega(model, path: str) -> None:
    """Write SiteGA motif to a file in the .mat format understood by mco_prc.exe."""
    from .models import get_score_bounds

    sitega_matrix = model.representation
    minimum, maximum = get_score_bounds(model)
    converter = {0: "A", 1: "C", 2: "G", 3: "T"}
    dinuc_map = {"".join(dinuc): index for index, dinuc in enumerate(itertools.product("acgt", repeat=2))}

    segments = []

    for nuc1 in range(4):
        for nuc2 in range(4):
            if np.all(np.abs(sitega_matrix[nuc1, nuc2, :]) <= 1e-9):
                continue

            dinucleotide = converter[nuc1] + converter[nuc2]
            pos = 0

            while pos < model.length:
                while pos < model.length and abs(sitega_matrix[nuc1, nuc2, pos]) <= 1e-9:
                    pos += 1

                if pos >= model.length:
                    break

                start_pos = pos
                current_val = sitega_matrix[nuc1, nuc2, pos]

                while pos + 1 < model.length and abs(sitega_matrix[nuc1, nuc2, pos + 1] - current_val) < 1e-9:
                    pos += 1

                segments.append({"start": start_pos, "stop": pos, "val": current_val, "dinucl": dinucleotide})

                pos += 1

    lpd_count = len(segments)

    with open(path, "w") as f:
        f.write(f"{model.name}\n")
        f.write(f"{lpd_count}\tLPD count\n")
        f.write(f"{model.length}\tModel length\n")
        f.write(f"{minimum:.12f}\tMinimum\n")
        f.write(f"{maximum:.12f}\tRazmah\n")

        for seg in segments:
            range_length = seg["stop"] - seg["start"] + 1
            total_value = seg["val"] * range_length
            dinuc_index = dinuc_map[seg["dinucl"].lower()]
            f.write(f"{seg['start']}\t{seg['stop']}\t{total_value:.12f}\t{dinuc_index}\t{seg['dinucl'].lower()}\n")


def write_pfm(pfm: np.ndarray, name: str, length: int, path: str) -> None:
    """Write a Position Frequency Matrix to a file."""
    with open(path, "w") as f:
        f.write(f">{name}\n")

        np.savetxt(f, pfm.T, fmt="%.6f", delimiter="\t")


def read_pfm(path: str) -> tuple[np.ndarray, int]:
    """Read a Position Frequency Matrix (PFM) from a file."""
    pfm = np.loadtxt(path, comments=">").T
    length = pfm.shape[1]
    return pfm, length


def write_dist(threshold_table: np.ndarray, max_score, min_score, path: str) -> None:
    """Write the threshold table of motif to a DIST formatted file."""
    table = copy.deepcopy(threshold_table)
    table[:, 0] = (table[:, 0] - min_score) / (max_score - min_score)
    with open(path, "w") as fname:
        np.savetxt(fname, table, fmt="%.18f", delimiter="\t", newline="\n", footer="", comments="", encoding=None)
