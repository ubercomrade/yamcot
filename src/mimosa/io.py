from __future__ import annotations

import copy
import itertools
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np

from mimosa.ragged import RaggedData


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


def write_meme(motifs: List[np.ndarray], info: List[Tuple[str, int]], path: str) -> None:
    """Write a list of motifs to a MEME formatted file."""
    with open(path, "w") as out:
        out.write("MEME version 4\n\n")
        out.write("ALPHABET= ACGT\n\n")
        out.write("strands: + -\n\n")
        out.write("Background letter frequencies\n")
        out.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for motif, (name, length) in zip(motifs, info, strict=False):
            out.write(f"MOTIF {name}\n")
            out.write(f"letter-probability matrix: alength= 4 w= {length}\n")
            for row in motif[:4].T:
                out.write(" " + " ".join(f"{val:.6f}" for val in row) + "\n")
            out.write("\n")


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


def read_bamm(motif_path: str, bg_path: str, target_order: int) -> np.ndarray:
    """Read BaMM files, apply ramp-up logic, and add padding for 'N' (index 4)."""

    motif_raw, max_order_file, motif_length = parse_file_content(motif_path)
    bg_raw, max_order_bg, _ = parse_file_content(bg_path)

    if max_order_file > max_order_bg:
        max_order_file = max_order_bg
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
        p_bg = bg_raw[current_k][0]

        epsilon = 1e-10
        log_odds = np.log2((p_motif + epsilon) / (p_bg + epsilon))

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
    """Read a Position Frequency Matrix (PFM) from a file and convert to PWM."""
    pfm = np.loadtxt(path, comments=">").T
    length = pfm.shape[1]
    return pfm, length


def write_dist(threshold_table: np.ndarray, max_score, min_score, path: str) -> None:
    """Write the threshold table of motif to a DIST formatted file."""
    table = copy.deepcopy(threshold_table)
    table[:, 0] = (table[:, 0] - min_score) / (max_score - min_score)
    with open(path, "w") as fname:
        np.savetxt(fname, table, fmt="%.18f", delimiter="\t", newline="\n", footer="", comments="", encoding=None)
