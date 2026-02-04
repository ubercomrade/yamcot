from __future__ import annotations

import logging
import os
from typing import Iterable, List, Tuple, Union

import numpy as np

from mimosa.functions import pfm_to_pwm
from mimosa.ragged import RaggedData


def read_fasta(path: str) -> RaggedData:
    """Read a FASTA file and return integer-encoded sequences.

    Parameters
    ----------
    path : str
        Path to a FASTA formatted file.
    return_ragged : bool, default True
        If True, returns RaggedData object. If False, returns List[np.ndarray]
        for backward compatibility.

    Returns
    -------
    Union[RaggedData, List[np.ndarray]]
        Integer-encoded sequences (dtype=np.int8).
        Using RaggedData is more memory-efficient and faster for large files
        as it avoids multiple small allocations.
    """
    # Translation table for fast conversion of strings to bytes (int8)
    # 0=A, 1=C, 2=G, 3=T, 4=N (and others)
    trans_table = bytearray([4] * 256)
    for char, code in zip(b"ACGTacgt", [0, 1, 2, 3] * 2, strict=False):
        trans_table[char] = code

    # First collect lengths and data in one pass if possible,
    # but FASTA requires parsing to determine lengths.
    # For efficiency, we use a temporary list or intermediate buffer.

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

    # Convert to RaggedData
    n = len(sequences)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(seq)

    total_data = np.empty(offsets[-1], dtype=np.int8)
    for i, seq in enumerate(sequences):
        total_data[offsets[i] : offsets[i + 1]] = seq

    return RaggedData(total_data, offsets)


def write_fasta(sequences: Union[RaggedData, Iterable[np.ndarray]], path: str) -> None:
    """Write integer-encoded sequences to a FASTA file.

    Parameters
    ----------
    sequences : Union[RaggedData, Iterable[np.ndarray]]
        RaggedData object or a collection of integer-encoded sequences (0=A, 1=C, 2=G, 3=T, 4=N).
    path : str
        Path to the output file.
    """
    # Array for converting indices back to symbols
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
    """Read a specific motif from a MEME formatted file and return total count.

    Parameters
    ----------
    path : str
        Path to the MEME file.
    index : int, default 0
        The zero-based index of the motif to return.

    Returns
    -------
    Tuple[np.ndarray, Tuple[str, int], int]
        A tuple containing:
        - The requested motif matrix (shape (4, L))
        - A tuple with the motif's name and length
        - Total number of motifs found in the file
    """
    target_motif: np.ndarray | None = None
    target_info: Tuple[str, int] | None = None
    motif_count = 0

    with open(path) as handle:
        line = handle.readline()
        while line:
            if line.startswith("MOTIF"):
                # Check if this is the motif we are looking for
                is_target = motif_count == index
                motif_count += 1

                parts = line.strip().split()
                name = parts[1]

                # Read header line containing motif length (w=)
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

                    # Transpose into shape (4, length)
                    target_motif = np.array(matrix, dtype=np.float32).T
                    target_info = (name, length)
                else:
                    # Skip the matrix rows for other motifs to save time
                    for _ in range(length):
                        handle.readline()

            line = handle.readline()

    if target_motif is None:
        if motif_count == 0:
            raise ValueError(f"No motifs found in {path}")
        else:
            raise IndexError(f"Motif index {index} out of range. File contains {motif_count} motifs.")

    # We know that if target_motif is not None, then target_info is also not None
    # because they are set together in the same condition
    assert target_info is not None

    return target_motif, target_info, motif_count


def write_meme(motifs: List[np.ndarray], info: List[Tuple[str, int]], path: str) -> None:
    """Write a list of motifs to a MEME formatted file.

    Parameters
    ----------
    motifs : List[np.ndarray]
        List of motif matrices of shape (5, L).  Only the first four rows
        (A, C, G, T) are written; the fifth row is ignored.
    info : List[Tuple[str, int]]
        A list of (name, length) tuples corresponding to the motifs.
    path : str
        Path of the output file.
    """
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


def read_sitega(path: str) -> tuple[np.ndarray, int, float, float]:
    """Parse SiteGA output file and return the motif matrix with metadata.

    Parameters
    ----------
    path : str
        Path to the SiteGA output file (typically ends with '.mat').

    Returns
    -------
    tuple[np.ndarray, int, float, float]
        A tuple containing:
        - SiteGA matrix of shape (5, 5, length) representing dinucleotide dependencies
        - Length of the motif
        - Minimum score value
        - Maximum score value
    """
    converter = {"A": 0, "C": 1, "G": 2, "T": 3}
    with open(path) as file:
        _name = file.readline().strip()
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
    return np.array(sitega, dtype=np.float32), length, minimum, maximum


def parse_file_content(filepath: str) -> tuple[dict[int, list[np.ndarray]], int, int]:
    """Parse BaMM file content, ignoring comments starting with '#'.

    Parameters
    ----------
    filepath : str
        Path to the BaMM file to parse.

    Returns
    -------
    tuple[dict[int, list[np.ndarray]], int, int]
        A tuple containing:
        - Dictionary mapping order indices to lists of coefficient arrays
        - Maximum order found in the file
        - Number of positions (length of motif)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If no valid data is found in the file or inconsistent orders are detected.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    with open(filepath, "r") as f:
        raw_text = f.read()

    # Split blocks by double newline
    raw_blocks = raw_text.strip().split("\n\n")
    clean_blocks_data = []

    for raw_block in raw_blocks:
        lines = raw_block.strip().split("\n")

        # Filter comments and empty lines
        valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        if not valid_lines:
            continue

        block_arrays = []
        for line in valid_lines:
            # Check for potential empty strings after split
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
    """Read BaMM files, apply ramp-up logic, and add padding for 'N' (index 4).

    This function reads motif and background BaMM files, computes log-odds ratios,
    applies a ramp-up strategy for lower-order coefficients, and adds padding for
    ambiguous nucleotides ('N') by setting their scores to the minimum of each position.

    Parameters
    ----------
    motif_path : str
        Path to the motif BaMM file (.ihbcp format).
    bg_path : str
        Path to the background BaMM file (.hbcp format).
    target_order : int
        Target order for the BaMM model (determines tensor dimensions).

    Returns
    -------
    np.ndarray
        3D+ tensor of shape (5, 5, ..., 5, Length) where the number of dimensions
        equals target_order + 2. The final dimension represents motif length,
        and the first target_order+1 dimensions represent nucleotide dependencies
        including 'N' padding at index 4.

    Raises
    ------
    ValueError
        If target order exceeds the maximum order in the file.
    """
    # 1. Parse Data
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

    # 2. Build 4x4...x4 Tensor slices (Standard Logic)
    # We first build the pure ACGT tensor to compute minimums correctly
    acgt_slices = []

    for pos in range(motif_length):
        current_k = min(pos, target_order)

        p_motif = motif_raw[current_k][pos]
        p_bg = bg_raw[current_k][0]

        epsilon = 1e-10
        log_odds = np.log2((p_motif + epsilon) / (p_bg + epsilon))

        # Reshape & Broadcast (Ramp-up)
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

    # Stack to (4, 4, ..., 4, Length)
    # This tensor contains only valid ACGT scores
    acgt_tensor = np.stack(acgt_slices, axis=-1)

    # 3. Create 5x5...x5 Tensor with N-padding

    # Calculate global minimum per position (over all ACGT contexts)
    # We want min over axes (0, 1, ..., target_order).
    # acgt_tensor shape is (4, 4, ..., L). Last axis is Length.
    reduce_axes = tuple(range(target_order + 1))
    min_scores_per_pos = np.min(acgt_tensor, axis=reduce_axes)  # Shape (Length,)

    # Define new shape: (5, 5, ..., 5, Length)
    new_shape = [5] * (target_order + 1) + [motif_length]

    # Initialize with min values broadcasted
    # NumPy broadcasts from last dimension: (Length,) broadcasts to (5, 5, ..., 5, Length)
    final_tensor = np.ones(new_shape, dtype=np.float32) * min_scores_per_pos

    # 4. Copy ACGT data into the 5x5 structure
    # We need to slice [0:4, 0:4, ..., :]
    slice_objs = [slice(0, 4)] * (target_order + 1) + [slice(None)]
    final_tensor[tuple(slice_objs)] = acgt_tensor

    # Return as contiguous array for Numba
    return np.array(final_tensor, dtype=np.float32)


def write_sitega(motif, path: str) -> None:
    """Write SiteGA motif to a file in the .mat format understood by mco_prc.exe.

    Parameters
    ----------
    motif : SitegaMotif
        The SiteGA motif to write.
    path : str
        Path to the output file.
    """
    sitega_matrix = motif.matrix
    converter = {0: "A", 1: "C", 2: "G", 3: "T"}

    # Список для хранения найденных сегментов (start, stop, value, dinucleotide)
    segments = []

    # Один проход для сбора всех данных
    for nuc1 in range(4):
        for nuc2 in range(4):
            # Пропускаем, если все значения для динуклеотида нулевые
            if np.all(np.abs(sitega_matrix[nuc1, nuc2, :]) <= 1e-9):
                continue

            dinucleotide = converter[nuc1] + converter[nuc2]
            pos = 0

            while pos < motif.length:
                # Пропускаем нули
                while pos < motif.length and abs(sitega_matrix[nuc1, nuc2, pos]) <= 1e-9:
                    pos += 1

                if pos >= motif.length:
                    break

                # Начало ненулевой последовательности
                start_pos = pos
                current_val = sitega_matrix[nuc1, nuc2, pos]

                # Ищем конец последовательности с одинаковым значением
                while pos + 1 < motif.length and abs(sitega_matrix[nuc1, nuc2, pos + 1] - current_val) < 1e-9:
                    pos += 1

                # Сохраняем сегмент
                segments.append({"start": start_pos, "stop": pos, "val": current_val, "dinucl": dinucleotide})

                pos += 1

    # Количество строк данных теперь просто длина списка
    lpd_count = len(segments)

    with open(path, "w") as f:
        f.write(f"{motif.name}\n")
        f.write(f"{lpd_count}\tLPD count\n")
        f.write(f"{motif.length}\tModel length\n")
        f.write(f"{motif.minimum:.12f}\tMinimum\n")
        f.write(f"{motif.maximum:.12f}\tRazmah\n")

        # Записываем данные из собранного списка
        for seg in segments:
            range_length = seg["stop"] - seg["start"] + 1
            total_value = seg["val"] * range_length

            f.write(f"{seg['start']}\t{seg['stop']}\t{total_value:.12f}\t0\t{seg['dinucl'].lower()}\n")


def write_pfm(pfm: np.ndarray, name: str, length: int, path: str) -> None:
    """Write a Position Frequency Matrix to a file.

    Parameters
    ----------
    pfm : np.ndarray
        Position frequency matrix of shape (4, length).
    name : str
        Name of the motif.
    length : int
        Length of the motif.
    path : str
        Path to the output file.
    """
    with open(path, "w") as f:
        f.write(f">{name}\n")
        # Transpose the matrix to get the right format
        np.savetxt(f, pfm.T, fmt="%.6f", delimiter="\t")


def read_pfm(path: str) -> tuple[np.ndarray, int]:
    """Read a Position Frequency Matrix (PFM) from a file and convert to PWM.

    Parameters
    ----------
    path : str
        Path to the PFM file.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - PFM matrix with shape (4, L)
        - Length of the motif
    """
    pfm = np.loadtxt(path, comments=">").T
    length = pfm.shape[1]
    return pfm, length
