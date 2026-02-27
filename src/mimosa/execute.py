import logging
import os
import subprocess
from io import StringIO

import numpy as np
import pandas as pd

from mimosa._core import run_motali_cpp


def run_prosampler(foreground_path, background_path, output_dir, motif_length, number_of_motifs):
    """Run ProSampler and write results to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = [
        "ProSampler",
        "-i",
        foreground_path,
        "-b",
        background_path,
        "-k",
        f"{motif_length}",
        "-l",
        "0",
        "-m",
        f"{number_of_motifs}",
        "-z",
        "0",
        "-t",
        "4",
        "-w",
        "2",
        "-o",
        f"{output_dir}/motifs",
    ]
    logger = logging.getLogger(__name__)
    logger.debug(" ".join(args))
    stdout = subprocess.run(args, shell=False, capture_output=True)
    logger.debug(stdout)
    return 0


def run_tomtom(motifs_1, motifs_2):
    """Run TomTom and return the top match as a dictionary."""
    args = ["tomtom", motifs_1, motifs_2, "-thresh", "1", "-text"]

    logger = logging.getLogger(__name__)
    logger.debug(" ".join(args))

    stdout = subprocess.run(args, shell=False, capture_output=True)

    logger.debug(stdout)

    table = pd.read_csv(StringIO(stdout.stdout.decode()), sep="\t", comment="#")
    table = table[["Query_ID", "Target_ID", "p-value", "Overlap", "Orientation"]]
    table.columns = ["query", "target", "p-value", "overlap", "orientation"]
    table = table.reset_index(drop=True)

    return table.iloc[0].to_dict()


def run_motali(
    fasta_path, motif_1, motif_2, type_1, type_2, dist_1, dist_2, overlap_path, all_path, prc_path, hist_path, sta_path
):
    """
    Run motali comparison using either the C++ extension or subprocess fallback.
    """
    logger = logging.getLogger(__name__)

    if type_1 == "sitega":
        type_1 = "sga"

    if type_2 == "sitega":
        type_2 = "sga"

    result = run_motali_cpp(
        file_fasta=fasta_path,
        type_model_1=type_1,
        type_model_2=type_2,
        file_model_1=motif_1,
        file_model_2=motif_2,
        file_table_1=dist_1,
        file_table_2=dist_2,
        shift=50,
        threshold=0.002,
        file_hist=hist_path,
        yes_out_hist=1,
        file_prc=prc_path,
        yes_out_prc=1,
        file_short_over=overlap_path,
        file_short_all=all_path,
        file_sta_long=sta_path,
    )

    if result != 0:
        logger.error(f"C++ function returned error code: {result}")
        raise RuntimeError(f"C++ function failed with error code: {result}")

    with open(all_path) as file:
        score = float(file.readline().strip())

    container = []
    with open(hist_path) as file:
        file.readline()
        for i in range(3):
            container.append(file.readline().strip().split())

    positions = np.array([int(float(i)) for i in container[0]])
    directed = np.array([float(i) for i in container[1][2:]])
    inverted = np.array([float(i) for i in container[2][2:]])

    condition = np.logical_and(positions <= 10, positions >= -10)
    positions = positions[condition]
    directed = directed[condition]
    inverted = inverted[condition]

    ind_dir_max = np.argmax(directed)
    ind_inv_max = np.argmax(inverted)

    if directed[ind_dir_max] > inverted[ind_inv_max]:
        off = positions[ind_dir_max]
        orient = "++"
    else:
        off = positions[ind_inv_max]
        orient = "+-"

    return score, off, orient
