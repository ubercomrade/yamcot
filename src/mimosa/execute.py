import logging
import os
import subprocess
from io import StringIO

import pandas as pd

# Import the C++ extension
from mimosa._core import run_motali_cpp


def run_prosampler(foreground_path, background_path, output_dir, motif_length, number_of_motifs):
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

    # Use the C++ extension directly
    result = run_motali_cpp(
        file_fasta=fasta_path,
        type_model_1=type_1,
        type_model_2=type_2,
        file_model_1=motif_1,
        file_model_2=motif_2,
        file_table_1=dist_1,
        file_table_2=dist_2,
        shift=50,  # Default shift value
        threshold=0.002,  # Default threshold
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

    # Read the result from the output file
    with open(all_path) as file:
        score = float(file.readline().strip())

    return score
