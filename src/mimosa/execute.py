import logging

import numpy as np

from mimosa._core import run_motali_cpp


def run_motali(
    fasta_path,
    motif_1,
    motif_2,
    type_1,
    type_2,
    dist_1,
    dist_2,
    overlap_path,
    all_path,
    prc_path,
    hist_path,
    sta_path,
    shift: int = 50,
    err: float = 0.002,
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
        shift=shift,
        threshold=err,
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
        for _ in range(3):
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
