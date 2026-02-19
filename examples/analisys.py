# %%


from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mimosa import compare_motifs, create_config
from mimosa.io import read_fasta
from mimosa.models import BammStrategy, PwmStrategy, get_pfm
from mimosa.ragged import ragged_from_list

# %%


def load_sequences(seq_source, num_sequences=0, seq_length=500, seed=111):
    """Load or generate sequences for pipeline."""
    if seq_source and Path(seq_source).exists():
        return read_fasta(seq_source)
    else:
        rng = np.random.default_rng(seed)
        sequences = []
        for _ in range(num_sequences):
            seq = rng.integers(0, 4, size=seq_length, dtype=np.int8)
            sequences.append(seq)
        return ragged_from_list(sequences, dtype=np.int8)


BASES = ["A", "C", "G", "T"]
RC_COLS = [3, 2, 1, 0]


def _normalize_pfm(pfm):
    arr = np.asarray(pfm, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    if arr.shape[1] != 4 and arr.shape[0] == 4:
        arr = arr.T
    if arr.shape[1] != 4:
        raise ValueError(f"PFM must have 4 columns (A,C,G,T), got shape={arr.shape}")

    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return arr / row_sums


def _orient_pfm(pfm, orientation):
    if orientation in (-1, "-1", "-", "reverse", "rc"):
        return pfm[::-1][:, RC_COLS]
    return pfm


def _extract_offset_orientation(result):
    if isinstance(result, (tuple, list)):
        if len(result) >= 3:
            return int(result[1]), result[2]
        if len(result) >= 2:
            return int(result[0]), result[1]
        return 0, 1

    if isinstance(result, dict):
        offset = result.get("offset", 0)
        orientation = result.get("orientation", 1)
        return int(offset), orientation

    offset = getattr(result, "offset", 0)
    orientation = getattr(result, "orientation", 1)
    return int(offset), orientation


def _pad_for_alignment(pfm_1, pfm_2, offset):
    len_1, len_2 = pfm_1.shape[0], pfm_2.shape[0]

    if offset >= 0:
        left_1, left_2 = 0, offset
    else:
        left_1, left_2 = -offset, 0

    total_len = max(left_1 + len_1, left_2 + len_2)
    right_1 = total_len - left_1 - len_1
    right_2 = total_len - left_2 - len_2

    pad_1 = np.pad(
        pfm_1,
        ((left_1, right_1), (0, 0)),
        mode="constant",
        constant_values=0.25,
    )
    pad_2 = np.pad(
        pfm_2,
        ((left_2, right_2), (0, 0)),
        mode="constant",
        constant_values=0.25,
    )
    return pad_1, pad_2


def plot_aligned_logos(pfm_1, pfm_2, result, name_1="model_1", name_2="model_2"):
    offset, orientation = _extract_offset_orientation(result)

    pfm_1 = _normalize_pfm(pfm_1)
    pfm_2 = _normalize_pfm(pfm_2)
    pfm_2 = _orient_pfm(pfm_2, orientation)

    pad_1, pad_2 = _pad_for_alignment(pfm_1, pfm_2, offset)

    df_1 = pd.DataFrame({base: pad_1[:, i] for i, base in enumerate(BASES)})
    df_2 = pd.DataFrame({base: pad_2[:, i] for i, base in enumerate(BASES)})

    info_1 = logomaker.transform_matrix(df_1, from_type="probability", to_type="information")
    info_2 = logomaker.transform_matrix(df_2, from_type="probability", to_type="information")

    fig, axes = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True, constrained_layout=True)
    logomaker.Logo(info_1, ax=axes[0], color_scheme="classic")
    logomaker.Logo(info_2, ax=axes[1], color_scheme="classic")

    axes[0].set_title(name_1)
    axes[1].set_title(f"{name_2} (orientation={orientation}, offset={offset})")
    axes[0].set_ylabel("bits")
    axes[1].set_ylabel("bits")
    axes[1].set_xlabel("aligned position")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axes


sequences = load_sequences(None, num_sequences=5000, seq_length=500, seed=111)
promoters = read_fasta("/home/anton/Documents/genomes/hs/promoters.p400m100.fa")

# %%


bamm = BammStrategy.load(
    "/home/anton/Downloads/GTRD_v1804_HUMAN/models/ATF-1/ATF-1.ihbcp",
    kwargs={"order": 1},
)

pwm = PwmStrategy.load(
    "/home/anton/Downloads/HOCOMOCOv11_HUMAN/hocomocoV11_HUMAN/source/initPWMs/ATF1_HUMAN.H11MO.0.B.meme",
    kwargs={},
)
# %%

config = create_config(
    model1=pwm,
    model2=bamm,
    strategy="universal",
    sequences=sequences,
    seed=100,
    metric="co",
)
result = compare_motifs(
    model1=config.model1,
    model2=config.model2,
    strategy=config.strategy,
    sequences=config.sequences,
    comparator=config.comparator,
)

fig, axes = plot_aligned_logos(
    get_pfm(pwm, sequences),
    get_pfm(bamm, sequences),
    result,
    name_1="PWM",
    name_2="BaMM",
)
plt.show()


# %%

result = compare_motifs(
    model1=pwm,
    model2=bamm,
    strategy="universal",
    sequences=sequences,
    seed=100,
    metric="co",
    n_permutations=500,
)
fig, axes = plot_aligned_logos(
    get_pfm(pwm, sequences),
    get_pfm(bamm, sequences),
    result,
    name_1="PWM",
    name_2="BaMM",
)
plt.show()

# %%


result = compare_motifs(
    model1=pwm,
    model2=bamm,
    strategy="tomtom",
    sequences=sequences,
    seed=100,
    metric="pcc",
    n_permutations=500,
)
fig, axes = plot_aligned_logos(
    get_pfm(pwm, sequences),
    get_pfm(bamm, sequences),
    result,
    name_1="PWM",
    name_2="BaMM",
)
plt.show()

# %%
