# MIMOSA

MIMOSA (`mimosa-tool`) is a Python package and CLI for model-independent motif similarity assessment.
It supports two complementary comparison strategies:

- `profile`: compare score profiles, either from precomputed score tracks or from motif scans on sequences
- `motif`: compare motif representations directly, with optional PFM reconstruction when native representations are incompatible

The current package version is `1.1.4`.

## Supported Inputs

| Family | CLI key | Typical inputs |
| :--- | :--- | :--- |
| Precomputed score profiles | `scores` | FASTA-like file with numeric rows |
| PWM / PFM | `pwm` | `.meme`, `.pfm`, or compatible `joblib`-pickled `GenericModel` |
| BaMM | `bamm` | `.ihbcp` or a basename resolvable to `.ihbcp` |
| SiteGA | `sitega` | `.mat` or compatible `joblib`-pickled `GenericModel` |
| DiMotif / Dimont | `dimont` | `.xml` or compatible `joblib`-pickled `GenericModel` |
| Slim | `slim` | `.xml` or compatible `joblib`-pickled `GenericModel` |

Notes:

- `profile` supports all six types above, including direct `scores` vs `scores`
- `motif` supports all motif families except `scores`
- heterogeneous `motif` comparisons automatically switch to sequence-driven PFM reconstruction
- BaMM loading uses a uniform background model; a separate background file is not required

## Installation

MIMOSA requires Python `3.10+`.

Install from PyPI:

```bash
uv pip install mimosa-tool
```

or:

```bash
pip install mimosa-tool
```

Install from source:

```bash
git clone https://github.com/ubercomrade/mimosa.git
cd mimosa
uv sync --group dev
uv pip install -e . --no-build-isolation
```

Main runtime dependencies are declared in [pyproject.toml](pyproject.toml): `numpy`, `scipy`, `pandas`,
`joblib`, and `numba`.

## Quick Start

Compare two precomputed score profiles:

```bash
mimosa profile examples/scores_1.fasta examples/scores_2.fasta \
  --model1-type scores \
  --model2-type scores \
  --metric cosine
```

Compare two motifs through normalized score profiles:

```bash
mimosa profile examples/gata2.meme examples/gata4.meme \
  --model1-type pwm \
  --model2-type pwm \
  --fasta examples/foreground.fa \
  --background examples/background.fa \
  --metric co \
  --min-logfpr 2 \
  --window-radius 6
```

Compare two motifs directly:

```bash
mimosa motif examples/pif4.meme examples/gata2.meme \
  --model1-type pwm \
  --model2-type pwm \
  --metric pcc
```

Compare motifs from different model families in `motif` mode:

```bash
mimosa motif examples/sitega_stat6.mat examples/pif4.meme \
  --model1-type sitega \
  --model2-type pwm \
  --metric pcc \
  --pfm-mode
```

Clear cached normalized profiles:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

## CLI Overview

The CLI exposes three top-level commands:

- `mimosa profile`
- `mimosa motif`
- `mimosa cache clear`

Comparison commands print one JSON object to `stdout`. Cache maintenance also prints JSON.

Typical `profile` result:

```json
{
  "query": "model_or_profile_1",
  "target": "model_or_profile_2",
  "score": 0.0,
  "offset": 0,
  "orientation": "++",
  "metric": "co",
  "n_sites": 0
}
```

Typical `motif` result:

```json
{
  "query": "model_1",
  "target": "model_2",
  "score": 0.0,
  "offset": 0,
  "orientation": "++",
  "metric": "pcc"
}
```

If `--permutations` is greater than `0`, the result may also include:

- `p-value`
- `z-score`
- `null_mean`
- `null_std`

`orientation` is one of `++`, `+-`, `-+`, `--`.

## `profile` Mode

`profile` is the general-purpose workflow for comparing positional score signals.

It supports:

- direct `scores` vs `scores` comparisons
- motif-vs-motif comparisons through sequence scanning
- empirical score normalization with background calibration
- optional on-disk caching of normalized profile bundles
- Monte Carlo p-value estimation with surrogate profiles

If motif scanning is required and `--fasta` is omitted, MIMOSA generates random A/C/G/T sequences with:

- `--num-sequences` default `1000`
- `--seq-length` default `200`

If `--background` is provided, normalization is fitted on those sequences and applied to the comparison set.
If omitted, normalization is fitted on the same sequence set used for comparison. The hidden legacy alias
`--promoters` is still accepted for compatibility, but `--background` is the documented name.

Supported metrics:

- `co`
- `dice`
- `cosine`

Important arguments:

| Flag | Meaning |
| :--- | :--- |
| `--model1-type`, `--model2-type` | `scores`, `pwm`, `bamm`, `sitega`, `dimont`, `slim` |
| `--fasta` | FASTA sequences used to scan motif inputs |
| `--background` | FASTA used to calibrate empirical profile normalization |
| `--num-sequences` | number of random sequences when `--fasta` is omitted |
| `--seq-length` | random sequence length when `--fasta` is omitted |
| `--metric` | `co`, `dice`, or `cosine` |
| `--permutations` | number of surrogate-profile permutations |
| `--distortion` | surrogate distortion level, default `0.4` |
| `--search-range` | maximum motif shift explored, default `10` |
| `--window-radius` | site-centered half-window size, default `10` |
| `--realign-window` | local target-anchor realignment half-width, default `3` |
| `--min-kernel-size` | minimum surrogate kernel size, default `3` |
| `--max-kernel-size` | maximum surrogate kernel size, default `11` |
| `--min-logfpr` | select all anchors at or above this empirical log-tail threshold |
| `--cache` | `off` or `on` |
| `--cache-dir` | cache directory, default `.mimosa-cache` |
| `--seed` | random seed, default `127` |
| `--jobs` | number of parallel jobs, default `-1` |
| `-v`, `--verbose` | verbose logging |

Example with two motif models:

```bash
mimosa profile examples/myog.ihbcp examples/pif4.meme \
  --model1-type bamm \
  --model2-type pwm \
  --metric co \
  --permutations 100
```

## `motif` Mode

`motif` compares motif representations directly.

Supported metrics:

- `pcc`
- `ed`
- `cosine`

Execution rules:

- if model types match and `--pfm-mode` is not enabled, MIMOSA aligns native matrices or tensors directly
- if model types differ, MIMOSA reconstructs PFMs from sequences automatically
- `--pfm-mode` forces PFM reconstruction even when both inputs use the same model family

If PFM reconstruction is required and `--fasta` is omitted, MIMOSA generates random A/C/G/T sequences with:

- `--num-sequences` default `20000`
- `--seq-length` default `100`

Important arguments:

| Flag | Meaning |
| :--- | :--- |
| `--model1-type`, `--model2-type` | `pwm`, `bamm`, `sitega`, `dimont`, `slim` |
| `--fasta` | optional FASTA for PFM reconstruction |
| `--num-sequences` | number of random sequences when `--fasta` is omitted |
| `--seq-length` | random sequence length when `--fasta` is omitted |
| `--metric` | `pcc`, `ed`, or `cosine` |
| `--permutations` | number of Monte Carlo permutations |
| `--permute-rows` | also shuffle alphabet rows during null generation |
| `--pfm-mode` | force sequence-driven PFM reconstruction |
| `--pfm-top-fraction` | top fraction of reconstructed hits used for PFM building, default `0.05` |
| `--seed` | random seed, default `127` |
| `--jobs` | number of parallel jobs, default `-1` |
| `-v`, `--verbose` | verbose logging |

Example:

```bash
mimosa motif examples/sitega_gata2.mat examples/pif4.meme \
  --model1-type sitega \
  --model2-type pwm \
  --metric ed \
  --permutations 100 \
  --pfm-mode
```

## `cache` Command

`mimosa cache clear` removes derived profile artifacts created by `mimosa profile --cache on`.

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

The command prints:

```json
{
  "cache_dir": ".mimosa-cache",
  "removed": 0
}
```

## Python API

The public API is exported from [src/mimosa/__init__.py](src/mimosa/__init__.py).

High-level comparison helpers:

- `compare_motifs(...)`
- `compare_one_to_many(...)`
- `create_config(...)`
- `create_many_config(...)`
- `run_comparison(...)`
- `run_one_to_many(...)`
- `create_comparator_config(...)`
- `compare(...)`

Model and scanning helpers:

- `read_model(...)`
- `scan_model(...)`
- `get_scores(...)`
- `get_frequencies(...)`
- `get_sites(...)`
- `get_pfm(...)`
- `register_model_handler(...)`

Types and utility exports:

- `GenericModel`
- `ComparisonConfig`
- `OneToManyConfig`
- `ComparatorConfig`
- `clear_cache(...)`

Single comparison example:

```python
from mimosa import compare_motifs

result = compare_motifs(
    "examples/gata2.meme",
    "examples/gata4.meme",
    model1_type="pwm",
    model2_type="pwm",
    strategy="profile",
    sequences="examples/foreground.fa",
    background="examples/background.fa",
    metric="co",
    min_logfpr=2.0,
)
```

One-vs-many example:

```python
from mimosa import compare_one_to_many

results = compare_one_to_many(
    "examples/pif4.meme",
    [
        "examples/gata2.meme",
        "examples/foxa2.meme",
    ],
    query_type="pwm",
    target_type="pwm",
    strategy="motif",
    metric="pcc",
)
```

Site extraction and PFM reconstruction example:

```python
from mimosa import get_pfm, get_sites, read_model
from mimosa.io import read_fasta

model = read_model("examples/pif4.meme", "pwm")
sequences = read_fasta("examples/foreground.fa")

sites = get_sites(model, sequences, mode="best")
pfm = get_pfm(model, sequences, mode="best", top_fraction=0.05)
```

## Extension Hooks

Custom model families can be registered at runtime through `register_model_handler(...)`.
A handler bundle provides:

- `load`
- `scan`
- optional `scan_both`
- `write`
- `score_bounds`

This makes it possible to integrate new motif representations without changing the public comparison API.

## Development

Common local commands:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Focused test commands:

```bash
uv run pytest tests/test_unit.py
uv run pytest tests/test_integration.py
```

## License

MIT. See [LICENSE](LICENSE).
