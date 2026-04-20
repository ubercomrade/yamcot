# MIMOSA

MIMOSA (`mimosa-tool`) is a Python package and CLI for comparing transcription factor binding motifs across
different model families.

The project currently exposes two comparison workflows:

- `profile`: compare score profiles produced by scanning motifs on sequences, or compare precomputed score tracks
- `motif`: compare motif matrices/tensors directly, with automatic PFM reconstruction for heterogeneous model types

## What MIMOSA Supports

Supported model types in the current codebase:

| Type | CLI key | Typical input |
| :--- | :--- | :--- |
| Score profiles | `scores` | FASTA-like file with numeric values |
| PWM/PFM | `pwm` | `.meme`, `.pfm`, or compatible `.pkl` |
| BaMM | `bamm` | `.ihbcp` or basename resolvable to `.ihbcp` |
| SiteGA | `sitega` | `.mat` or compatible `.pkl` |
| Dimont | `dimont` | `.xml` or compatible `.pkl` |
| Slim | `slim` | `.xml` or compatible `.pkl` |

High-level capabilities:

- compare motifs in `profile` mode with `co` or `dice`
- compare motifs in `motif` mode with `pcc`, `ed`, or `cosine`
- estimate empirical null statistics with Monte Carlo permutations
- calibrate profile scores with empirical log-tail normalization
- cache normalized profile bundles on disk in `profile` mode
- use the package as a library, including one-vs-many comparisons
- register custom model handlers through `register_model_handler(...)`

## Installation

MIMOSA requires Python 3.10+.

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
  --metric co
```

Compare two motifs through sequence-derived profiles:

```bash
mimosa profile examples/foxa2.meme examples/gata4.meme \
  --model1-type pwm \
  --model2-type pwm \
  --fasta examples/foreground.fa \
  --promoters examples/background.fa \
  --metric co \
  --permutations 100
```

Compare two motifs directly:

```bash
mimosa motif examples/pif4.meme examples/gata2.meme \
  --model1-type pwm \
  --model2-type pwm \
  --metric pcc
```

Clear cached profile artifacts:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

## CLI Overview

The CLI has three subcommands:

- `mimosa profile`
- `mimosa motif`
- `mimosa cache clear`

All comparison commands print JSON to stdout.

Base result fields:

```json
{
  "query": "model_or_profile_1",
  "target": "model_or_profile_2",
  "score": 0.0,
  "offset": 0,
  "orientation": "++",
  "metric": "co"
}
```

If `--permutations` is greater than `0`, the result also includes:

- `p-value`
- `z-score`
- `null_mean`
- `null_std`

`orientation` is one of `++`, `+-`, `-+`, `--`. `offset` is the target start relative to the query start after the
reported orientation is applied.

## `profile` Mode

`profile` is the universal workflow. It supports:

- direct comparison of precomputed score profiles with `--model*-type scores`
- motif-vs-motif comparison through sequence scanning

Supported input types:

- `scores`
- `pwm`
- `bamm`
- `sitega`
- `dimont`
- `slim`

Profile normalization currently uses empirical log-tail calibration. If `--promoters` is provided, normalization is
fit on the promoter/background set and then applied to the comparison sequences. Otherwise it is fit on the same
sequence set that is used for comparison.

If motif scanning is required and `--fasta` is omitted, MIMOSA generates random A/C/G/T sequences with:

- `--num-sequences` default `1000`
- `--seq-length` default `200`

Common example:

```bash
mimosa profile examples/myog.ihbcp examples/pif4.meme \
  --model1-type bamm \
  --model2-type pwm \
  --metric co \
  --permutations 100
```

Key arguments:

| Flag | Values / meaning |
| :--- | :--- |
| `--model1-type`, `--model2-type` | `scores`, `pwm`, `bamm`, `sitega`, `dimont`, `slim` |
| `--fasta` | FASTA file used for scanning motif inputs |
| `--promoters` | optional FASTA used to fit profile normalization |
| `--metric` | `co` or `dice` |
| `--permutations` | number of surrogate-profile permutations |
| `--distortion` | surrogate distortion level, default `0.4` |
| `--search-range` | maximum alignment offset, default `10` |
| `--min-kernel-size` | minimum surrogate kernel size, default `3` |
| `--max-kernel-size` | maximum surrogate kernel size, default `11` |
| `--min-logfpr` | ignore aligned positions only when both values are below this threshold |
| `--cache` | `off` or `on` |
| `--cache-dir` | cache directory, default `.mimosa-cache` |
| `--seed` | random seed, default `127` |
| `--jobs` | numba thread count, default `-1` |
| `-v`, `--verbose` | verbose logging |

## `motif` Mode

`motif` compares motif representations directly.

Supported input types:

- `pwm`
- `bamm`
- `sitega`
- `dimont`
- `slim`

Two execution paths are used:

- if model types match and `--pfm-mode` is not enabled, MIMOSA aligns the native matrix/tensor representations
- if model types differ, or `--pfm-mode` is enabled, MIMOSA reconstructs PFMs from best hits on sequences and
  compares the reconstructed matrices

If sequence-driven reconstruction is required and `--fasta` is omitted, MIMOSA generates random sequences with:

- `--num-sequences` default `20000`
- `--seq-length` default `100`

Common example:

```bash
mimosa motif examples/sitega_stat6.mat examples/pif4.meme \
  --model1-type sitega \
  --model2-type pwm \
  --metric pcc \
  --pfm-mode \
  --permutations 100
```

Key arguments:

| Flag | Values / meaning |
| :--- | :--- |
| `--model1-type`, `--model2-type` | `pwm`, `bamm`, `sitega`, `dimont`, `slim` |
| `--fasta` | optional FASTA for PFM reconstruction |
| `--metric` | `pcc`, `ed`, or `cosine` |
| `--permutations` | number of Monte Carlo motif permutations |
| `--permute-rows` | also permute alphabet rows during null generation |
| `--pfm-mode` | force sequence-driven PFM reconstruction |
| `--pfm-top-fraction` | top fraction of hits used for PFM reconstruction, default `0.05` |
| `--seed` | random seed, default `127` |
| `--jobs` | numba thread count, default `-1` |
| `-v`, `--verbose` | verbose logging |

## `cache` Command

The cache command currently supports one action:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

This removes profile-cache artifacts created by `mimosa profile --cache on`.

## Python API

The package exports the high-level API from [`src/mimosa/__init__.py`](src/mimosa/__init__.py):

- `compare_motifs(...)`
- `compare_one_to_many(...)`
- `create_config(...)`
- `create_many_config(...)`
- `run_comparison(...)`
- `run_one_to_many(...)`
- `create_comparator_config(...)`
- `read_model(...)`
- `scan_model(...)`
- `get_scores(...)`
- `get_frequencies(...)`
- `get_sites(...)`
- `get_pfm(...)`
- `register_model_handler(...)`
- `clear_cache(...)`

Single comparison example:

```python
from mimosa import compare_motifs

result = compare_motifs(
    "examples/pif4.meme",
    "examples/gata2.ihbcp",
    model1_type="pwm",
    model2_type="bamm",
    strategy="profile",
    sequences="examples/foreground.fa",
    promoters="examples/background.fa",
    metric="co",
    n_permutations=100,
    seed=42,
)

print(result)
```

One-vs-many example:

```python
from mimosa import compare_one_to_many

results = compare_one_to_many(
    query="examples/pif4.meme",
    targets=[
        "examples/gata2.meme",
        "examples/gata4.meme",
        "examples/foxa2.meme",
    ],
    query_type="pwm",
    target_type="pwm",
    strategy="motif",
    metric="pcc",
    n_permutations=0,
)

for row in results:
    print(row)
```

Lower-level building blocks live in:

- [`src/mimosa/api.py`](src/mimosa/api.py)
- [`src/mimosa/comparison.py`](src/mimosa/comparison.py)
- [`src/mimosa/models.py`](src/mimosa/models.py)
- [`src/mimosa/io.py`](src/mimosa/io.py)

## Custom Model Types

Custom motif families can be plugged in through `register_model_handler(...)`.

A handler must provide:

| Function | Required | Purpose |
| :--- | :--- | :--- |
| `scan(model, sequences, strand)` | yes | score sequences for one strand or `best` |
| `load(path, kwargs)` | yes | create a `GenericModel` from disk |
| `write(model, path)` | yes | serialize the model |
| `score_bounds(model)` | yes | return theoretical score bounds |
| `scan_both(model, sequences)` | optional | optimized joint `+` / `-` scan |

See [`src/mimosa/models.py`](src/mimosa/models.py) for the built-in handlers.

## Development

Useful local commands:

```bash
uv sync --group dev
uv sync --group test
uv run pytest
uv run pytest tests/test_unit.py
uv run pytest tests/test_integration.py
uv run ruff check .
uv run ruff format --check .
```

Tests use fixtures from [`tests/fixtures/`](tests/fixtures/) and sample CLI data from [`examples/`](examples/).

## Examples

The [`examples/`](examples/) directory contains ready-to-run inputs for:

- score profile comparisons
- PWM vs PWM comparisons
- BaMM vs PWM comparisons
- SiteGA vs PWM comparisons
- example shell scripts: [`examples/run.sh`](examples/run.sh) and [`examples/run.ps1`](examples/run.ps1)

## License

MIMOSA is distributed under the [MIT License](LICENSE).
