import argparse
import json
import logging
import sys
from typing import Any, Dict

from mimosa.api import create_config, run_comparison
from mimosa.cache import clear_cache
from mimosa.comparison import create_comparator_config
from mimosa.validation import validate_file_exists

PROFILE_MODEL_TYPES = ["scores", "pwm", "bamm", "sitega", "dimont", "slim"]
MOTIF_MODEL_TYPES = ["pwm", "bamm", "sitega", "dimont", "slim"]


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="MIMOSA: Compare motifs in `profile` and `motif` modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare precomputed score profiles directly
  mimosa profile scores_1.fasta scores_2.fasta \
    --model1-type scores --model2-type scores --metric co

  # Compare motifs through sequence-derived profiles
  mimosa profile model1.meme model2.ihbcp \
    --model1-type pwm --model2-type bamm \
    --fasta sequences.fa --metric co --min-logfpr 2

  # Direct motif comparison (former tomtom-like mode)
  mimosa motif model1.meme model2.pfm \
    --model1-type pwm --model2-type pwm \
    --metric pcc --permutations 1000 --permute-rows

        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    _add_profile_parser(subparsers)
    _add_motif_parser(subparsers)
    _add_cache_parser(subparsers)

    return parser


def _add_input_file_arguments(
    parser: argparse.ArgumentParser,
    model_types: list[str],
    first_help: str,
    second_help: str,
) -> argparse._ArgumentGroup:
    """Add required model inputs and types to one parser."""
    parser.add_argument("model1", help=first_help)
    parser.add_argument("model2", help=second_help)
    io_group = parser.add_argument_group("Input Options")
    io_group.add_argument(
        "--model1-type",
        choices=model_types,
        required=True,
        help=f"Format of the first input. Choices: {', '.join(model_types)}.",
    )
    io_group.add_argument(
        "--model2-type",
        choices=model_types,
        required=True,
        help=f"Format of the second input. Choices: {', '.join(model_types)}.",
    )
    return io_group


def _add_sequence_generation_arguments(
    io_group: argparse._ArgumentGroup,
    *,
    fasta_help: str,
    num_sequences_default: int,
    seq_length_default: int,
    promoters_help: str | None = None,
) -> None:
    """Add FASTA- and random-sequence-related arguments."""
    io_group.add_argument("--fasta", help=fasta_help)
    if promoters_help is not None:
        io_group.add_argument("--promoters", help=promoters_help)
    io_group.add_argument(
        "--num-sequences",
        type=int,
        default=num_sequences_default,
        help="Number of random sequences to generate when FASTA input is omitted. (default: %(default)s)",
    )
    io_group.add_argument(
        "--seq-length",
        type=int,
        default=seq_length_default,
        help="Length of random sequences generated when FASTA input is omitted. (default: %(default)s)",
    )


def _add_common_technical_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seed: bool = True,
    include_jobs: bool = True,
    include_cache: bool = False,
) -> argparse._ArgumentGroup:
    """Add shared technical arguments for profile/motif parsers."""
    technical_group = parser.add_argument_group("Technical Options")
    technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    if include_seed:
        technical_group.add_argument(
            "--seed",
            type=int,
            default=127,
            help="Global random seed for reproducible stochastic steps. (default: %(default)s)",
        )
    if include_jobs:
        technical_group.add_argument(
            "--jobs",
            type=int,
            default=-1,
            help="Number of parallel jobs. Use -1 for all cores. (default: %(default)s)",
        )
    if include_cache:
        technical_group.add_argument(
            "--cache",
            choices=["off", "on"],
            default="off",
            help="Enable lazy disk cache for derived profiles. (default: %(default)s)",
        )
        technical_group.add_argument(
            "--cache-dir",
            default=".mimosa-cache",
            help="Directory used for lazy profile cache files. (default: %(default)s)",
        )
    return technical_group


def _add_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the profile mode parser."""
    parser = subparsers.add_parser(
        "profile",
        help="Compare motifs via score profiles: either precomputed scores or profiles generated from motif scans.",
    )
    io_group = _add_input_file_arguments(
        parser,
        PROFILE_MODEL_TYPES,
        "Path to the first input model or score-profile file.",
        "Path to the second input model or score-profile file.",
    )
    _add_sequence_generation_arguments(
        io_group,
        fasta_help=(
            "Path to FASTA sequences used to scan motif inputs. "
            "If omitted and motif scanning is required, random sequences are generated."
        ),
        promoters_help=(
            "Optional FASTA sequences used to calibrate profile normalization. "
            "If omitted, normalization is fitted on the comparison sequences."
        ),
        num_sequences_default=1000,
        seq_length_default=200,
    )
    profile_group = parser.add_argument_group("Profile Comparison Options")
    profile_group.add_argument(
        "--metric",
        choices=["co", "dice"],
        default="co",
        help="Profile similarity metric. Choices: co, dice. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of permutations for p-value estimation. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--distortion",
        type=float,
        default=0.4,
        help="Surrogate-profile distortion level in the 0.0-1.0 range. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--search-range",
        type=int,
        default=10,
        help="Maximum alignment offset explored between profiles. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--min-kernel-size",
        type=int,
        default=3,
        help="Minimum surrogate convolution kernel size. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--max-kernel-size",
        type=int,
        default=11,
        help="Maximum surrogate convolution kernel size. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--min-logfpr",
        type=float,
        default=None,
        help="Ignore aligned positions only when both profile values are below this logFPR threshold.",
    )

    _add_common_technical_arguments(parser, include_cache=True)


def _add_motif_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the motif mode parser."""
    parser = subparsers.add_parser(
        "motif",
        help="Compare motifs directly by aligning their matrix or tensor representations.",
    )
    io_group = _add_input_file_arguments(
        parser,
        MOTIF_MODEL_TYPES,
        "Path to the first motif model file.",
        "Path to the second motif model file.",
    )
    _add_sequence_generation_arguments(
        io_group,
        fasta_help=(
            "Optional FASTA sequences used for PFM reconstruction. "
            "If omitted when reconstruction is required, random sequences are generated."
        ),
        num_sequences_default=20000,
        seq_length_default=100,
    )

    motif_group = parser.add_argument_group("Motif Comparison Options")
    motif_group.add_argument(
        "--metric",
        choices=["pcc", "ed", "cosine"],
        default="pcc",
        help="Column-wise comparison metric. Choices: pcc, ed, cosine. (default: %(default)s)",
    )
    motif_group.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of Monte Carlo permutations for p-value estimation. (default: %(default)s)",
    )
    motif_group.add_argument(
        "--permute-rows",
        action="store_true",
        help="Shuffle matrix rows in addition to positions during permutation testing.",
    )
    motif_group.add_argument(
        "--pfm-mode",
        action="store_true",
        help="Force sequence-driven PFM reconstruction before direct motif comparison.",
    )
    motif_group.add_argument(
        "--pfm-top-fraction",
        type=float,
        default=0.05,
        help="Fraction of top-scoring reconstructed sites used for cross-type PFM comparison. (default: %(default)s)",
    )

    _add_common_technical_arguments(parser)


def _add_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the cache management parser."""
    parser = subparsers.add_parser("cache", help="Manage lazy profile cache artifacts.")
    nested = parser.add_subparsers(dest="cache_action", required=True)

    clear_parser = nested.add_parser("clear", help="Remove all cached profile artifacts.")
    clear_parser.add_argument(
        "--cache-dir",
        default=".mimosa-cache",
        help="Directory containing cached profile artifacts. (default: %(default)s)",
    )
    clear_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def validate_inputs(args) -> None:
    """Validate input files and parameters."""
    logger = logging.getLogger(__name__)

    if args.mode == "cache":
        return

    file_checks = [
        (args.model1, "Input file"),
        (args.model2, "Input file"),
    ]
    if getattr(args, "fasta", None):
        file_checks.append((args.fasta, "FASTA file"))
    if getattr(args, "promoters", None):
        file_checks.append((args.promoters, "Promoter FASTA file"))

    try:
        for path, label in file_checks:
            validate_file_exists(path, label)
        if args.mode in {"profile", "motif"}:
            create_comparator_config(**map_args_to_comparator_kwargs(args))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def map_args_to_comparator_kwargs(args) -> Dict[str, Any]:
    """Map CLI arguments to comparator configuration kwargs."""
    if args.mode == "profile":
        return {
            "metric": args.metric,
            "n_permutations": args.permutations,
            "distortion_level": args.distortion,
            "numba_threads": args.jobs,
            "seed": args.seed,
            "search_range": args.search_range,
            "min_kernel_size": args.min_kernel_size,
            "max_kernel_size": args.max_kernel_size,
            "min_logfpr": args.min_logfpr,
            "cache_mode": args.cache,
            "cache_dir": args.cache_dir,
        }

    if args.mode == "motif":
        return {
            "metric": args.metric,
            "n_permutations": args.permutations,
            "permute_rows": args.permute_rows,
            "numba_threads": args.jobs,
            "seed": args.seed,
            "pfm_mode": args.pfm_mode,
            "pfm_top_fraction": args.pfm_top_fraction,
        }

    return {}


def build_comparison_config_from_args(args):
    """Build ComparisonConfig from parsed CLI args."""
    comparator_kwargs = map_args_to_comparator_kwargs(args)
    comparator = create_comparator_config(**comparator_kwargs)

    sequences = getattr(args, "fasta", None)
    promoters = getattr(args, "promoters", None)

    return create_config(
        model1=args.model1,
        model2=args.model2,
        model1_type=args.model1_type,
        model2_type=args.model2_type,
        strategy=args.mode,
        sequences=sequences,
        promoters=promoters,
        num_sequences=getattr(args, "num_sequences", 1000),
        seq_length=getattr(args, "seq_length", 200),
        seed=getattr(args, "seed", 127),
        comparator=comparator,
    )


def run_comparison_from_args(args) -> None:
    """Run comparison with parsed CLI arguments."""
    logger = logging.getLogger(__name__)
    logger.info("Running comparison in mode: %s", args.mode)

    try:
        config = build_comparison_config_from_args(args)
        result = run_comparison(config)
        logger.info("Comparison completed successfully")
        print(json.dumps(result))
    except Exception as exc:
        logger.error("Comparison execution failed: %s", exc)
        raise


def run_cache_command_from_args(args) -> None:
    """Run a cache maintenance command."""
    logger = logging.getLogger(__name__)

    if args.cache_action == "clear":
        removed = clear_cache(args.cache_dir)
        logger.info("Cleared cache directory '%s' (%s entries removed).", args.cache_dir, removed)
        print(json.dumps({"cache_dir": args.cache_dir, "removed": removed}))
        return

    raise ValueError(f"Unknown cache action: {args.cache_action}")


def main_cli() -> None:
    """Main CLI entry point."""
    parser = create_arg_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    setup_logging(args.verbose)
    validate_inputs(args)
    if args.mode == "cache":
        run_cache_command_from_args(args)
    else:
        run_comparison_from_args(args)


if __name__ == "__main__":
    main_cli()
