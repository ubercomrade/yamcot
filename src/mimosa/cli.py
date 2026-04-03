import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from mimosa.api import create_config, run_comparison
from mimosa.cache import clear_cache
from mimosa.comparison import create_comparator_config

PROFILE_MODEL_TYPES = ["scores", "pwm", "bamm", "sitega", "dimont", "slim"]
MOTIF_MODEL_TYPES = ["pwm", "bamm", "sitega", "dimont", "slim"]
MOTALI_MODEL_TYPES = ["pwm", "sitega"]


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if verbose:
        logging.getLogger("numba").setLevel(logging.WARNING)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="MIMOSA: Compare motifs in `profile`, `motif`, and `motali` modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare precomputed score profiles directly
  mimosa profile scores_1.fasta scores_2.fasta \
    --model1-type scores --model2-type scores --metric cj

  # Compare motifs through sequence-derived profiles
  mimosa profile model1.meme model2.ihbcp \
    --model1-type pwm --model2-type bamm \
    --fasta sequences.fa --promoters promoters.fa --metric co --min-logfpr 2

  # Direct motif comparison (former tomtom-like mode)
  mimosa motif model1.meme model2.pfm \
    --model1-type pwm --model2-type pwm \
    --metric pcc --permutations 1000 --permute-rows

  # Motali comparison
  mimosa motali model1.mat model2.meme \
    --model1-type sitega --model2-type pwm \
    --fasta sequences.fa --promoters promoters.fa
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    _add_profile_parser(subparsers)
    _add_motif_parser(subparsers)
    _add_motali_parser(subparsers)
    _add_cache_parser(subparsers)

    return parser


def _add_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the profile mode parser."""
    parser = subparsers.add_parser(
        "profile",
        help="Compare motifs via score profiles: either precomputed scores or profiles generated from motif scans.",
    )
    parser.add_argument("model1", help="Path to the first input model or score-profile file.")
    parser.add_argument("model2", help="Path to the second input model or score-profile file.")

    io_group = parser.add_argument_group("Input Options")
    io_group.add_argument(
        "--model1-type",
        choices=PROFILE_MODEL_TYPES,
        required=True,
        help="Format of the first input. Choices: scores, pwm, bamm, sitega, dimont, slim.",
    )
    io_group.add_argument(
        "--model2-type",
        choices=PROFILE_MODEL_TYPES,
        required=True,
        help="Format of the second input. Choices: scores, pwm, bamm, sitega, dimont, slim.",
    )
    io_group.add_argument(
        "--fasta",
        help=(
            "Path to FASTA sequences used to scan motif inputs. "
            "If omitted and motif scanning is required, random sequences are generated."
        ),
    )
    io_group.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Number of random sequences to generate when motif scanning is required. (default: %(default)s)",
    )
    io_group.add_argument(
        "--seq-length",
        type=int,
        default=200,
        help="Length of random sequences generated for motif scanning. (default: %(default)s)",
    )
    io_group.add_argument(
        "--promoters",
        help="Path to FASTA promoter sequences used for threshold-table calibration of profile values.",
    )

    profile_group = parser.add_argument_group("Profile Comparison Options")
    profile_group.add_argument(
        "--metric",
        choices=["cj", "co", "dice", "l1sim"],
        default="cj",
        help="Profile similarity metric. Choices: cj, co, dice, l1sim. (default: %(default)s)",
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

    technical_group = parser.add_argument_group("Technical Options")
    technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    technical_group.add_argument(
        "--seed",
        type=int,
        default=127,
        help="Global random seed for reproducible stochastic steps. (default: %(default)s)",
    )
    technical_group.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs. Use -1 for all cores. (default: %(default)s)",
    )
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


def _add_motif_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the motif mode parser."""
    parser = subparsers.add_parser(
        "motif",
        help="Compare motifs directly by aligning their matrix or tensor representations.",
    )
    parser.add_argument("model1", help="Path to the first motif model file.")
    parser.add_argument("model2", help="Path to the second motif model file.")

    io_group = parser.add_argument_group("Input Options")
    io_group.add_argument(
        "--model1-type",
        choices=MOTIF_MODEL_TYPES,
        required=True,
        help="Format of the first motif. Choices: pwm, bamm, sitega, dimont, slim.",
    )
    io_group.add_argument(
        "--model2-type",
        choices=MOTIF_MODEL_TYPES,
        required=True,
        help="Format of the second motif. Choices: pwm, bamm, sitega, dimont, slim.",
    )
    io_group.add_argument(
        "--fasta",
        help=(
            "Optional FASTA sequences used for PFM reconstruction. "
            "If omitted when reconstruction is required, random sequences are generated."
        ),
    )
    io_group.add_argument(
        "--num-sequences",
        type=int,
        default=20000,
        help="Number of random sequences to generate for PFM reconstruction. (default: %(default)s)",
    )
    io_group.add_argument(
        "--seq-length",
        type=int,
        default=100,
        help="Length of random sequences for PFM reconstruction. (default: %(default)s)",
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

    technical_group = parser.add_argument_group("Technical Options")
    technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    technical_group.add_argument(
        "--seed",
        type=int,
        default=127,
        help="Global random seed for reproducible stochastic steps. (default: %(default)s)",
    )
    technical_group.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs. Use -1 for all cores. (default: %(default)s)",
    )


def _add_motali_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the motali mode parser."""
    parser = subparsers.add_parser(
        "motali",
        help="Compare motifs with the Motali scoring workflow.",
    )
    parser.add_argument("model1", help="Path to the first motif model file.")
    parser.add_argument("model2", help="Path to the second motif model file.")

    motali_group = parser.add_argument_group("Motali Options")
    motali_group.add_argument(
        "--err",
        type=float,
        default=0.002,
        help="Expected recognition rate cutoff used by Motali. (default: %(default)s)",
    )
    motali_group.add_argument(
        "--shift",
        type=int,
        default=50,
        help="Maximum motif-center shift considered by Motali. (default: %(default)s)",
    )

    io_group = parser.add_argument_group("Input Options")
    io_group.add_argument(
        "--model1-type",
        choices=MOTALI_MODEL_TYPES,
        required=True,
        help="Format of the first motif. Choices: pwm, sitega.",
    )
    io_group.add_argument(
        "--model2-type",
        choices=MOTALI_MODEL_TYPES,
        required=True,
        help="Format of the second motif. Choices: pwm, sitega.",
    )
    io_group.add_argument(
        "--fasta",
        help="Path to FASTA sequences used in the Motali comparison. Random sequences are generated if omitted.",
    )
    io_group.add_argument(
        "--promoters",
        help="Path to FASTA promoter sequences used for threshold-table calculation.",
    )
    io_group.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of random sequences to generate when --fasta is omitted. (default: %(default)s)",
    )
    io_group.add_argument(
        "--seq-length",
        type=int,
        default=200,
        help="Length of random sequences generated when --fasta is omitted. (default: %(default)s)",
    )
    io_group.add_argument(
        "--tmp-dir",
        default=".",
        help="Directory for temporary Motali intermediate files. (default: %(default)s)",
    )

    technical_group = parser.add_argument_group("Technical Options")
    technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


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

    def validate_kernel_size_range(min_kernel_size: int, max_kernel_size: int) -> None:
        """Validate kernel-size bounds used by surrogate generation."""
        if min_kernel_size <= 0 or max_kernel_size <= 0:
            logger.error("Kernel sizes must be positive integers.")
            sys.exit(1)
        if min_kernel_size > max_kernel_size:
            logger.error(
                "Invalid kernel-size range: min-kernel-size (%s) must be <= max-kernel-size (%s).",
                min_kernel_size,
                max_kernel_size,
            )
            sys.exit(1)
        first_odd = min_kernel_size if min_kernel_size % 2 == 1 else min_kernel_size + 1
        if first_odd > max_kernel_size:
            logger.error("Kernel-size range must include at least one odd value.")
            sys.exit(1)

    def validate_file(path: str, label: str) -> None:
        """Validate that a required input file exists."""
        if not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    validate_file(args.model1, "Input file")
    validate_file(args.model2, "Input file")

    if getattr(args, "fasta", None):
        validate_file(args.fasta, "FASTA file")

    if getattr(args, "promoters", None):
        validate_file(args.promoters, "Promoter FASTA file")

    if args.mode == "profile":
        validate_kernel_size_range(args.min_kernel_size, args.max_kernel_size)
        if args.min_logfpr is not None and args.min_logfpr < 0:
            logger.error("min-logfpr must be non-negative.")
            sys.exit(1)


def map_args_to_comparator_kwargs(args) -> Dict[str, Any]:
    """Map CLI arguments to comparator configuration kwargs."""
    if args.mode == "profile":
        return {
            "metric": args.metric,
            "n_permutations": args.permutations,
            "distortion_level": args.distortion,
            "n_jobs": args.jobs,
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
            "n_jobs": args.jobs,
            "seed": args.seed,
            "pfm_mode": args.pfm_mode,
        }

    if args.mode == "motali":
        return {
            "fasta_path": args.fasta,
            "tmp_directory": args.tmp_dir,
            "motali_err": args.err,
            "motali_shift": args.shift,
        }

    return {}


def build_comparison_config_from_args(args):
    """Build ComparisonConfig from parsed CLI args."""
    comparator_kwargs = map_args_to_comparator_kwargs(args)
    comparator = create_comparator_config(**comparator_kwargs)

    sequences = getattr(args, "fasta", None)
    promoters = getattr(args, "promoters", None)

    if args.mode == "motali":
        sequences = promoters or sequences

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
