import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from mimosa.pipeline import run_pipeline


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if verbose:
        logging.getLogger("numba").setLevel(logging.WARNING)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "MIMOSA: Compare motifs using three distinct approaches - profile, motif, and tomtom-like"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Profile-based comparison
   mimosa profile scores_1.fasta scores_2.fasta \\
     --metric corr --permutations 1000 --distortion 0.5

   # Motif-based comparison with PWM models
   mimosa motif model1.meme model2.pfm --model1-type pwm --model2-type pwm \\
     --fasta sequences.fa --metric co --permutations 1000 \\
     --distortion 0.3

   # Motif-based comparison with BAMM models
   mimosa motif model1.hbcp model2.ihbcp --model1-type bamm --model2-type bamm \\
     --fasta sequences.fa --promoters promoters.fa --search-range 15 \\
     --min-kernel-size 5 --max-kernel-size 15 --jobs 4 --seed 42

   # Motali comparison with SiteGA models
   mimosa motali model1.mat model2.meme --model1-type sitega --model2-type pwm \\
     --fasta sequences.fa --promoters promoters.fa \\
     --tmp-dir . --num-sequences 5000 --seq-length 150

   # TomTom-like comparison with PWM models
   mimosa tomtom-like model1.meme model2.pfm --model1-type pwm --model2-type pwm \\
     --metric pcc --permutations 1000 --permute-rows \\
     --jobs 8 --seed 123

   # TomTom-like comparison with BAMM models using PFM mode
   mimosa tomtom-like model1.hbcp model2.ihbcp --model1-type bamm --model2-type bamm \\
     --pfm-mode --num-sequences 10000 --seq-length 120 --metric ed \\
     --permutations 500 --permute-rows
         """,
    )

    # Create subparsers for the three main modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    # Score-based comparison subcommand
    profile_parser = subparsers.add_parser(
        "profile", help="Compare motifs based on pre-calculated score profiles (uses DataComparator engine)."
    )
    profile_parser.add_argument("profile1", help="Path to the first profile file containing pre-calculated scores.")
    profile_parser.add_argument("profile2", help="Path to the second profile file containing pre-calculated scores.")

    profile_group = profile_parser.add_argument_group("Profile Comparator Options")
    profile_group.add_argument(
        "--metric",
        choices=["cj", "co", "corr"],
        default="cj",
        help=(
            "Similarity metric for comparing frequency profiles. "
            "Choices: cj (Continuous Jaccard), co (Continuous Overlap), "
            "corr (Pearson Correlation). (default: %(default)s)"
        ),
    )
    profile_group.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of permutations to perform for p-value calculation. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--distortion",
        type=float,
        default=0.4,
        help=(
            "Distortion level (0.0-1.0) applied to kernels during surrogate data generation. "
            "Higher values increase variance in the null model. Used for cj and co options. "
            "(default: %(default)s)"
        ),
    )
    profile_group.add_argument(
        "--search-range",
        type=int,
        default=10,
        help="Maximum offset (shift) range to explore when aligning profiles. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--min-kernel-size",
        type=int,
        default=3,
        help=(
            "Minimum kernel size for convolution during surrogate generation. "
            "Used for cj and co options. (default: %(default)s)"
        ),
    )
    profile_group.add_argument(
        "--max-kernel-size",
        type=int,
        default=11,
        help=(
            "Maximum kernel size for convolution during surrogate generation. "
            "Used for cj and co options. (default: %(default)s)"
        ),
    )

    profile_technical_group = profile_parser.add_argument_group("Technical Options")
    profile_technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to standard output for detailed execution tracking.",
    )
    profile_technical_group.add_argument(
        "--seed",
        type=int,
        help=(
            "Set a global random seed for reproducible results in stochastic operations "
            "(e.g., permutations, surrogate generation)."
        ),
    )
    profile_technical_group.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. Set to -1 to use all available CPU cores. (default: %(default)s)",
    )

    # Motif scan-based comparison subcommand
    motif_parser = subparsers.add_parser(
        "motif", help="Compare motifs by calculating scores derived from scanning sequences with models."
    )
    motif_parser.add_argument("model1", help="Path to the first motif model file.")
    motif_parser.add_argument("model2", help="Path to the second motif model file.")

    # Input/Output Options for sequence parser
    motif_io_group = motif_parser.add_argument_group("Input/Output Options")
    motif_io_group.add_argument(
        "--model1-type",
        choices=["pwm", "bamm", "sitega"],
        required=True,
        help="Format of the first model. Choices: pwm, bamm, sitega.",
    )
    motif_io_group.add_argument(
        "--model2-type",
        choices=["pwm", "bamm", "sitega"],
        required=True,
        help="Format of the second model. Choices: pwm, bamm, sitega.",
    )
    motif_io_group.add_argument(
        "--fasta",
        help=(
            "Path to a FASTA file containing target sequences for comparison. "
            "If omitted, random sequences are generated."
        ),
    )
    motif_io_group.add_argument(
        "--promoters",
        help=(
            "Path to a FASTA file containing promoter sequences, required for "
            "calculating threshold tables in Motali comparisons."
        ),
    )
    motif_io_group.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Number of random sequences to generate if --fasta is not provided. (default: %(default)s)",
    )
    motif_io_group.add_argument(
        "--seq-length",
        type=int,
        default=200,
        help="Length of each random sequence to generate if --fasta is not provided. (default: %(default)s)",
    )

    # Motif / Data Comparator options
    motif_group = motif_parser.add_argument_group("Motif Comparator Options")
    motif_group.add_argument(
        "--metric",
        choices=["cj", "co", "corr"],
        default="cj",
        help=(
            "Similarity metric for comparing frequency profiles. "
            "Choices: cj (Continuous Jaccard), co (Continuous Overlap), "
            "corr (Pearson Correlation). (default: %(default)s)"
        ),
    )
    motif_group.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of permutations to perform for p-value calculation. (default: %(default)s)",
    )
    motif_group.add_argument(
        "--distortion",
        type=float,
        default=0.4,
        help=(
            "Distortion level (0.0-1.0) applied to kernels during surrogate data generation. "
            "Higher values increase variance in the null model. Used for cj and co options. "
            "(default: %(default)s)"
        ),
    )
    motif_group.add_argument(
        "--search-range",
        type=int,
        default=10,
        help="Maximum offset (shift) range to explore when aligning profiles. (default: %(default)s)",
    )
    motif_group.add_argument(
        "--min-kernel-size",
        type=int,
        default=3,
        help=(
            "Minimum kernel size for convolution during surrogate generation. "
            "Used for cj and co options. (default: %(default)s)"
        ),
    )
    motif_group.add_argument(
        "--max-kernel-size",
        type=int,
        default=11,
        help=(
            "Maximum kernel size for convolution during surrogate generation. "
            "Used for cj and co options. (default: %(default)s)"
        ),
    )

    motif_technical_group = motif_parser.add_argument_group("Technical Options")
    motif_technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to standard output for detailed execution tracking.",
    )
    motif_technical_group.add_argument(
        "--seed",
        type=int,
        help=(
            "Set a global random seed for reproducible results in stochastic operations "
            "(e.g., permutations, surrogate generation)."
        ),
    )
    motif_technical_group.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. Set to -1 to use all available CPU cores. (default: %(default)s)",
    )

    # Motali
    motali_parser = subparsers.add_parser(
        "motali", help="Compare motifs by calculating PRC AUC derived from scanning sequences with models."
    )

    # Motali-specific options
    motali_group = motali_parser.add_argument_group("Motali Options")

    motali_group.add_argument("model1", help="Path to the first motif model file.")
    motali_group.add_argument("model2", help="Path to the second motif model file.")

    # Input/Output Options for sequence parser
    motali_io_group = motali_parser.add_argument_group("Input/Output Options")
    motali_io_group.add_argument(
        "--model1-type",
        choices=["pwm", "sitega"],
        required=True,
        help="Format of the first model. Choices: pwm, bamm, sitega.",
    )
    motali_io_group.add_argument(
        "--model2-type",
        choices=["pwm", "sitega"],
        required=True,
        help="Format of the second model. Choices: pwm, bamm, sitega.",
    )
    motali_io_group.add_argument(
        "--fasta",
        help=(
            "Path to a FASTA file containing target sequences for comparison. "
            "If omitted, random sequences are generated."
        ),
    )
    motali_io_group.add_argument(
        "--promoters",
        help=(
            "Path to a FASTA file containing promoter sequences, required for "
            "calculating threshold tables in Motali comparisons."
        ),
    )
    motali_io_group.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of random sequences to generate if --fasta is not provided. (default: %(default)s)",
    )
    motali_io_group.add_argument(
        "--seq-length",
        type=int,
        default=200,
        help="Length of each random sequence to generate if --fasta is not provided. (default: %(default)s)",
    )

    motali_io_group.add_argument(
        "--tmp-dir",
        default=".",
        help=(
            "Directory path for storing temporary intermediate files generated "
            "during Motali execution. (default: %(default)s)"
        ),
    )

    motali_technical_group = motali_parser.add_argument_group("Technical Options")
    motali_technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to standard output for detailed execution tracking.",
    )

    # TomTom-like comparison subcommand
    tomtom_parser = subparsers.add_parser(
        "tomtom-like", help="Compare motifs by direct matrix comparison (uses TomTomComparator engine)."
    )
    tomtom_parser.add_argument("model1", help="Path to the first motif model file.")
    tomtom_parser.add_argument("model2", help="Path to the second motif model file.")

    # Input/Output Options for tomtom parser
    tomtom_io_group = tomtom_parser.add_argument_group("Input/Output Options")
    tomtom_io_group.add_argument(
        "--model1-type",
        choices=["pwm", "bamm", "sitega"],
        required=True,
        help="Format of the first model. Choices: pwm, bamm, sitega.",
    )
    tomtom_io_group.add_argument(
        "--model2-type",
        choices=["pwm", "bamm", "sitega"],
        required=True,
        help="Format of the second model. Choices: pwm, bamm, sitega.",
    )

    # TomTom-specific options
    tomtom_options_group = tomtom_parser.add_argument_group("TomTom Options")
    tomtom_options_group.add_argument(
        "--metric",
        choices=["pcc", "ed", "cosine"],
        default="pcc",
        help=(
            "Metric for column-wise motif comparison. "
            "Choices: pcc (Pearson Correlation Coefficient), ed (Euclidean Distance), "
            "cosine (Cosine Similarity). (default: %(default)s)"
        ),
    )
    tomtom_options_group.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of Monte Carlo permutations for p-value estimation. (default: %(default)s)",
    )
    tomtom_options_group.add_argument(
        "--permute-rows",
        action="store_true",
        help=(
            "If set, shuffles values within columns during permutation, destroying "
            "nucleotide dependencies. Default behavior shuffles only columns (positions)."
        ),
    )
    tomtom_options_group.add_argument(
        "--pfm-mode",
        action="store_true",
        help=(
            "If set, a Position Frequency Matrix (PFM) is derived for the model motifs "
            "by scanning sequences and constructing the PFM based on the top 5% of "
            "predicted binding sites"
        ),
    )

    tomtom_options_group.add_argument(
        "--num-sequences",
        type=int,
        default=20000,
        help="Number of random sequences to generate if --pfm-mode is used. (default: %(default)s)",
    )
    tomtom_options_group.add_argument(
        "--seq-length",
        type=int,
        default=100,
        help="Length of each random sequence to generate if --pfm-mode is used. (default: %(default)s)",
    )

    tomtom_technical_group = tomtom_parser.add_argument_group("Technical Options")
    tomtom_technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to standard output for detailed execution tracking.",
    )
    tomtom_technical_group.add_argument(
        "--seed",
        type=int,
        help=(
            "Set a global random seed for reproducible results in stochastic operations "
            "(e.g., permutations, surrogate generation)."
        ),
    )
    tomtom_technical_group.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. Set to -1 to use all available CPU cores. (default: %(default)s)",
    )

    return parser


def validate_inputs(args) -> None:
    """Validate input files and parameters."""
    logger = logging.getLogger(__name__)
    # Validate mode-specific inputs
    if args.mode == "profile":
        if not os.path.exists(args.profile1):
            logger.error(f"Profile file not found: {args.profile1}")
            sys.exit(1)
        if not os.path.exists(args.profile2):
            logger.error(f"Profile file not found: {args.profile2}")
            sys.exit(1)

    elif args.mode in ["motif", "motali"]:
        suffix_1 = ""
        suffix_2 = ""
        if args.model1_type == "bamm":
            suffix_1 = ".ihbcp"
        if args.model2_type == "bamm":
            suffix_2 = ".ihbcp"

        if not os.path.exists(args.model1 + suffix_1):
            logger.error(f"Model file not found: {args.model1}")
            sys.exit(1)
        if not os.path.exists(args.model2 + suffix_2):
            logger.error(f"Model file not found: {args.model2}")
            sys.exit(1)
        if args.fasta and not os.path.exists(args.fasta):
            logger.error(f"FASTA file not found: {args.fasta}")
            sys.exit(1)
        if args.promoters and not os.path.exists(args.promoters):
            logger.error(f"Promoter threshold file not found: {args.promoters}")
            sys.exit(1)

    elif args.mode == "tomtom-like":
        suffix_1 = ""
        suffix_2 = ""
        if args.model1_type == "bamm":
            suffix_1 = ".ihbcp"
        if args.model2_type == "bamm":
            suffix_2 = ".ihbcp"

        if not os.path.exists(args.model1 + suffix_1):
            logger.error(f"Model file not found: {args.model1}")
            sys.exit(1)
        if not os.path.exists(args.model2 + suffix_2):
            logger.error(f"Model file not found: {args.model2}")
            sys.exit(1)


def map_args_to_pipeline_kwargs(args) -> Dict[str, Any]:
    """Map CLI arguments to pipeline keyword arguments."""
    kwargs = {}

    if args.mode == "tomtom-like":
        kwargs.update(
            {
                "metric": getattr(args, "metric", "pcc"),
                "n_permutations": getattr(args, "permutations", 1000),
                "permute_rows": getattr(args, "permute_rows", False),
                "n_jobs": getattr(args, "jobs", -1),
                "seed": getattr(args, "seed", None),
                "pfm_mode": getattr(args, "pfm_mode", False),
                "comparator": "tomtom",
            }
        )
    elif args.mode == "motali":
        kwargs.update({"fasta_path": getattr(args, "fasta", None), "tmp_directory": getattr(args, "tmp_dir", ".")})
    elif args.mode == "motif":
        kwargs.update(
            {
                "metric": getattr(args, "metric", "cj"),
                "n_permutations": getattr(args, "permutations", 1000),
                "distortion_level": getattr(args, "distortion", 0.4),
                "n_jobs": getattr(args, "jobs", -1),
                "permute_rows": getattr(args, "permute_rows", False),
                "pfm_mode": getattr(args, "pfm_mode", False),
                "seed": getattr(args, "seed", None),
                "search_range": getattr(args, "search_range", 10),
                "min_kernel_size": getattr(args, "min_kernel_size", 3),
                "max_kernel_size": getattr(args, "max_kernel_size", 11),
            }
        )
    elif args.mode == "profile":
        kwargs.update(
            {
                "metric": getattr(args, "metric", "cj"),
                "n_permutations": getattr(args, "permutations", 1000),
                "distortion_level": getattr(args, "distortion", 0.4),
                "n_jobs": getattr(args, "jobs", -1),
                "seed": getattr(args, "seed", None),
                "search_range": getattr(args, "search_range", 10),
                "min_kernel_size": getattr(args, "min_kernel_size", 3),
                "max_kernel_size": getattr(args, "max_kernel_size", 11),
            }
        )

    return kwargs


def main_cli():
    """Main CLI entry point."""
    # Parse arguments
    parser = create_arg_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate inputs
    validate_inputs(args)

    # Prepare pipeline arguments based on mode
    if args.mode == "profile":
        # Profile-based comparison
        model1_path = args.profile1
        model2_path = args.profile2
        seq_source1 = None
        seq_source2 = None

    elif args.mode in ["motif", "motali"]:
        # Sequence-based comparison
        model1_path = args.model1
        model2_path = args.model2
        seq_source1 = args.fasta
        seq_source2 = args.promoters

    elif args.mode == "tomtom-like":
        model1_path = args.model1
        model2_path = args.model2
        seq_source1 = None
        seq_source2 = None

    else:
        logger = logging.getLogger(__name__)
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Map CLI arguments to pipeline kwargs
    pipeline_kwargs = map_args_to_pipeline_kwargs(args)

    if args.verbose:
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"UniMotifComparator Pipeline - {args.mode.capitalize()} Mode")
        logger.info("=" * 60)
        logger.info(f"Comparison method: {args.mode}")
        logger.info(f"Model 1: {model1_path}")
        logger.info(f"Model 2: {model2_path}")
        if args.mode in ["motif", "motali", "tomtom-like"]:
            logger.info(f"Model 1 type: {getattr(args, 'model1_type', 'N/A')}")
            logger.info(f"Model 2 type: {getattr(args, 'model2_type', 'N/A')}")
            if args.mode in ["motif", "motali"]:
                logger.info(f"Sequences: {args.fasta or 'Generated internally'}")
        logger.info("=" * 60)

    try:
        # Run the pipeline
        comparison_type = args.mode
        result = run_pipeline(
            model1_path=model1_path,
            model2_path=model2_path,
            model1_type=getattr(args, "model1_type", ""),
            model2_type=getattr(args, "model2_type", ""),
            comparison_type=comparison_type,
            seq_source1=seq_source1,
            seq_source2=seq_source2,
            num_sequences=getattr(args, "num_sequences", 1000),
            seq_length=getattr(args, "seq_length", 200),
            **pipeline_kwargs,
        )

        logger = logging.getLogger(__name__)
        json_string = json.dumps(result)
        print(json_string)

    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
