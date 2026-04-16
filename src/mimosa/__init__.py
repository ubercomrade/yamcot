"""Public package API."""

from mimosa.api import compare_motifs, create_config, run_comparison
from mimosa.cache import clear_cache
from mimosa.comparison import compare, create_comparator_config
from mimosa.models import (
    GenericModel,
    get_frequencies,
    get_pfm,
    get_scores,
    get_sites,
    read_model,
    register_model_handler,
    scan_model,
)

ComparisonConfig = dict
ComparatorConfig = dict

__all__ = [
    "ComparatorConfig",
    "ComparisonConfig",
    "GenericModel",
    "clear_cache",
    "compare",
    "compare_motifs",
    "create_comparator_config",
    "create_config",
    "get_frequencies",
    "get_pfm",
    "get_scores",
    "get_sites",
    "read_model",
    "register_model_handler",
    "run_comparison",
    "scan_model",
]
