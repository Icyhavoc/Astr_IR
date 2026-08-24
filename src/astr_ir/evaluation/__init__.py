"""Model-agnostic astronomical source-injection evaluation."""

from .mock_sources import (
    EvaluationConfig,
    build_empirical_psf,
    detect_sources,
    inject_sources,
    match_catalogs,
)
from .pipeline import run_mock_source_evaluation

__all__ = [
    "EvaluationConfig",
    "build_empirical_psf",
    "detect_sources",
    "inject_sources",
    "match_catalogs",
    "run_mock_source_evaluation",
]
