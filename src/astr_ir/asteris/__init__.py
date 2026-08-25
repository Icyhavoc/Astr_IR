"""ASTERIS self-supervised spatiotemporal denoising.

The public training API is loaded lazily so lightweight analysis utilities do
not initialize PyTorch and its OpenMP runtime merely by importing this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AsterisAdapter",
    "AsterisConfig",
    "build_asteris_model",
    "calibrate_and_finalize",
    "prepare_manifests",
    "run_inference",
    "train_model",
]


def __getattr__(name: str) -> Any:
    if name in {"AsterisAdapter", "build_asteris_model"}:
        return getattr(import_module(".model", __name__), name)
    if name in {
        "AsterisConfig",
        "calibrate_and_finalize",
        "prepare_manifests",
        "run_inference",
        "train_model",
    }:
        return getattr(import_module(".processor", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
