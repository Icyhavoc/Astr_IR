"""Thin, auditable adapter around the unmodified upstream ASTERIS networks."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch
from torch import nn


def default_upstream_root() -> Path:
    """Return the adjacent, read-only ASTERIS source tree shipped with this workspace."""

    return Path(__file__).resolve().parents[4] / "Asteris" / "ASTERIS_THU-main" / "asteris"


def upstream_model_path(model_name: str, upstream_root: str | Path | None = None) -> Path:
    name = model_name.lower()
    if name not in {"asteris4", "asteris8"}:
        raise ValueError("model_name must be 'asteris4' or 'asteris8'")
    root = Path(upstream_root) if upstream_root is not None else default_upstream_root()
    path = root / ("ASTERIS_net_4.py" if name == "asteris4" else "ASTERIS_net_8.py")
    if not path.is_file():
        raise FileNotFoundError(f"Upstream ASTERIS model source not found: {path}")
    return path.resolve()


def upstream_source_sha256(model_name: str, upstream_root: str | Path | None = None) -> str:
    return hashlib.sha256(upstream_model_path(model_name, upstream_root).read_bytes()).hexdigest()


def _load_module(path: Path) -> ModuleType:
    module_name = f"astr_ir_upstream_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ASTERIS source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AsterisAdapter(nn.Module):
    """Wrap ASTERIS4/8 without copying or modifying the authors' implementation.

    The upstream network already implements ``direct = input + learned_correction``.
    ``direct`` therefore preserves the authors' training semantics.  ``residual`` is
    provided only for controlled validation experiments and interprets the learned
    correction as noise: ``prediction = input - learned_correction``.
    """

    def __init__(self, network: nn.Module, model_name: str, output_mode: str = "direct") -> None:
        super().__init__()
        if output_mode not in {"direct", "residual"}:
            raise ValueError("output_mode must be 'direct' or 'residual'")
        self.network = network
        self.model_name = model_name.lower()
        self.output_mode = output_mode
        self.expected_depth = 4 if self.model_name == "asteris4" else 8
        self.spatial_divisor = 4 if self.model_name == "asteris4" else 8

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        if model_input.ndim != 5 or model_input.shape[1] != 1:
            raise ValueError("ASTERIS input must have shape (batch, 1, time, height, width)")
        if model_input.shape[2] != self.expected_depth:
            raise ValueError(
                f"{self.model_name} requires temporal depth {self.expected_depth}, "
                f"got {model_input.shape[2]}"
            )
        if any(int(size) % self.spatial_divisor for size in model_input.shape[-2:]):
            raise ValueError(
                f"{self.model_name} height and width must be divisible by {self.spatial_divisor}"
            )
        direct = self.network(model_input)
        if self.output_mode == "direct":
            return direct
        learned_correction = direct - model_input
        return model_input - learned_correction


def build_asteris_model(
    model_name: str = "asteris4",
    *,
    upstream_root: str | Path | None = None,
    f_maps: int = 24,
    num_blocks: Sequence[int] | None = None,
    num_refinement_blocks: int = 4,
    heads: Sequence[int] | None = None,
    output_mode: str = "direct",
) -> AsterisAdapter:
    """Instantiate an original ASTERIS network through a thin local adapter."""

    name = model_name.lower()
    path = upstream_model_path(name, upstream_root)
    module = _load_module(path)
    if name == "asteris4":
        cls = module.ASTERIS4
        blocks = list(num_blocks or (4, 6, 8))
        attention_heads = list(heads or (1, 2, 4))
    else:
        cls = module.ASTERIS8
        blocks = list(num_blocks or (4, 6, 6, 8))
        attention_heads = list(heads or (1, 2, 4, 8))
    network = cls(
        inp_channels=1,
        out_channels=1,
        f_maps=int(f_maps),
        num_blocks=blocks,
        num_refinement_blocks=int(num_refinement_blocks),
        heads=attention_heads,
    )
    return AsterisAdapter(network, name, output_mode=output_mode)
