"""Leak-free temporal windows and masked 3-D patch sampling for ASTERIS."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy.ndimage import shift
from torch.utils.data import Dataset

from astr_ir.noise2noise.dataset import load_detector_mask

from .preprocessing import circular_source_mask, normalize_stack, sigma_clip_stack


def relabel_manifest_for_patch_t(split_manifest: pd.DataFrame, patch_t: int) -> pd.DataFrame:
    """Ensure validation and test blocks can each hold one full 2T window.

    ASTERIS4 can use the established N2N 48/2/12/2/16 split. ASTERIS8
    needs 16 validation frames, so for an 80-frame sequence it uses
    44/2/16/2/16. The final 16-frame test block remains identical, enabling
    a paired frozen-test comparison with N2N.
    """

    manifest = split_manifest.copy()
    if int(patch_t) == 4:
        return manifest
    if int(patch_t) != 8:
        raise ValueError("patch_t must be 4 or 8")
    for sequence, group in manifest.groupby("sequence", sort=False):
        ordered = group.sort_values("frame_index")
        count = len(ordered)
        guard = max(1, int(round(0.025 * count)))
        validation = test = 2 * int(patch_t)
        training = count - validation - test - 2 * guard
        if training < 2 * int(patch_t):
            raise ValueError(f"Sequence {sequence} is too short for an isolated ASTERIS8 split")
        labels = (
            ["train"] * training
            + ["guard"] * guard
            + ["validation"] * validation
            + ["guard"] * guard
            + ["test"] * test
        )
        manifest.loc[ordered.index, "split"] = labels
    return manifest


def build_window_manifest(
    split_manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    patch_t: int = 4,
    stride: int | None = None,
) -> pd.DataFrame:
    """Create consecutive 2T-frame windows only after the frame split is frozen."""

    patch_t = int(patch_t)
    if patch_t not in {4, 8}:
        raise ValueError("patch_t must be 4 (ASTERIS4) or 8 (ASTERIS8)")
    length = 2 * patch_t
    stride = int(stride or patch_t)
    rows: list[dict] = []
    for (sequence, split), group in split_manifest.groupby(["sequence", "split"], sort=True):
        if split == "guard":
            continue
        group = group.sort_values("frame_index").reset_index(drop=True)
        if len(group) < length:
            continue
        starts = list(range(0, len(group) - length + 1, stride))
        final_start = len(group) - length
        if final_start not in starts:
            starts.append(final_start)
        for start in starts:
            window = group.iloc[start : start + length]
            indices = window["frame_index"].to_numpy(int)
            if not np.all(np.diff(indices) == 1):
                continue
            ids = window["frame_id"].astype(str).tolist()
            rows.append(
                {
                    "window_id": f"{sequence}:{split}:{indices[0]:03d}-{indices[-1]:03d}",
                    "sequence": str(sequence),
                    "split": str(split),
                    "start_index": int(indices[0]),
                    "end_index": int(indices[-1]),
                    "patch_t": patch_t,
                    "frame_ids": "|".join(ids),
                    "input_frame_ids": "|".join(ids[0::2]),
                    "target_frame_ids": "|".join(ids[1::2]),
                    # Quality-gate fallback products are valid, auditable FITS copies.
                    # Exclude them while fitting/training, but retain validation and
                    # test in full so model selection/reporting cannot cherry-pick.
                    "usable": bool(
                        str(split) in {"validation", "test"}
                        or window["upstream_applied"].astype(bool).all()
                    ),
                }
            )
    manifest = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    return manifest


def assert_window_isolation(split_manifest: pd.DataFrame, windows: pd.DataFrame) -> None:
    frame_split = split_manifest.set_index("frame_id")["split"].astype(str).to_dict()
    for row in windows.itertuples(index=False):
        ids = str(row.frame_ids).split("|")
        if any(frame_split[frame_id] != str(row.split) for frame_id in ids):
            raise ValueError(f"Window crosses a data split: {row.window_id}")
        if set(str(row.input_frame_ids).split("|")) & set(str(row.target_frame_ids).split("|")):
            raise ValueError(f"Input and target overlap in {row.window_id}")


def _source_mask_for_rows(rows: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    sources = []
    for row in rows.itertuples(index=False):
        radius_values = [getattr(row, "r_out", np.nan), 3.0 * getattr(row, "fwhm", np.nan), 8.0]
        radius = float(np.nanmax(radius_values))
        sources.append((float(row.reference_x), float(row.reference_y), radius))
    return circular_source_mask(shape, sources)


def load_registered_stack(
    rows: pd.DataFrame,
    input_root: str | Path,
    detector_mask: np.ndarray,
    normalization: Mapping[str, float] | None = None,
    *,
    sigma: float = 3.0,
    edge_width: int = 32,
    temporal_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Load, register, source-protect, clip and optionally normalize frame rows."""

    input_root = Path(input_root)
    images, validity = [], []
    for row in rows.itertuples(index=False):
        image = np.asarray(fits.getdata(input_root / row.product_path), dtype=np.float32)
        valid = ~np.asarray(detector_mask, bool) & np.isfinite(image)
        dy, dx = float(row.alignment_dy), float(row.alignment_dx)
        aligned = shift(image, (dy, dx), order=1, mode="constant", cval=np.nan, prefilter=False)
        aligned_valid = shift(
            valid.astype(np.float32), (dy, dx), order=0, mode="constant", cval=0.0, prefilter=False
        ) > 0.5
        images.append(aligned)
        validity.append(aligned_valid)
    stack = np.stack(images)
    source_mask = _source_mask_for_rows(rows, stack.shape[1:])
    clipping = sigma_clip_stack(
        stack,
        detector_mask,
        source_mask,
        sigma=sigma,
        edge_width=edge_width,
        temporal=temporal_clip,
    )
    valid = clipping.valid_mask & np.stack(validity)
    data = clipping.data
    if normalization is not None:
        data = normalize_stack(data, valid, normalization["mean"], normalization["std"])
    return data, valid, clipping


class AsterisPatchDataset(Dataset):
    """Online sampling of paired ASTERIS volumes with a bounded window cache."""

    def __init__(
        self,
        split_manifest: pd.DataFrame,
        window_manifest: pd.DataFrame,
        input_root: str | Path,
        detector_mask: np.ndarray,
        normalizations: Mapping[str, Mapping[str, float]],
        split: str,
        *,
        patch_size: int = 64,
        samples_per_epoch: int = 128,
        source_fraction: float = 0.5,
        sigma: float = 3.0,
        edge_width: int = 32,
        temporal_clip: bool = False,
        augment: bool = True,
        seed: int = 20260824,
        cache_size: int = 4,
    ) -> None:
        self.frames = split_manifest.set_index("frame_id", drop=False)
        self.windows = window_manifest.loc[
            (window_manifest["split"] == split) & window_manifest["usable"].astype(bool)
        ].reset_index(drop=True)
        if self.windows.empty:
            raise ValueError(f"No usable ASTERIS windows for split={split}")
        self.input_root = Path(input_root)
        self.detector_mask = np.asarray(detector_mask, bool)
        self.normalizations = normalizations
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.source_fraction = float(source_fraction)
        self.sigma = float(sigma)
        self.edge_width = int(edge_width)
        self.temporal_clip = bool(temporal_clip)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_window(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = str(row.window_id)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        ids = str(row.frame_ids).split("|")
        frame_rows = self.frames.loc[ids]
        data, valid, clipping = load_registered_stack(
            frame_rows,
            self.input_root,
            self.detector_mask,
            self.normalizations[str(row.sequence)],
            sigma=self.sigma,
            edge_width=self.edge_width,
            temporal_clip=self.temporal_clip,
        )
        value = (data, valid, clipping.source_mask)
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        window = self.windows.iloc[int(rng.integers(0, len(self.windows)))]
        stack, valid, source_mask = self._load_window(window)
        size, height, width = self.patch_size, stack.shape[1], stack.shape[2]
        if size > min(height, width):
            raise ValueError(f"patch_size={size} exceeds image shape {height}x{width}")
        if rng.random() < self.source_fraction and np.any(source_mask):
            yy, xx = np.nonzero(source_mask)
            pick = int(rng.integers(0, len(xx)))
            x0 = int(np.clip(xx[pick] - size // 2, 0, width - size))
            y0 = int(np.clip(yy[pick] - size // 2, 0, height - size))
        else:
            x0 = int(rng.integers(0, width - size + 1))
            y0 = int(rng.integers(0, height - size + 1))
        ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
        first, second = stack[0::2, ys, xs].copy(), stack[1::2, ys, xs].copy()
        first_valid, second_valid = valid[0::2, ys, xs], valid[1::2, ys, xs]
        loss_mask = first_valid & second_valid
        if rng.random() < 0.5:
            first, second = second, first
        if self.augment:
            k = int(rng.integers(0, 4))
            first, second, loss_mask = (
                np.rot90(first, k, axes=(-2, -1)),
                np.rot90(second, k, axes=(-2, -1)),
                np.rot90(loss_mask, k, axes=(-2, -1)),
            )
            if rng.random() < 0.5:
                first, second, loss_mask = first[..., ::-1], second[..., ::-1], loss_mask[..., ::-1]
        return {
            "input": torch.from_numpy(first[None].copy()),
            "target": torch.from_numpy(second[None].copy()),
            "loss_mask": torch.from_numpy(loss_mask[None].astype(np.float32).copy()),
            "window_id": str(window.window_id),
        }
