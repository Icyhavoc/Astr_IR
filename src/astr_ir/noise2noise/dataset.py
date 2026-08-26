"""Deterministic frame splitting, temporal pairing, registration, and patch sampling."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation
from torch.utils.data import Dataset
from astr_ir.registration import masked_shift, science_valid


def discover_sequences(input_root: str | Path) -> tuple[str, ...]:
    """Discover sequence directories that contain background-subtracted science FITS."""

    input_root = Path(input_root)
    sequences = tuple(
        path.name
        for path in sorted(input_root.iterdir())
        if path.is_dir() and any(path.glob("background_subtracted_*.fits"))
    )
    if not sequences:
        raise FileNotFoundError(
            f"No sequence directories containing background_subtracted_*.fits under {input_root}"
        )
    return sequences


def raw_filename(product_name: str) -> str:
    return product_name.removeprefix("background_subtracted_")


def load_detector_mask(dataset_root: str | Path) -> np.ndarray:
    dataset_root = Path(dataset_root)
    dead = np.asarray(Image.open(dataset_root / "盲点表" / "DeadBlindMap.tiff")) != 0
    noise = np.asarray(Image.open(dataset_root / "盲点表" / "NoiseBlindMap.tiff")) != 0
    if dead.shape != noise.shape:
        raise ValueError("DeadBlindMap and NoiseBlindMap have different shapes")
    return dead | noise


def _robust_linear_track(index: np.ndarray, values: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    def fit_line(mask: np.ndarray) -> np.ndarray:
        x = index[mask]
        y = values[mask]
        x_center = float(np.mean(x))
        y_center = float(np.mean(y))
        denominator = float(np.sum((x - x_center) ** 2))
        slope = float(np.sum((x - x_center) * (y - y_center)) / denominator) if denominator > 0 else 0.0
        intercept = y_center - slope * x_center
        return intercept + slope * index

    valid = np.isfinite(values) & accepted
    if np.count_nonzero(valid) < 4:
        valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full_like(index, np.nanmedian(values), dtype=float)
    for _ in range(4):
        predicted = fit_line(valid)
        residual = values - predicted
        center = np.nanmedian(residual[valid])
        scale = 1.4826 * np.nanmedian(np.abs(residual[valid] - center))
        if not np.isfinite(scale) or scale <= 0:
            break
        updated = valid & (np.abs(residual - center) <= 3.5 * scale)
        if np.array_equal(updated, valid) or np.count_nonzero(updated) < 2:
            break
        valid = updated
    return fit_line(valid)


def _registration_image(path: Path, downsample: int = 4) -> np.ndarray:
    """Build a robust, high-pass thumbnail for source-independent registration."""

    image = np.asarray(fits.getdata(path), dtype=np.float32)[::downsample, ::downsample]
    finite = np.isfinite(image)
    fill = float(np.nanmedian(image)) if np.any(finite) else 0.0
    image = np.where(finite, image, fill)
    highpass = image - gaussian_filter(image, sigma=6.0, mode="nearest")
    center = float(np.median(highpass))
    scale = float(1.4826 * np.median(np.abs(highpass - center)))
    if np.isfinite(scale) and scale > 0:
        highpass = np.clip(highpass, center - 6.0 * scale, center + 6.0 * scale)
    taper = np.outer(np.hanning(highpass.shape[0]), np.hanning(highpass.shape[1])).astype(np.float32)
    return ((highpass - center) * taper).astype(np.float32)


def _phase_correlation_shifts(files: Sequence[Path], downsample: int = 4) -> np.ndarray:
    """Estimate (dy, dx) shifts when the measurement table has no usable star track."""

    reference = _registration_image(Path(files[0]), downsample=downsample)
    shifts = []
    for path in files:
        moving = _registration_image(Path(path), downsample=downsample)
        measured, error, _ = phase_cross_correlation(
            reference, moving, upsample_factor=10, normalization=None
        )
        measured = np.asarray(measured, dtype=float) * float(downsample)
        if not np.all(np.isfinite(measured)) or np.any(np.abs(measured) > 32.0) or not np.isfinite(error):
            measured = np.zeros(2, dtype=float)
        shifts.append(measured)
    shifts = np.asarray(shifts, dtype=float)
    shifts -= np.median(shifts, axis=0, keepdims=True)
    return shifts


def _split_labels(count: int) -> list[str]:
    if count < 24:
        raise ValueError("Noise2Noise splitting requires at least 24 frames per sequence")
    train_count = int(round(0.60 * count))
    validation_count = int(round(0.15 * count))
    guard = max(1, int(round(0.025 * count)))
    test_count = count - train_count - validation_count - 2 * guard
    if test_count < 4:
        raise ValueError("Not enough frames remain for a test block")
    return (
        ["train"] * train_count
        + ["guard"] * guard
        + ["validation"] * validation_count
        + ["guard"] * guard
        + ["test"] * test_count
    )


def build_split_manifest(
    input_root: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    sequences: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a chronological frame-level split before any temporal pairs or patches."""

    input_root, dataset_root, output_path = Path(input_root), Path(dataset_root), Path(output_path)
    sequences = tuple(sequences) if sequences is not None else discover_sequences(input_root)
    from astr_ir.evaluation.blind_joint import inspect_frame, register_features
    detector = load_detector_mask(dataset_root)
    rows: list[dict] = []
    for sequence in sequences:
        files = sorted((input_root / sequence).glob("background_subtracted_*.fits"))
        labels = _split_labels(len(files))
        reference = inspect_frame(files[0], detector)
        for index, (path, split) in enumerate(zip(files, labels)):
            current = reference if index == 0 else inspect_frame(path, detector)
            offset, _, registration = register_features(reference[3], reference[4], current[3], current[4])
            with fits.open(path, memmap=False) as hdul:
                header = hdul[0].header
                shape = tuple(hdul[0].data.shape)
                flicker_applied = bool(header.get("HIERARCH FLK APPL", False))
                background_applied = bool(header.get("HIERARCH BKG APPL", False))
            if len(shape) != 2:
                raise ValueError(f"Expected 2-D FITS image, got {shape}: {path}")
            rows.append(
                {
                    "frame_id": f"{sequence}:{index:03d}",
                    "sequence": sequence,
                    "frame_index": index,
                    "split": split,
                    "product_path": path.relative_to(input_root).as_posix(),
                    "product_filename": path.name,
                    "filename": raw_filename(path.name),
                    **{key: np.nan for key in ("star_id", "input_status", "input_snr", "xc", "yc", "r_ap", "r_in", "r_out", "fwhm", "track_x", "track_y", "reference_x", "reference_y")},
                    "source_measurement_available": False,
                    "registration_method": "blind_image_stars_first_frame",
                    "alignment_dx": float(offset[1]),
                    "alignment_dy": float(offset[0]),
                    **registration,
                    "flicker_applied": flicker_applied,
                    "background_applied": background_applied,
                    "upstream_applied": flicker_applied and background_applied,
                    "height": shape[0],
                    "width": shape[1],
                }
            )
    manifest = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    return manifest


def build_pair_manifest(
    split_manifest: pd.DataFrame,
    output_path: str | Path,
    lags: Sequence[int] = (2, 3, 4, 5),
) -> pd.DataFrame:
    """Pair frames only within the same sequence and already-fixed data split."""

    rows: list[dict] = []
    for (sequence, split), group in split_manifest.groupby(["sequence", "split"], sort=True):
        if split == "guard":
            continue
        group = group.sort_values("frame_index").reset_index(drop=True)
        by_index = {int(row.frame_index): row for row in group.itertuples(index=False)}
        for frame_index, first in by_index.items():
            for lag in lags:
                second = by_index.get(frame_index + int(lag))
                if second is None:
                    continue
                usable = bool(
                    split == "test"
                    or (bool(first.upstream_applied) and bool(second.upstream_applied))
                )
                rows.append(
                    {
                        "pair_id": f"{sequence}:{split}:{frame_index:03d}:{second.frame_index:03d}",
                        "sequence": sequence,
                        "split": split,
                        "frame_a": first.frame_id,
                        "frame_b": second.frame_id,
                        "lag": int(lag),
                        "usable": usable,
                        "upstream_both_applied": bool(
                            first.upstream_applied and second.upstream_applied
                        ),
                    }
                )
    pairs = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pairs


def estimate_sequence_scales(
    split_manifest: pd.DataFrame,
    input_root: str | Path,
    detector_mask: np.ndarray,
    edge_width: int = 32,
) -> dict[str, float]:
    """Fit normalization scales from training frames only."""

    input_root = Path(input_root)
    scales: dict[str, float] = {}
    training = split_manifest.loc[
        (split_manifest["split"] == "train") & split_manifest["upstream_applied"]
    ]
    for sequence, group in training.groupby("sequence"):
        frame_scales = []
        for row in group.itertuples(index=False):
            image = np.asarray(fits.getdata(input_root / row.product_path), dtype=np.float32)
            valid = ~detector_mask & np.isfinite(image)
            valid[:edge_width] = False
            valid[-edge_width:] = False
            valid[:, :edge_width] = False
            valid[:, -edge_width:] = False
            values = image[valid][::16].astype(np.float64)
            center = np.median(values)
            scale = 1.4826 * np.median(np.abs(values - center))
            if np.isfinite(scale) and scale > 0:
                frame_scales.append(scale)
        if not frame_scales:
            raise ValueError(f"Could not estimate a training scale for sequence {sequence}")
        scales[str(sequence)] = float(np.median(frame_scales))
    return scales


class PairedPatchDataset(Dataset):
    """Online aligned N2N patch sampling with a bounded full-frame cache."""

    def __init__(
        self,
        split_manifest: pd.DataFrame,
        pair_manifest: pd.DataFrame,
        input_root: str | Path,
        detector_mask: np.ndarray,
        sequence_scales: dict[str, float],
        split: str,
        patch_size: int = 128,
        samples_per_epoch: int = 512,
        source_fraction: float = 0.5,
        augment: bool = True,
        seed: int = 20260820,
        cache_size: int = 96,
    ) -> None:
        self.frames = split_manifest.set_index("frame_id", drop=False)
        self.pairs = pair_manifest.loc[
            (pair_manifest["split"] == split) & pair_manifest["usable"]
        ].reset_index(drop=True)
        if self.pairs.empty:
            raise ValueError(f"No usable Noise2Noise pairs for split={split}")
        self.input_root = Path(input_root)
        self.detector_mask = np.asarray(detector_mask, dtype=bool)
        self.sequence_scales = sequence_scales
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.source_fraction = float(source_fraction)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_frame(self, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
        if frame_id in self._cache:
            image, valid = self._cache.pop(frame_id)
            self._cache[frame_id] = (image, valid)
            return image, valid
        row = self.frames.loc[frame_id]
        image = np.asarray(fits.getdata(self.input_root / row.product_path), dtype=np.float32)
        valid = science_valid(self.input_root / row.product_path, image, self.detector_mask)
        dy, dx = float(row.alignment_dy), float(row.alignment_dx)
        aligned, aligned_valid, _, _ = masked_shift(image, valid, (dy, dx))
        center = float(np.median(aligned[aligned_valid & np.isfinite(aligned)]))
        scale = float(self.sequence_scales[str(row.sequence)])
        normalized = (aligned - center) / scale
        normalized[~aligned_valid | ~np.isfinite(normalized)] = 0.0
        result = (normalized.astype(np.float32), aligned_valid)
        self._cache[frame_id] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _crop_origin(self, rng: np.random.Generator, row: pd.Series) -> tuple[int, int]:
        height, width = int(row.height), int(row.width)
        size = self.patch_size
        if size > min(height, width):
            raise ValueError(f"patch_size={size} is larger than image {height}x{width}")
        # Uniform spatial sampling, including when a legacy manifest happens
        # to contain known-source coordinates. Catalogs are evaluation-only.
        return int(rng.integers(0, width - size + 1)), int(rng.integers(0, height - size + 1))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        pair = self.pairs.iloc[int(rng.integers(0, len(self.pairs)))]
        frame_a, frame_b = str(pair.frame_a), str(pair.frame_b)
        if rng.random() < 0.5:
            frame_a, frame_b = frame_b, frame_a
        input_image, input_valid = self._load_frame(frame_a)
        target_image, target_valid = self._load_frame(frame_b)
        frame_row = self.frames.loc[frame_a]
        x0, y0 = self._crop_origin(rng, frame_row)
        size = self.patch_size
        ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
        input_patch = input_image[ys, xs].copy()
        target_patch = target_image[ys, xs].copy()
        input_mask = input_valid[ys, xs].copy()
        target_mask = target_valid[ys, xs].copy()
        loss_mask = input_mask & target_mask
        if self.augment and rng.random() < 0.5:
            input_patch = input_patch[:, ::-1]
            target_patch = target_patch[:, ::-1]
            input_mask = input_mask[:, ::-1]
            loss_mask = loss_mask[:, ::-1]
        if self.augment and rng.random() < 0.5:
            input_patch = input_patch[::-1, :]
            target_patch = target_patch[::-1, :]
            input_mask = input_mask[::-1, :]
            loss_mask = loss_mask[::-1, :]
        model_input = np.stack([input_patch, input_mask.astype(np.float32)], axis=0).copy()
        return {
            "input": torch.from_numpy(model_input),
            "target": torch.from_numpy(target_patch[None].copy()),
            "loss_mask": torch.from_numpy(loss_mask[None].astype(np.float32).copy()),
            "pair_id": str(pair.pair_id),
        }
