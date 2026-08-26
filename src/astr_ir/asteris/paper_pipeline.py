"""Paper-faithful ASTERIS8 training and multi-exposure coadd inference.

This module follows the released ASTERIS workflow: sixteen independent
exposures form interlaced eight-frame input/target volumes, temporal and global
3-sigma clipping are applied before training, each half is median-centred, the
official ASTERIS8 network is optimized with stack SmoothL1 plus temporal-mean
MSE, and inference collapses the eight denoised slices into one science coadd.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits
from astr_ir.registration import masked_shift, science_valid
from scipy.stats import sigmaclip
from torch.utils.data import DataLoader, Dataset

from astr_ir.noise2noise.dataset import build_split_manifest, load_detector_mask
from astr_ir.noise2noise.processor import neighbor_difference_noise
from astr_ir.dq import build_dq, write_fits_with_dq

from .dataset import relabel_manifest_for_patch_t
from .inference import infer_volume
from .model import build_asteris_model, upstream_source_sha256
from .preprocessing import fill_invalid_with_temporal_mean


def _masked_mean(stack: np.ndarray, valid: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean with an explicit validity mask and no all-NaN edge warnings."""

    count = np.sum(valid, axis=axis)
    total = np.sum(np.where(valid, stack, 0.0), axis=axis, dtype=np.float64)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


@dataclass(frozen=True)
class PaperAsterisConfig:
    model: str = "asteris8"
    patch_t: int = 8
    patch_size: int = 128
    f_maps: int = 24
    num_refinement_blocks: int = 4
    scale_factor: float = 4.0
    temporal_sigma: float = 3.0
    global_sigma: float = 3.0
    mse_select: bool = True
    epochs: int = 20
    samples_per_sequence: int = 64
    validation_samples_per_sequence: int = 16
    batch_size: int = 1
    learning_rate: float = 1.5e-4
    weight_decay: float = 1e-4
    scheduler_t_max: int = 2_000_000
    loss_scale: float = 1e6
    patience: int = 6
    seed: int = 20260825
    inference_tile_size: int = 128
    inference_overlap: int = 16
    amp: bool = True
    initialize_from_official: bool = True
    official_checkpoint: str = (
        "D:/Astr_IR/Asteris/ASTERIS_THU-main/pth/ASTERIS8_nrcshort/ASTERIS8_nrcshort.pth"
    )

    def validate(self) -> None:
        if self.model.lower() != "asteris8" or self.patch_t != 8:
            raise ValueError("Paper workflow requires the official eight-slice ASTERIS8 model")
        if self.patch_size % 8 or self.inference_tile_size % 8:
            raise ValueError("ASTERIS8 spatial sizes must be divisible by 8")
        if self.patch_size < 32 or self.inference_tile_size < 32:
            raise ValueError("ASTERIS8 spatial sizes must be at least 32 pixels")
        if min(self.epochs, self.samples_per_sequence, self.batch_size) < 1:
            raise ValueError("Training sizes and epochs must be positive")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_registered_stack(
    rows: pd.DataFrame, input_root: Path, detector_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    images, validity = [], []
    for row in rows.itertuples(index=False):
        image = np.asarray(fits.getdata(input_root / row.product_path), dtype=np.float32)
        valid = science_valid(input_root / row.product_path, image, detector_mask)
        dy, dx = float(row.alignment_dy), float(row.alignment_dx)
        registered, registered_valid, _, _ = masked_shift(image, valid, (dy, dx))
        images.append(registered)
        validity.append(registered_valid & np.isfinite(registered))
    return np.stack(images), np.stack(validity)


def paper_sigma_clip(
    stack: np.ndarray,
    valid: np.ndarray,
    *,
    temporal_sigma: float = 3.0,
    global_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Reproduce released temporal clipping plus global 3-D clipping."""

    work = np.where(valid, np.asarray(stack, np.float32), np.nan).astype(np.float32)
    temporal_outliers = np.zeros_like(valid, dtype=bool)
    if temporal_sigma > 0:
        # Registration creates fully masked borders.  NumPy warns for those
        # expected pixels even though they remain invalid throughout.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            center = np.nanmedian(work, axis=0)
            spread = np.nanstd(work, axis=0)
        usable = np.isfinite(spread) & (spread > 0)
        lower = center - float(temporal_sigma) * spread
        upper = center + float(temporal_sigma) * spread
        temporal_outliers = valid & usable[None] & ((work <= lower[None]) | (work >= upper[None]))
        work[temporal_outliers] = np.nan
    valid_after_temporal = valid & np.isfinite(work)
    values = work[valid_after_temporal].astype(np.float64)
    if values.size < 32:
        raise ValueError("Too few valid voxels for ASTERIS sigma clipping")
    if global_sigma > 0:
        _, low, high = sigmaclip(values, low=global_sigma, high=global_sigma)
        clipped = work.copy()
        clipped[valid_after_temporal] = np.clip(values, low, high).astype(np.float32)
    else:
        low, high = float(np.min(values)), float(np.max(values))
        clipped = work.copy()
    residual = np.where(valid_after_temporal, work - clipped, 0.0).astype(np.float32)
    metrics = {
        "temporal_clipped_fraction": float(np.count_nonzero(temporal_outliers) / max(valid.size, 1)),
        "global_clipped_fraction": float(
            np.count_nonzero(np.abs(residual) > 0) / max(np.count_nonzero(valid_after_temporal), 1)
        ),
        "clip_low": float(low),
        "clip_high": float(high),
    }
    return clipped.astype(np.float32), residual, valid_after_temporal, metrics


def _mse_order(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    reference = _masked_mean(stack, valid)
    scores = []
    for image, mask in zip(stack, valid):
        use = mask & np.isfinite(reference)
        scores.append(float(np.mean((image[use] - reference[use]) ** 2)) if np.any(use) else np.inf)
    return np.argsort(scores)


def _normalize_stack(
    stack: np.ndarray, valid: np.ndarray, scale_factor: float
) -> tuple[np.ndarray, float, float]:
    values = stack[valid & np.isfinite(stack)].astype(np.float64)
    center = float(np.median(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid ASTERIS normalization standard deviation")
    normalized = ((stack - center) / std / float(scale_factor) + 1.0).astype(np.float32)
    normalized[~valid | ~np.isfinite(normalized)] = 0.0
    return normalized, center, std


def _save_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path.as_posix()


def prepare_paper_dataset(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    sequences: Sequence[str],
    config: PaperAsterisConfig = PaperAsterisConfig(),
    frozen_manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze splits and materialize paper-style normalized training stacks."""

    config.validate()
    input_root, dataset_root, output_root = Path(input_root), Path(dataset_root), Path(output_root)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    if frozen_manifest is None:
        split = build_split_manifest(
            input_root,
            dataset_root,
            manifest_root / "split_manifest.csv",
            sequences=tuple(sequences),
        )
        split = relabel_manifest_for_patch_t(split, 8)
    else:
        # A newly generated image-only manifest can be shared by background
        # ablations: freeze both frame membership and registration across arms.
        split = frozen_manifest.copy(deep=True)
        required = {'frame_id', 'sequence', 'frame_index', 'split', 'product_path',
                    'filename', 'alignment_dx', 'alignment_dy'}
        if not required.issubset(split.columns) or split.frame_id.duplicated().any():
            raise ValueError('Invalid frozen experiment manifest')
        if set(split.sequence.astype(str)) != set(map(str, sequences)):
            raise ValueError('Frozen sequences differ from requested sequences')
        if not split.split.isin(['train', 'validation', 'test', 'guard']).all():
            raise ValueError('Unknown frozen split')
        if not np.isfinite(split[['alignment_dx', 'alignment_dy']].to_numpy(float)).all():
            raise ValueError('Invalid frozen registration')
        for value in split.product_path:
            resolved = (input_root / value).resolve()
            if not resolved.is_relative_to(input_root.resolve()) or not resolved.is_file():
                raise ValueError('Missing or unsafe frozen product path')
    split.to_csv(manifest_root / "split_manifest.csv", index=False, encoding="utf-8-sig")
    detector_mask = load_detector_mask(dataset_root)
    stack_rows = []
    for (sequence, split_name), group in split.groupby(["sequence", "split"], sort=True):
        if split_name == "guard":
            continue
        group = group.sort_values("frame_index").reset_index(drop=True)
        physical, valid = _load_registered_stack(group, input_root, detector_mask)
        clipped, clip_residual, valid, clip_metrics = paper_sigma_clip(
            physical,
            valid,
            temporal_sigma=config.temporal_sigma,
            global_sigma=config.global_sigma,
        )
        order = _mse_order(clipped, valid) if split_name == "train" and config.mse_select else np.arange(len(group))
        group = group.iloc[order].reset_index(drop=True)
        clipped, clip_residual, valid = clipped[order], clip_residual[order], valid[order]
        normalized, center, std = _normalize_stack(clipped, valid, config.scale_factor)
        stem = f"{sequence}_{split_name}"
        data_path = output_root / "paper_stacks" / f"{stem}_normalized.npy"
        valid_path = output_root / "paper_stacks" / f"{stem}_valid.npy"
        _save_array(data_path, normalized)
        _save_array(valid_path, valid.astype(np.uint8))
        stack_rows.append(
            {
                "sequence": str(sequence),
                "split": str(split_name),
                "frames": len(group),
                "frame_ids": "|".join(group.frame_id.astype(str)),
                "normalized_path": data_path.relative_to(output_root).as_posix(),
                "valid_path": valid_path.relative_to(output_root).as_posix(),
                "normalization_center": center,
                "normalization_std": std,
                "scale_factor": config.scale_factor,
                "mse_ordered": bool(split_name == "train" and config.mse_select),
                **clip_metrics,
            }
        )
    stacks = pd.DataFrame(stack_rows)
    stacks.to_csv(manifest_root / "paper_stack_manifest.csv", index=False, encoding="utf-8-sig")
    (manifest_root / "paper_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return split, stacks


class PaperAsterisDataset(Dataset):
    """Random paper-style 16-exposure pairs sampled from prepared sequence stacks."""

    def __init__(
        self,
        stack_manifest: pd.DataFrame,
        output_root: str | Path,
        split: str,
        *,
        patch_size: int,
        samples_per_sequence: int,
        seed: int,
        augment: bool,
    ) -> None:
        self.rows = stack_manifest.loc[stack_manifest["split"].eq(split)].reset_index(drop=True)
        if self.rows.empty or (self.rows["frames"] < 16).any():
            raise ValueError(f"Every {split} stack must contain at least 16 frames")
        self.root = Path(output_root)
        self.split = split
        self.patch_size = int(patch_size)
        self.samples_per_sequence = int(samples_per_sequence)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_sequence

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _stack(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        if row_index not in self._cache:
            row = self.rows.iloc[row_index]
            data = np.load(self.root / row.normalized_path, mmap_mode="r")
            valid = np.load(self.root / row.valid_path, mmap_mode="r")
            self._cache[row_index] = (data, valid)
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        return self._cache[row_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row_index = int(index) % len(self.rows)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        stack, valid = self._stack(row_index)
        count, height, width = stack.shape
        if self.split == "train":
            selected = rng.choice(count, size=16, replace=False)
        else:
            start = (int(index) // len(self.rows)) % max(count - 15, 1)
            selected = np.arange(start, start + 16) % count
        size = self.patch_size
        best = None
        for _ in range(24):
            y0 = int(rng.integers(0, height - size + 1))
            x0 = int(rng.integers(0, width - size + 1))
            use = valid[selected, y0 : y0 + size, x0 : x0 + size].astype(bool)
            best = (y0, x0, use)
            if float(np.mean(use)) >= 0.90:
                break
        assert best is not None
        y0, x0, use = best
        volume = np.asarray(stack[selected, y0 : y0 + size, x0 : x0 + size], dtype=np.float32)
        first, second = volume[0::2].copy(), volume[1::2].copy()
        first_valid, second_valid = use[0::2], use[1::2]
        first_mean = _masked_mean(first, first_valid)
        second_mean = _masked_mean(second, second_valid)
        first_bias = float(np.nanmedian(first_mean))
        second_bias = float(np.nanmedian(second_mean))
        first -= first_bias
        second -= second_bias
        loss_mask = first_valid & second_valid
        first = fill_invalid_with_temporal_mean(first, first_valid)
        second = fill_invalid_with_temporal_mean(second, second_valid)
        if rng.random() < 0.5:
            first, second = second, first
        if self.augment:
            k = int(rng.integers(0, 4))
            first = np.rot90(first, k, axes=(-2, -1))
            second = np.rot90(second, k, axes=(-2, -1))
            loss_mask = np.rot90(loss_mask, k, axes=(-2, -1))
            if rng.random() < 0.5:
                first, second, loss_mask = (
                    first[..., ::-1], second[..., ::-1], loss_mask[..., ::-1]
                )
        return {
            "input": torch.from_numpy(np.ascontiguousarray(first[None])),
            "target": torch.from_numpy(np.ascontiguousarray(second[None])),
            "mask": torch.from_numpy(np.ascontiguousarray(loss_mask[None].astype(np.float32))),
            "sequence": str(self.rows.iloc[row_index].sequence),
        }


def paper_asteris_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_scale: float = 1e6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Invalid detector voxels contribute neither directly nor through the
    # temporal-mean term.  Reduction by the valid count also prevents samples
    # with more blind pixels from receiving a smaller effective loss weight.
    stack_error = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    stack_l1 = stack_error.sum() / mask.sum().clamp_min(1.0) * float(loss_scale)
    mean_mask = (mask.sum(dim=2) > 0).float()
    valid_count = mask.sum(dim=2).clamp_min(1.0)
    prediction_mean = (prediction * mask).sum(dim=2) / valid_count
    target_mean = (target * mask).sum(dim=2) / valid_count
    prediction_mean = torch.clamp(prediction_mean, max=target_mean.max())
    mean_error = (prediction_mean - target_mean).square() * mean_mask
    mean_l2 = mean_error.sum() / mean_mask.sum().clamp_min(1.0) * float(loss_scale)
    total = 0.125 * stack_l1 + mean_l2
    return total, stack_l1, mean_l2


def _official_initialization(model: torch.nn.Module, path: str | Path) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.network.load_state_dict(state, strict=True)


def train_paper_model(
    output_root: str | Path,
    *,
    config: PaperAsterisConfig = PaperAsterisConfig(),
    device: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    config.validate()
    output_root = Path(output_root)
    stacks = pd.read_csv(
        output_root / "manifests" / "paper_stack_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    train_data = PaperAsterisDataset(
        stacks,
        output_root,
        "train",
        patch_size=config.patch_size,
        samples_per_sequence=config.samples_per_sequence,
        seed=config.seed,
        augment=True,
    )
    validation_data = PaperAsterisDataset(
        stacks,
        output_root,
        "validation",
        patch_size=config.patch_size,
        samples_per_sequence=config.validation_samples_per_sequence,
        seed=config.seed + 17,
        augment=False,
    )
    _seed_everything(config.seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_asteris_model(
        "asteris8", f_maps=config.f_maps, num_refinement_blocks=config.num_refinement_blocks
    ).to(device_obj)
    if config.initialize_from_official:
        _official_initialization(model, config.official_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_t_max, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device_obj.type == "cuda")
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_path, last_path = checkpoint_root / "best_checkpoint.pt", checkpoint_root / "last_checkpoint.pt"
    history, best_loss, stale = [], float("inf"), 0
    amp_enabled = bool(config.amp and device_obj.type == "cuda")
    training_started = time.monotonic()
    for epoch in range(config.epochs):
        epoch_started = time.monotonic()
        train_data.set_epoch(epoch)
        model.train()
        train_values = []
        for batch_index, batch in enumerate(train_loader):
            inputs = batch["input"].to(device_obj)
            targets = batch["target"].to(device_obj)
            masks = batch["mask"].to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_obj.type, enabled=amp_enabled):
                prediction = model(inputs)
                loss, stack_l1, mean_l2 = paper_asteris_loss(
                    prediction, targets, masks, loss_scale=config.loss_scale
                )
            # The released trainer is float32.  Its global 1e6 multiplier
            # overflows float16 activation gradients; remove only that common
            # factor for AMP backpropagation while retaining official logged
            # losses and the exact relative weighting of both objectives.
            optimization_loss = loss / config.loss_scale if amp_enabled else loss
            if not torch.isfinite(optimization_loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch+1}, batch {batch_index+1}')
            old_scale = scaler.get_scale()
            scaler.scale(optimization_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            train_values.append((loss.item(), stack_l1.item(), mean_l2.item()))
            if (batch_index + 1) % 40 == 0:
                (checkpoint_root / 'training_progress.json').write_text(json.dumps(dict(
                    phase='train', epoch=epoch+1, batch=batch_index+1, batches=len(train_loader),
                    elapsed_seconds=time.monotonic()-training_started, complete=False)), encoding='utf-8')
                print(f'epoch {epoch+1:03d} batch {batch_index+1}/{len(train_loader)}', flush=True)
        model.eval()
        validation_values = []
        with torch.inference_mode():
            for batch in validation_loader:
                inputs = batch["input"].to(device_obj)
                targets = batch["target"].to(device_obj)
                masks = batch["mask"].to(device_obj)
                with torch.autocast(device_type=device_obj.type, enabled=amp_enabled):
                    prediction = model(inputs)
                    values = paper_asteris_loss(
                        prediction, targets, masks, loss_scale=config.loss_scale
                    )
                validation_values.append(tuple(value.item() for value in values))
        train_mean = np.mean(train_values, axis=0)
        validation_mean = np.mean(validation_values, axis=0)
        if not np.isfinite(validation_mean).all():
            raise FloatingPointError(f'Non-finite validation loss at epoch {epoch+1}')
        row = {
            "epoch": epoch + 1,
            "train_loss": train_mean[0],
            "train_stack_l1": train_mean[1],
            "train_mean_l2": train_mean[2],
            "validation_loss": validation_mean[0],
            "validation_stack_l1": validation_mean[1],
            "validation_mean_l2": validation_mean[2],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(checkpoint_root / 'training_history.csv', index=False, encoding='utf-8-sig')
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": asdict(config),
            "upstream_source_sha256": upstream_source_sha256("asteris8"),
            "official_initialization": config.official_checkpoint if config.initialize_from_official else None,
            "best_validation_loss": min(best_loss, float(validation_mean[0])),
            "history": history,
        }
        torch.save(state, last_path)
        if validation_mean[0] < best_loss:
            best_loss, stale = float(validation_mean[0]), 0
            torch.save(state, best_path)
        else:
            stale += 1
        print(
            f"epoch {epoch + 1:03d} train={train_mean[0]:.6f} "
            f"validation={validation_mean[0]:.6f}",
            flush=True,
        )
        if stale >= config.patience:
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(checkpoint_root / "training_history.csv", index=False, encoding="utf-8-sig")
    (checkpoint_root / 'training_progress.json').write_text(json.dumps(dict(
        phase='complete', epochs=len(history), best_validation_loss=best_loss,
        elapsed_seconds=time.monotonic()-training_started, complete=True)), encoding='utf-8')
    return best_path, history_frame


def load_paper_model(
    checkpoint_path: str | Path, device: str | None = None
) -> tuple[torch.nn.Module, dict, PaperAsterisConfig]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_obj, weights_only=False)
    config = PaperAsterisConfig(**checkpoint["config"])
    model = build_asteris_model(
        "asteris8", f_maps=config.f_maps, num_refinement_blocks=config.num_refinement_blocks
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, config


def process_nmean(stack: np.ndarray, valid: np.ndarray, nmean: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an arbitrary exposure stack into the released model's eight bins."""

    indices = np.array_split(np.arange(len(stack)), nmean)
    outputs, masks = [], []
    for group in indices:
        group_valid = valid[group]
        count = group_valid.sum(axis=0)
        mean = np.where(
            count > 0,
            np.where(group_valid, stack[group], 0.0).sum(axis=0) / np.maximum(count, 1),
            0.0,
        )
        outputs.append(mean.astype(np.float32))
        masks.append(count > 0)
    return np.stack(outputs), np.stack(masks)


def _restore_clip_residual(residual: np.ndarray) -> np.ndarray:
    values = np.where(np.abs(residual) > 0, residual, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        restored = np.nanmedian(values, axis=0)
    return np.nan_to_num(restored, nan=0.0).astype(np.float32)


def denoise_registered_exposures(
    physical: np.ndarray,
    valid: np.ndarray,
    model: torch.nn.Module,
    config: PaperAsterisConfig,
    *,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the released clipping, eight-bin inference and temporal collapse."""

    clipped, clip_residual, clipped_valid, clip_metrics = paper_sigma_clip(
        physical,
        valid,
        temporal_sigma=config.temporal_sigma,
        global_sigma=config.global_sigma,
    )
    normalized, center, std = _normalize_stack(clipped, clipped_valid, config.scale_factor)
    model_stack, model_valid = process_nmean(normalized, clipped_valid, nmean=8)
    mean_image = _masked_mean(model_stack, model_valid)
    bias = float(np.nanmedian(mean_image))
    model_input = model_stack - bias
    model_input = fill_invalid_with_temporal_mean(model_input, model_valid)
    prediction = infer_volume(
        model_input,
        model,
        device=device,
        tile_size=config.inference_tile_size,
        overlap=config.inference_overlap,
        amp=config.amp,
    )
    prediction += bias
    clipped_input_coadd = _masked_mean(clipped, clipped_valid)
    prediction_mean = _masked_mean(prediction, model_valid)
    restored_clip = _restore_clip_residual(clip_residual)
    input_coadd = (clipped_input_coadd + restored_clip).astype(np.float32)
    denoised = (
        (prediction_mean - 1.0) * config.scale_factor * std + center + restored_clip
    ).astype(np.float32)
    output_valid = np.any(model_valid, axis=0) & np.isfinite(input_coadd) & np.isfinite(denoised)
    input_coadd[~output_valid] = np.nan
    denoised[~output_valid] = np.nan
    return input_coadd, denoised, output_valid, clip_metrics


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_paper_inference(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    *,
    evaluation_sequences: Sequence[str] | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write one full-strength ASTERIS8 temporal coadd for each held-out sequence."""

    input_root, dataset_root, output_root = Path(input_root), Path(dataset_root), Path(output_root)
    checkpoint_path = Path(checkpoint_path)
    split = pd.read_csv(
        output_root / "manifests" / "split_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    if evaluation_sequences is not None:
        split = split.loc[split.sequence.isin(map(str, evaluation_sequences))]
    detector_mask = load_detector_mask(dataset_root)
    model, _, config = load_paper_model(checkpoint_path, device=device)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_hash = _checkpoint_hash(checkpoint_path)
    records = []
    for sequence, group in split.loc[split.split.eq("test")].groupby("sequence", sort=True):
        group = group.sort_values("frame_index").reset_index(drop=True)
        physical, valid = _load_registered_stack(group, input_root, detector_mask)
        input_coadd, denoised, output_valid, clip_metrics = denoise_registered_exposures(
            physical, valid, model, config, device=device_obj
        )
        residual = (input_coadd - denoised).astype(np.float32)
        coverage = valid.sum(axis=0)
        dq = build_dq(
            input_coadd.shape,
            no_coverage=~output_valid,
            partial_coverage=(coverage > 0) & (coverage < len(valid)),
        )
        header = fits.getheader(input_root / group.iloc[-1].product_path).copy()
        header["HIERARCH AST MODEL"] = "ASTERIS8_PAPER"
        header["HIERARCH AST NEXP"] = len(group)
        header["HIERARCH AST NBIN"] = 8
        header["HIERARCH AST CKPT"] = checkpoint_hash[:16]
        header["HIERARCH AST ALPHA"] = 1.0
        sequence_root = output_root / "coadds" / str(sequence)
        paths = {
            "input": sequence_root / f"input_coadd_{sequence}.fits",
            "denoised": sequence_root / f"asteris8_coadd_{sequence}.fits",
            "residual": sequence_root / f"asteris8_residual_{sequence}.fits",
        }
        sequence_root.mkdir(parents=True, exist_ok=True)
        for kind, path, data in (
            ("INPUT_COADD", paths["input"], input_coadd),
            ("DENOISED_COADD", paths["denoised"], denoised),
            ("RESIDUAL", paths["residual"], residual),
        ):
            out_header = header.copy()
            out_header["HIERARCH AST KIND"] = kind
            write_fits_with_dq(
                path,
                data,
                out_header,
                dq,
                overwrite=overwrite,
                output_verify="silentfix",
            )
        metric_mask = ~output_valid
        metric_mask[:32] = True
        metric_mask[-32:] = True
        metric_mask[:, :32] = True
        metric_mask[:, -32:] = True
        noise_before = neighbor_difference_noise(input_coadd, metric_mask)
        noise_after = neighbor_difference_noise(denoised, metric_mask)
        records.append(
            {
                "sequence": str(sequence),
                "test_exposures": len(group),
                "input_path": paths["input"].relative_to(output_root).as_posix(),
                "denoised_path": paths["denoised"].relative_to(output_root).as_posix(),
                "residual_path": paths["residual"].relative_to(output_root).as_posix(),
                "noise_before": noise_before,
                "noise_after": noise_after,
                "noise_ratio": noise_after / noise_before,
                "equation_max_abs_error_float32": float(
                    np.nanmax(np.abs(denoised - (input_coadd - residual)))
                ),
                "checkpoint_sha256": checkpoint_hash,
                **clip_metrics,
            }
        )
    statistics = pd.DataFrame(records)
    statistics.to_csv(output_root / "paper_coadd_statistics.csv", index=False, encoding="utf-8-sig")
    return statistics
