"""Training, tiled inference, FITS products, and reproducibility controls for Noise2Noise."""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .dataset import (
    PairedPatchDataset,
    build_pair_manifest,
    build_split_manifest,
    estimate_sequence_scales,
    load_detector_mask,
)
from .model import ResidualDnCNN


@dataclass(frozen=True)
class Noise2NoiseConfig:
    seed: int = 20260820
    lags: tuple[int, ...] = (2, 3, 4, 5)
    patch_size: int = 128
    source_patch_fraction: float = 0.5
    train_samples_per_epoch: int = 512
    validation_samples: int = 192
    batch_size: int = 8
    epochs: int = 30
    early_stopping_patience: int = 7
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    model_depth: int = 8
    model_features: int = 32
    gradient_clip_norm: float = 5.0
    inference_tile_size: int = 256
    inference_overlap: int = 32
    edge_width: int = 32
    max_validation_photometry_change: float = 0.01

    def validate(self) -> None:
        if not self.lags or any(lag < 2 for lag in self.lags):
            raise ValueError("Noise2Noise temporal lags must be >= 2")
        if self.patch_size < 32 or self.patch_size % 8 != 0:
            raise ValueError("patch_size must be >= 32 and divisible by 8")
        if not 0 <= self.source_patch_fraction <= 1:
            raise ValueError("source_patch_fraction must be in [0, 1]")
        if min(self.train_samples_per_epoch, self.validation_samples, self.batch_size) < 1:
            raise ValueError("sample and batch counts must be positive")
        if self.epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epochs and early stopping patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are outside their valid range")
        if self.model_depth < 3 or self.model_features < 4:
            raise ValueError("model_depth >= 3 and model_features >= 4 are required")
        if self.inference_tile_size < self.patch_size:
            raise ValueError("inference_tile_size must be at least patch_size")
        if not 0 <= self.inference_overlap < self.inference_tile_size // 2:
            raise ValueError("inference_overlap must be less than half the tile size")
        if not 0 < self.max_validation_photometry_change < 1:
            raise ValueError("max_validation_photometry_change must be in (0, 1)")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_manifests(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    config: Noise2NoiseConfig | None = None,
    sequences: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    config = config or Noise2NoiseConfig()
    config.validate()
    output_root = Path(output_root)
    manifest_root = output_root / "manifests"
    split = build_split_manifest(
        input_root,
        dataset_root,
        manifest_root / "split_manifest.csv",
        sequences=sequences,
    )
    pairs = build_pair_manifest(
        split,
        manifest_root / "pair_manifest.csv",
        lags=config.lags,
    )
    detector_mask = load_detector_mask(dataset_root)
    scales = estimate_sequence_scales(
        split,
        input_root,
        detector_mask,
        edge_width=config.edge_width,
    )
    manifest_root.mkdir(parents=True, exist_ok=True)
    with (manifest_root / "normalization.json").open("w", encoding="utf-8") as handle:
        json.dump(scales, handle, indent=2, ensure_ascii=False)
    with (manifest_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
    return split, pairs, scales


def load_manifests(
    output_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    manifest_root = Path(output_root) / "manifests"
    split = pd.read_csv(
        manifest_root / "split_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    pairs = pd.read_csv(
        manifest_root / "pair_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    with (manifest_root / "normalization.json").open(encoding="utf-8") as handle:
        scales = {str(key): float(value) for key, value in json.load(handle).items()}
    return split, pairs, scales


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def _run_validation(
    model: ResidualDnCNN,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total, weight = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            model_input = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(model_input)
                loss = masked_mse(prediction, target, mask)
            batch_size = int(model_input.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            weight += batch_size
    return total / max(weight, 1)


def train_model(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    config: Noise2NoiseConfig | None = None,
    device: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    config = config or Noise2NoiseConfig()
    config.validate()
    set_reproducible_seed(config.seed)
    split, pairs, scales = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    train_dataset = PairedPatchDataset(
        split,
        pairs,
        input_root,
        detector_mask,
        scales,
        split="train",
        patch_size=config.patch_size,
        samples_per_epoch=config.train_samples_per_epoch,
        source_fraction=config.source_patch_fraction,
        augment=True,
        seed=config.seed,
    )
    validation_dataset = PairedPatchDataset(
        split,
        pairs,
        input_root,
        detector_mask,
        scales,
        split="validation",
        patch_size=config.patch_size,
        samples_per_epoch=config.validation_samples,
        source_fraction=config.source_patch_fraction,
        augment=False,
        seed=config.seed + 97,
        cache_size=32,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ResidualDnCNN(config.model_depth, config.model_features).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1.0e-6,
    )
    use_amp = device_obj.type == "cuda"
    scaler = torch.amp.GradScaler(device_obj.type, enabled=use_amp)
    checkpoint_root = Path(output_root) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_root / "best_checkpoint.pt"
    last_path = checkpoint_root / "last_checkpoint.pt"
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []
    for epoch in range(config.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running, samples = 0.0, 0
        for batch in train_loader:
            model_input = batch["input"].to(device_obj, non_blocking=True)
            target = batch["target"].to(device_obj, non_blocking=True)
            mask = batch["loss_mask"].to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_obj.type, enabled=use_amp):
                prediction = model(model_input)
                loss = masked_mse(prediction, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(model_input.shape[0])
            running += float(loss.detach().cpu()) * batch_size
            samples += batch_size
        validation_loss = _run_validation(model, validation_loader, device_obj, use_amp)
        train_loss = running / max(samples, 1)
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": asdict(config),
            "sequence_scales": scales,
            "validation_loss": validation_loss,
        }
        torch.save(payload, last_path)
        improved = validation_loss < best_validation - 1.0e-6
        if improved:
            best_validation = validation_loss
            epochs_without_improvement = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch + 1:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} lr={record['learning_rate']:.2e}",
            flush=True,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"early stopping after epoch {epoch + 1}", flush=True)
            break
    history_table = pd.DataFrame(history)
    history_table.to_csv(
        checkpoint_root / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return best_path, history_table


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[ResidualDnCNN, dict]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    config = checkpoint["config"]
    model = ResidualDnCNN(config["model_depth"], config["model_features"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device_obj).eval()
    return model, checkpoint


def _tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - 2 * overlap
    positions = list(range(0, length - tile_size + 1, step))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def denoise_array(
    image: np.ndarray,
    valid_mask: np.ndarray,
    model: ResidualDnCNN,
    scale: float,
    tile_size: int = 256,
    overlap: int = 32,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise a native-orientation frame with overlap-weighted tiled inference."""

    original = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(original)
    center = float(np.median(original[valid]))
    normalized = (original - center) / float(scale)
    normalized[~valid] = 0.0
    height, width = original.shape
    tile = min(tile_size, height, width)
    overlap = min(overlap, max(0, tile // 2 - 1))
    y_positions = _tile_positions(height, tile, overlap)
    x_positions = _tile_positions(width, tile, overlap)
    window_1d = np.hanning(tile).astype(np.float32)
    window_1d = np.clip(window_1d, 0.05, None)
    window = window_1d[:, None] * window_1d[None, :]
    accumulated = np.zeros_like(normalized, dtype=np.float32)
    weights = np.zeros_like(normalized, dtype=np.float32)
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for y0 in y_positions:
            for x0 in x_positions:
                patch = normalized[y0 : y0 + tile, x0 : x0 + tile]
                mask_patch = valid[y0 : y0 + tile, x0 : x0 + tile].astype(np.float32)
                model_input = np.stack([patch, mask_patch], axis=0)[None]
                tensor = torch.from_numpy(model_input).to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    denoised_patch = model(tensor)
                result = denoised_patch[0, 0].float().cpu().numpy()
                accumulated[y0 : y0 + tile, x0 : x0 + tile] += result * window
                weights[y0 : y0 + tile, x0 : x0 + tile] += window
    full_denoised_normalized = np.divide(
        accumulated,
        weights,
        out=normalized.copy(),
        where=weights > 0,
    )
    if not 0 <= strength <= 1:
        raise ValueError("denoise strength must be in [0, 1]")
    denoised_normalized = normalized - float(strength) * (
        normalized - full_denoised_normalized
    )
    denoised = denoised_normalized * float(scale) + center
    denoised[~valid] = original[~valid]
    residual = original - denoised.astype(np.float32)
    return denoised.astype(np.float32), residual.astype(np.float32)


def robust_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def neighbor_difference_noise(image: np.ndarray, mask: np.ndarray) -> float:
    valid_h = ~mask[:, 1:] & ~mask[:, :-1]
    valid_v = ~mask[1:, :] & ~mask[:-1, :]
    differences = np.concatenate(
        [
            (image[:, 1:] - image[:, :-1])[valid_h],
            (image[1:, :] - image[:-1, :])[valid_v],
        ]
    )
    return robust_std(differences) / np.sqrt(2.0)


def aperture_flux_snr(image: np.ndarray, target: Mapping) -> tuple[float, float]:
    required = ("xc", "yc", "r_ap", "r_in", "r_out")
    if any(target.get(key) is None or not np.isfinite(float(target.get(key))) for key in required):
        return float("nan"), float("nan")
    x, y = float(target["xc"]) - 1.0, float(target["yc"]) - 1.0
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    radius = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    aperture = radius <= float(target["r_ap"])
    annulus = (radius >= float(target["r_in"])) & (radius <= float(target["r_out"]))
    annulus_values = image[annulus & np.isfinite(image)]
    aperture_values = image[aperture & np.isfinite(image)]
    if annulus_values.size < 16 or aperture_values.size == 0:
        return float("nan"), float("nan")
    background = float(np.median(annulus_values))
    flux = float(np.sum(aperture_values - background))
    noise = robust_std(annulus_values) * np.sqrt(aperture_values.size)
    return flux, float(flux / noise) if np.isfinite(noise) and noise > 0 else float("nan")


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate_denoise_strength(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    config: Noise2NoiseConfig | None = None,
    device: str | None = None,
    strengths: Sequence[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose the strongest validation-only blend that satisfies the high-SNR flux gate."""

    config = config or Noise2NoiseConfig()
    config.validate()
    input_root, output_root = Path(input_root), Path(output_root)
    split, _, scales = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    model, _ = load_model(checkpoint_path, device=device)
    validation = split.loc[
        (split["split"] == "validation") & (split["input_snr"] >= 10)
    ]
    if validation.empty:
        raise RuntimeError("No high-SNR validation frames are available for strength calibration")
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = []
    for frame in validation.itertuples(index=False):
        image = np.asarray(fits.getdata(input_root / frame.product_path), dtype=np.float32)
        valid = ~detector_mask & np.isfinite(image)
        full_denoised, _ = denoise_array(
            image,
            valid,
            model,
            scales[str(frame.sequence)],
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            strength=1.0,
        )
        metric_mask = detector_mask | ~np.isfinite(image)
        metric_mask[: config.edge_width] = True
        metric_mask[-config.edge_width :] = True
        metric_mask[:, : config.edge_width] = True
        metric_mask[:, -config.edge_width :] = True
        target = {
            key: getattr(frame, key)
            for key in ("xc", "yc", "r_ap", "r_in", "r_out")
        }
        cached.append((image, full_denoised, metric_mask, target))
    strengths = strengths or tuple(np.round(np.linspace(0.05, 1.0, 96), 2))
    rows = []
    for strength in strengths:
        changes = []
        for image, full_denoised, metric_mask, target in cached:
            candidate = image - float(strength) * (image - full_denoised)
            flux_before, _ = aperture_flux_snr(image, target)
            flux_after, _ = aperture_flux_snr(candidate, target)
            change = (
                abs((flux_after - flux_before) / flux_before)
                if np.isfinite(flux_before) and flux_before != 0 and np.isfinite(flux_after)
                else np.inf
            )
            changes.append(change)
        rows.append(
            {
                "strength": float(strength),
                "validation_max_abs_photometry_change": float(np.max(changes)),
                "validation_median_noise_ratio": np.nan,
                "validation_max_noise_ratio": np.nan,
                "passes_photometry_gate": bool(
                    np.max(changes) <= config.max_validation_photometry_change
                ),
            }
        )
    table = pd.DataFrame(rows)
    passing = table.loc[table["passes_photometry_gate"]]
    if passing.empty:
        raise RuntimeError("No non-zero denoise strength satisfies the validation photometry gate")
    selected = float(passing.sort_values("strength").iloc[-1]["strength"])
    selected_ratios = []
    for image, full_denoised, metric_mask, _ in cached:
        candidate = image - selected * (image - full_denoised)
        noise_before = neighbor_difference_noise(image, metric_mask)
        noise_after = neighbor_difference_noise(candidate, metric_mask)
        selected_ratios.append(noise_after / max(noise_before, np.finfo(float).eps))
    selected_index = table.index[np.isclose(table["strength"], selected)][0]
    table.loc[selected_index, "validation_median_noise_ratio"] = float(
        np.median(selected_ratios)
    )
    table.loc[selected_index, "validation_max_noise_ratio"] = float(np.max(selected_ratios))
    manifest_root = output_root / "manifests"
    table.to_csv(
        manifest_root / "strength_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (manifest_root / "strength_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_strength": selected,
                "max_validation_photometry_change": config.max_validation_photometry_change,
            },
            handle,
            indent=2,
        )
    return selected, table


def load_calibrated_strength(output_root: str | Path) -> float:
    path = Path(output_root) / "manifests" / "strength_calibration.json"
    with path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["selected_strength"])


def _output_header(
    header: fits.Header,
    product: str,
    split: str,
    checkpoint_hash: str,
    scale: float,
    strength: float,
) -> fits.Header:
    out = header.copy()
    out["HIERARCH N2N PROD"] = product
    out["HIERARCH N2N SPLIT"] = split
    out["HIERARCH N2N CKPT"] = checkpoint_hash[:16]
    out["HIERARCH N2N SCALE"] = round(float(scale), 6)
    out["HIERARCH N2N ALPHA"] = round(float(strength), 6)
    out.add_history("Self-supervised Noise2Noise residual DnCNN implemented by astr_ir.noise2noise")
    out.add_history("Science equation: denoised = background_subtracted_input - predicted_residual")
    return out


def run_inference(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    config: Noise2NoiseConfig | None = None,
    device: str | None = None,
    overwrite: bool = False,
    strength: float | None = None,
) -> pd.DataFrame:
    config = config or Noise2NoiseConfig()
    config.validate()
    input_root, output_root, checkpoint_path = Path(input_root), Path(output_root), Path(checkpoint_path)
    split, _, scales = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    model, _ = load_model(checkpoint_path, device=device)
    checkpoint_hash = _checkpoint_hash(checkpoint_path)
    strength = load_calibrated_strength(output_root) if strength is None else float(strength)
    rows: list[dict] = []
    for frame in split.sort_values(["sequence", "frame_index"]).itertuples(index=False):
        input_path = input_root / frame.product_path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(input_path, memmap=False) as hdul:
                hdul.verify("silentfix")
                image = np.asarray(hdul[0].data, dtype=np.float32)
                header = hdul[0].header.copy()
        valid = ~detector_mask & np.isfinite(image)
        scale = scales[str(frame.sequence)]
        denoised, _ = denoise_array(
            image,
            valid,
            model,
            scale,
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            strength=strength,
        )
        original32 = image.astype(np.float32)
        residual32 = original32 - denoised.astype(np.float32)
        denoised32 = original32 - residual32
        sequence_root = output_root / "denoised" / str(frame.sequence)
        residual_root = output_root / "residual" / str(frame.sequence)
        sequence_root.mkdir(parents=True, exist_ok=True)
        residual_root.mkdir(parents=True, exist_ok=True)
        denoised_path = sequence_root / f"noise2noise_denoised_{frame.filename}"
        residual_path = residual_root / f"noise2noise_residual_{frame.filename}"
        fits.PrimaryHDU(
            denoised32,
            header=_output_header(
                header, "denoised", frame.split, checkpoint_hash, scale, strength
            ),
        ).writeto(denoised_path, overwrite=overwrite, output_verify="silentfix")
        fits.PrimaryHDU(
            residual32,
            header=_output_header(
                header, "residual", frame.split, checkpoint_hash, scale, strength
            ),
        ).writeto(residual_path, overwrite=overwrite, output_verify="silentfix")
        target = {
            key: getattr(frame, key)
            for key in ("xc", "yc", "r_ap", "r_in", "r_out")
        }
        flux_before, snr_before = aperture_flux_snr(original32, target)
        flux_after, snr_after = aperture_flux_snr(denoised32, target)
        metric_mask = detector_mask | ~np.isfinite(original32)
        metric_mask[: config.edge_width] = True
        metric_mask[-config.edge_width :] = True
        metric_mask[:, : config.edge_width] = True
        metric_mask[:, -config.edge_width :] = True
        noise_before = neighbor_difference_noise(original32, metric_mask)
        noise_after = neighbor_difference_noise(denoised32, metric_mask)
        written_denoised = np.asarray(fits.getdata(denoised_path), dtype=np.float32)
        written_residual = np.asarray(fits.getdata(residual_path), dtype=np.float32)
        expected = original32 - written_residual
        finite = np.isfinite(expected)
        if not np.array_equal(np.isfinite(written_denoised), finite):
            raise RuntimeError(f"Finite-pixel equation mask mismatch: {denoised_path}")
        equation_error = (
            float(np.max(np.abs(written_denoised[finite] - expected[finite])))
            if np.any(finite)
            else 0.0
        )
        rows.append(
            {
                "frame_id": frame.frame_id,
                "sequence": str(frame.sequence),
                "frame_index": int(frame.frame_index),
                "split": frame.split,
                "filename": frame.filename,
                "upstream_applied": bool(frame.upstream_applied),
                "input_snr": frame.input_snr,
                "normalization_scale": scale,
                "denoise_strength": strength,
                "noise_before": noise_before,
                "noise_after": noise_after,
                "noise_ratio": noise_after / max(noise_before, np.finfo(float).eps),
                "photometry_before": flux_before,
                "photometry_after": flux_after,
                "photometry_change_fraction": (
                    (flux_after - flux_before) / flux_before
                    if np.isfinite(flux_before) and flux_before != 0 and np.isfinite(flux_after)
                    else np.nan
                ),
                "aperture_snr_before": snr_before,
                "aperture_snr_after": snr_after,
                "aperture_snr_ratio": (
                    snr_after / snr_before
                    if np.isfinite(snr_before) and snr_before != 0 and np.isfinite(snr_after)
                    else np.nan
                ),
                "equation_max_abs_error_float32": equation_error,
                "denoised_path": denoised_path.relative_to(output_root).as_posix(),
                "residual_path": residual_path.relative_to(output_root).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        print(f"inference {frame.sequence} {int(frame.frame_index) + 1:02d}/80", flush=True)
    statistics = pd.DataFrame(rows)
    statistics.to_csv(
        output_root / "noise2noise_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return statistics
