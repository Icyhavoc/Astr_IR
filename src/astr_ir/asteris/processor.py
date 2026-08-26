"""Preparation, training, checkpointing and FITS inference for ASTERIS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits
from scipy.ndimage import shift
from torch.utils.data import DataLoader

from astr_ir.noise2noise.dataset import build_split_manifest, load_detector_mask
from astr_ir.noise2noise.processor import aperture_flux_snr, neighbor_difference_noise
from astr_ir.dq import build_dq, read_dq, write_fits_with_dq

from .dataset import (
    AsterisPatchDataset,
    assert_window_isolation,
    build_window_manifest,
    load_registered_stack,
    relabel_manifest_for_patch_t,
)
from .inference import denoise_registered_stack
from .model import build_asteris_model, upstream_source_sha256
from .preprocessing import (
    build_noise_estimation_mask,
    circular_source_mask,
    fill_invalid_with_temporal_mean,
    fit_normalization,
)


@dataclass(frozen=True)
class AsterisConfig:
    model: str = "asteris4"
    patch_t: int = 4
    patch_size: int = 64
    f_maps: int = 24
    num_blocks: tuple[int, ...] | None = None
    num_refinement_blocks: int = 4
    output_mode: str = "direct"
    sigma: float = 3.0
    temporal_clip: bool = False
    edge_width: int = 32
    epochs: int = 30
    train_samples_per_epoch: int = 128
    validation_samples: int = 48
    batch_size: int = 1
    learning_rate: float = 1e-4
    patience: int = 8
    seed: int = 20260824
    inference_tile_size: int = 64
    inference_overlap: int = 16
    amp: bool = True

    def validate(self) -> None:
        expected = 4 if self.model.lower() == "asteris4" else 8
        if self.model.lower() not in {"asteris4", "asteris8"}:
            raise ValueError("model must be asteris4 or asteris8")
        if self.patch_t != expected:
            raise ValueError(f"{self.model} requires patch_t={expected}")
        divisor = 4 if expected == 4 else 8
        if self.patch_size % divisor or self.inference_tile_size % divisor:
            raise ValueError(f"patch sizes must be divisible by {divisor}")
        if self.output_mode not in {"direct", "residual"}:
            raise ValueError("output_mode must be direct or residual")
        if min(self.epochs, self.batch_size, self.train_samples_per_epoch, self.validation_samples) < 1:
            raise ValueError("training sizes and epochs must be positive")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame_training_samples(
    split: pd.DataFrame,
    input_root: Path,
    detector_mask: np.ndarray,
    edge_width: int,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    normalizations: dict[str, dict[str, float]] = {}
    audit_rows: list[dict] = []
    training = split.loc[(split["split"] == "train") & split["upstream_applied"].astype(bool)]
    for sequence, group in training.groupby("sequence", sort=True):
        samples = []
        for row in group.itertuples(index=False):
            image = np.asarray(fits.getdata(input_root / row.product_path), dtype=np.float32)
            source = circular_source_mask(
                image.shape,
                [
                    (
                        float(row.track_x),
                        float(row.track_y),
                        float(np.nanmax([row.r_out, 3.0 * row.fwhm, 8.0])),
                    )
                ],
            )
            estimation = build_noise_estimation_mask(image.shape, detector_mask, source, edge_width)
            values = image[estimation & np.isfinite(image)][::16]
            samples.append(values.astype(np.float64))
        combined = np.concatenate(samples)
        stats = fit_normalization(combined)
        stats["training_sample_count"] = int(combined.size)
        normalizations[str(sequence)] = stats
        audit_rows.append({"sequence": str(sequence), **stats, "source": "train_only"})
    return normalizations, pd.DataFrame(audit_rows)


def prepare_manifests(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Freeze frame splits, construct non-crossing windows and fit train-only stats."""

    config.validate()
    input_root, dataset_root, output_root = Path(input_root), Path(dataset_root), Path(output_root)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    split = build_split_manifest(input_root, dataset_root, manifest_root / "split_manifest.csv")
    split = relabel_manifest_for_patch_t(split, config.patch_t)
    split.to_csv(manifest_root / "split_manifest.csv", index=False, encoding="utf-8-sig")
    windows = build_window_manifest(
        split, manifest_root / "window_manifest.csv", patch_t=config.patch_t
    )
    assert_window_isolation(split, windows)
    detector_mask = load_detector_mask(dataset_root)
    normalizations, audit = _frame_training_samples(
        split, input_root, detector_mask, config.edge_width
    )
    (manifest_root / "normalization.json").write_text(
        json.dumps(normalizations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    audit.to_csv(manifest_root / "normalization_audit.csv", index=False, encoding="utf-8-sig")
    frame_table = split.set_index("frame_id", drop=False)
    clipping_rows: list[dict] = []
    usable_training = windows.loc[
        (windows["split"] == "train") & windows["usable"].astype(bool)
    ]
    for sequence, group in usable_training.groupby("sequence", sort=True):
        window = group.sort_values("start_index").iloc[0]
        frame_rows = frame_table.loc[str(window.frame_ids).split("|")]
        _, _, clipping = load_registered_stack(
            frame_rows,
            input_root,
            detector_mask,
            None,
            sigma=config.sigma,
            edge_width=config.edge_width,
            temporal_clip=config.temporal_clip,
        )
        row = {
            "sequence": str(sequence),
            "window_id": str(window.window_id),
            "low": clipping.low,
            "high": clipping.high,
            "clipped_fraction": clipping.clipped_fraction,
            "source_voxels_clipped": int(
                np.count_nonzero(clipping.clipping_mask[:, clipping.source_mask])
            ),
        }
        for metric in ("peak", "flux", "fwhm", "snr"):
            before = float(clipping.source_metrics_before[metric])
            after = float(clipping.source_metrics_after[metric])
            row[f"source_{metric}_before"] = before
            row[f"source_{metric}_after"] = after
            row[f"source_{metric}_change"] = after - before
        row["source_quality_gate_passed"] = bool(
            row["source_voxels_clipped"] == 0
            and all(
                np.isclose(row[f"source_{metric}_change"], 0.0, equal_nan=True)
                for metric in ("peak", "flux", "fwhm", "snr")
            )
        )
        clipping_rows.append(row)
    pd.DataFrame(clipping_rows).to_csv(
        manifest_root / "clipping_audit.csv", index=False, encoding="utf-8-sig"
    )
    clipping_policy = {
        "sigma": config.sigma,
        "global": True,
        "temporal": config.temporal_clip,
        "source_protected": True,
        "clipped_pixels_excluded_from_loss": True,
        "normalization_fit_split": "train",
    }
    (manifest_root / "clipping_policy.json").write_text(
        json.dumps(clipping_policy, indent=2), encoding="utf-8"
    )
    return split, windows, normalizations


def load_manifests(
    output_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    root = Path(output_root) / "manifests"
    split = pd.read_csv(root / "split_manifest.csv", encoding="utf-8-sig", dtype={"sequence": str})
    windows = pd.read_csv(root / "window_manifest.csv", encoding="utf-8-sig", dtype={"sequence": str})
    normalizations = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    assert_window_isolation(split, windows)
    return split, windows, normalizations


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (prediction - target).square() * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def asteris_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original-inspired stack SmoothL1 plus temporal-mean MSE, both masked."""

    stack_l1 = masked_smooth_l1(prediction, target, mask)
    valid_count = mask.sum(dim=2).clamp_min(1.0)
    predicted_mean = (prediction * mask).sum(dim=2) / valid_count
    target_mean = (target * mask).sum(dim=2) / valid_count
    mean_mask = (mask.sum(dim=2) > 0).to(mask.dtype)
    mean_l2 = masked_mse(predicted_mean, target_mean, mean_mask)
    return stack_l1 + mean_l2, stack_l1, mean_l2


def _model_kwargs(config: AsterisConfig) -> dict:
    return {
        "model_name": config.model,
        "f_maps": config.f_maps,
        "num_blocks": config.num_blocks,
        "num_refinement_blocks": config.num_refinement_blocks,
        "output_mode": config.output_mode,
    }


def _validation(model, loader, device, amp_enabled: bool) -> tuple[float, float, float]:
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["loss_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(inputs)
                values = asteris_loss(prediction, targets, masks)
            totals += np.array([float(value) for value in values])
            count += 1
    return tuple((totals / max(count, 1)).tolist())


def train_model(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
    device: str | None = None,
    resume: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    config.validate()
    set_reproducible_seed(config.seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split, windows, normalizations = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    common = dict(
        split_manifest=split,
        window_manifest=windows,
        input_root=input_root,
        detector_mask=detector_mask,
        normalizations=normalizations,
        patch_size=config.patch_size,
        sigma=config.sigma,
        edge_width=config.edge_width,
        temporal_clip=config.temporal_clip,
    )
    train_data = AsterisPatchDataset(
        **common, split="train", samples_per_epoch=config.train_samples_per_epoch, augment=True
    )
    validation_data = AsterisPatchDataset(
        **common, split="validation", samples_per_epoch=config.validation_samples, augment=False
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_asteris_model(**_model_kwargs(config)).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device_obj.type == "cuda")
    start_epoch, best_loss, history_rows = 0, np.inf, []
    checkpoint_root = Path(output_root) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_path, last_path = checkpoint_root / "best_checkpoint.pt", checkpoint_root / "last_checkpoint.pt"
    if resume is not None:
        state = torch.load(resume, map_location=device_obj, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_validation_loss", np.inf))
        history_rows = list(state.get("history", []))
    stale = 0
    amp_enabled = bool(config.amp and device_obj.type == "cuda")
    for epoch in range(start_epoch, config.epochs):
        train_data.set_epoch(epoch)
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        batches = 0
        for batch in train_loader:
            inputs = batch["input"].to(device_obj)
            targets = batch["target"].to(device_obj)
            masks = batch["loss_mask"].to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_obj.type, enabled=amp_enabled):
                prediction = model(inputs)
                loss, stack_l1, mean_l2 = asteris_loss(prediction, targets, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += [float(loss.detach()), float(stack_l1.detach()), float(mean_l2.detach())]
            batches += 1
        validation = _validation(model, validation_loader, device_obj, amp_enabled)
        row = {
            "epoch": epoch + 1,
            "train_loss": totals[0] / max(batches, 1),
            "train_stack_l1": totals[1] / max(batches, 1),
            "train_mean_l2": totals[2] / max(batches, 1),
            "validation_loss": validation[0],
            "validation_stack_l1": validation[1],
            "validation_mean_l2": validation[2],
        }
        history_rows.append(row)
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "model_kwargs": _model_kwargs(config),
            "upstream_source_sha256": upstream_source_sha256(config.model),
            "best_validation_loss": min(best_loss, validation[0]),
            "history": history_rows,
        }
        torch.save(state, last_path)
        if validation[0] < best_loss:
            best_loss, stale = validation[0], 0
            torch.save(state, best_path)
        else:
            stale += 1
        print(
            f"epoch {epoch + 1:03d} train={row['train_loss']:.6f} "
            f"validation={row['validation_loss']:.6f}",
            flush=True,
        )
        if stale >= config.patience:
            break
    history = pd.DataFrame(history_rows)
    history.to_csv(checkpoint_root / "training_history.csv", index=False, encoding="utf-8-sig")
    return best_path, history


def load_model(
    checkpoint_path: str | Path, device: str | None = None
) -> tuple[torch.nn.Module, dict]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model = build_asteris_model(**checkpoint["model_kwargs"]).to(device_obj)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_inference(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
    device: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Denoise every non-guard frame without crossing fixed split boundaries."""

    config.validate()
    input_root, output_root, checkpoint_path = Path(input_root), Path(output_root), Path(checkpoint_path)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frames, _, normalizations = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    model, checkpoint = load_model(checkpoint_path, str(device_obj))
    checkpoint_hash = _checkpoint_hash(checkpoint_path)
    records: list[dict] = []
    for (sequence, split_name), group in frames.groupby(["sequence", "split"], sort=True):
        if split_name == "guard" or len(group) < 2 * config.patch_t:
            continue
        group = group.sort_values("frame_index").reset_index(drop=True)
        normalized, registered_valid, clipping = load_registered_stack(
            group,
            input_root,
            detector_mask,
            normalizations[str(sequence)],
            sigma=config.sigma,
            edge_width=config.edge_width,
            temporal_clip=config.temporal_clip,
        )
        model_input = fill_invalid_with_temporal_mean(normalized, registered_valid)
        prediction_norm = denoise_registered_stack(
            model_input,
            model,
            patch_t=config.patch_t,
            device=device_obj,
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            amp=config.amp,
        )
        prediction_registered = (
            prediction_norm * normalizations[str(sequence)]["std"]
            + normalizations[str(sequence)]["mean"]
        ).astype(np.float32)
        for local_index, row in enumerate(group.itertuples(index=False)):
            input_path = input_root / row.product_path
            with fits.open(input_path, memmap=False) as hdul:
                original = np.asarray(hdul[0].data, dtype=np.float32)
                header = hdul[0].header.copy()
            predicted_native = shift(
                prediction_registered[local_index],
                (-float(row.alignment_dy), -float(row.alignment_dx)),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            ).astype(np.float32)
            native_valid = shift(
                registered_valid[local_index].astype(np.float32),
                (-float(row.alignment_dy), -float(row.alignment_dx)),
                order=0,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ) > 0.5
            denoised = predicted_native
            denoised[~native_valid | ~np.isfinite(denoised)] = original[~native_valid | ~np.isfinite(denoised)]
            residual = (original - denoised).astype(np.float32)
            denoised = (original - residual).astype(np.float32)
            dq = build_dq(
                original.shape,
                detector_bad=detector_mask,
                no_coverage=(~native_valid & ~detector_mask) | ~np.isfinite(original),
            )
            sequence_dir = output_root / "raw_predictions" / str(sequence)
            residual_dir = output_root / "raw_residuals" / str(sequence)
            sequence_dir.mkdir(parents=True, exist_ok=True)
            residual_dir.mkdir(parents=True, exist_ok=True)
            denoised_path = sequence_dir / f"asteris_denoised_{row.filename}"
            residual_path = residual_dir / f"asteris_residual_{row.filename}"
            if not overwrite and (denoised_path.exists() or residual_path.exists()):
                raise FileExistsError(f"Refusing to overwrite ASTERIS output for {row.frame_id}")
            for kind, path, data in (
                ("RAW_PREDICTION", denoised_path, denoised),
                ("RAW_RESIDUAL", residual_path, residual),
            ):
                out_header = header.copy()
                out_header["HIERARCH AST MODEL"] = config.model
                out_header["HIERARCH AST KIND"] = kind
                out_header["HIERARCH AST SPLIT"] = str(split_name)
                out_header["HIERARCH AST CKPT"] = checkpoint_hash[:16]
                out_header["HIERARCH AST SIGMA"] = config.sigma
                out_header["HIERARCH AST EQ"] = "DENOISED=INPUT-RESIDUAL"
                write_fits_with_dq(
                    path,
                    data.astype(np.float32),
                    out_header,
                    dq,
                    overwrite=overwrite,
                    output_verify="exception",
                )
            metric_mask = detector_mask | ~native_valid | ~np.isfinite(original) | ~np.isfinite(denoised)
            metric_mask[: config.edge_width] = True
            metric_mask[-config.edge_width :] = True
            metric_mask[:, : config.edge_width] = True
            metric_mask[:, -config.edge_width :] = True
            noise_before = neighbor_difference_noise(original, metric_mask)
            noise_after = neighbor_difference_noise(denoised, metric_mask)
            target = {
                "xc": row.xc,
                "yc": row.yc,
                "r_ap": row.r_ap,
                "r_in": row.r_in,
                "r_out": row.r_out,
            }
            flux_before, snr_before = aperture_flux_snr(original, target)
            flux_after, snr_after = aperture_flux_snr(denoised, target)
            equation_error = float(np.max(np.abs(denoised - (original - residual))))
            records.append(
                {
                    "frame_id": row.frame_id,
                    "sequence": str(sequence),
                    "split": str(split_name),
                    "filename": row.filename,
                    "denoised_path": denoised_path.relative_to(output_root).as_posix(),
                    "residual_path": residual_path.relative_to(output_root).as_posix(),
                    "noise_before": noise_before,
                    "noise_after": noise_after,
                    "noise_ratio": noise_after / noise_before if noise_before > 0 else np.nan,
                    "aperture_flux_before": flux_before,
                    "aperture_flux_after": flux_after,
                    "photometry_change_fraction": (flux_after - flux_before) / flux_before if flux_before else np.nan,
                    "aperture_snr_before": snr_before,
                    "aperture_snr_after": snr_after,
                    "clipped_fraction": clipping.clipped_fraction,
                    "equation_max_abs_error_float32": equation_error,
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
    statistics = pd.DataFrame(records)
    output_root.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(output_root / "raw_inference_statistics.csv", index=False, encoding="utf-8-sig")
    return statistics


def _migrate_legacy_raw_outputs(output_root: Path) -> None:
    """Preserve raw alpha=1 products made before explicit calibration was added."""

    mappings = (
        (output_root / "denoised", output_root / "raw_predictions"),
        (output_root / "residuals", output_root / "raw_residuals"),
        (output_root / "asteris_statistics.csv", output_root / "raw_inference_statistics.csv"),
    )
    for old, new in mappings:
        if old.exists() and not new.exists():
            old.rename(new)
        elif old.exists() and new.exists():
            raise FileExistsError(f"Both legacy and calibrated/raw paths exist: {old}, {new}")


def calibrate_prediction_strength(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
    candidates: tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2)),
    photometry_gate: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """Select the largest alpha passing a 1% high-SNR validation photometry gate."""

    input_root, output_root = Path(input_root), Path(output_root)
    _migrate_legacy_raw_outputs(output_root)
    frames, _, _ = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    validation = frames.loc[
        (frames["split"] == "validation")
        & (pd.to_numeric(frames["input_snr"], errors="coerce") >= 10.0)
    ].copy()
    if validation.empty:
        validation = frames.loc[
            (frames["split"] == "validation") & (frames["sequence"].astype(str) == "90000003")
        ].copy()
    if validation.empty:
        raise RuntimeError("No high-SNR validation frames are available for ASTERIS calibration")
    cache = []
    for row in validation.itertuples(index=False):
        original = np.asarray(fits.getdata(input_root / row.product_path), dtype=np.float32)
        raw_path = output_root / "raw_predictions" / str(row.sequence) / f"asteris_denoised_{row.filename}"
        raw = np.asarray(fits.getdata(raw_path), dtype=np.float32)
        target = {key: getattr(row, key) for key in ("xc", "yc", "r_ap", "r_in", "r_out")}
        flux_before, snr_before = aperture_flux_snr(original, target)
        metric_mask = detector_mask | ~np.isfinite(original) | ~np.isfinite(raw)
        metric_mask[: config.edge_width] = True
        metric_mask[-config.edge_width :] = True
        metric_mask[:, : config.edge_width] = True
        metric_mask[:, -config.edge_width :] = True
        noise_before = neighbor_difference_noise(original, metric_mask)
        cache.append((original, raw, target, metric_mask, flux_before, snr_before, noise_before))
    rows = []
    for alpha in candidates:
        flux_changes, noise_ratios, snr_ratios = [], [], []
        for original, raw, target, metric_mask, flux_before, snr_before, noise_before in cache:
            candidate = (original + float(alpha) * (raw - original)).astype(np.float32)
            flux_after, snr_after = aperture_flux_snr(candidate, target)
            noise_after = neighbor_difference_noise(candidate, metric_mask)
            flux_changes.append((flux_after - flux_before) / flux_before)
            noise_ratios.append(noise_after / max(noise_before, np.finfo(float).eps))
            snr_ratios.append(snr_after / snr_before)
        maximum_change = float(np.nanmax(np.abs(flux_changes)))
        rows.append(
            {
                "strength": float(alpha),
                "validation_frames": len(cache),
                "validation_max_abs_photometry_change": maximum_change,
                "validation_median_noise_ratio": float(np.nanmedian(noise_ratios)),
                "validation_median_snr_ratio": float(np.nanmedian(snr_ratios)),
                "passes_photometry_gate": bool(maximum_change <= photometry_gate),
            }
        )
    calibration = pd.DataFrame(rows)
    passing = calibration.loc[calibration["passes_photometry_gate"]]
    if passing.empty:
        raise RuntimeError("No ASTERIS strength passes the validation photometry gate")
    selected = float(passing.sort_values("strength").iloc[-1]["strength"])
    manifest_root = output_root / "manifests"
    calibration.to_csv(manifest_root / "strength_calibration.csv", index=False, encoding="utf-8-sig")
    (manifest_root / "selected_strength.json").write_text(
        json.dumps(
            {
                "selected_strength": selected,
                "selection_split": "validation",
                "minimum_validation_snr": 10.0,
                "photometry_gate": photometry_gate,
                "selection_rule": "largest strength passing photometry gate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected, calibration


def load_calibrated_strength(output_root: str | Path) -> float:
    path = Path(output_root) / "manifests" / "selected_strength.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_strength"])


def finalize_calibrated_products(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Blend preserved raw predictions with inputs and write final science FITS."""

    input_root, output_root, checkpoint_path = Path(input_root), Path(output_root), Path(checkpoint_path)
    _migrate_legacy_raw_outputs(output_root)
    strength = load_calibrated_strength(output_root)
    frames, _, _ = load_manifests(output_root)
    detector_mask = load_detector_mask(dataset_root)
    checkpoint_hash = _checkpoint_hash(checkpoint_path)
    raw_stats_path = output_root / "raw_inference_statistics.csv"
    raw_stats = (
        pd.read_csv(raw_stats_path, encoding="utf-8-sig").set_index("frame_id")
        if raw_stats_path.exists()
        else pd.DataFrame()
    )
    records = []
    science_frames = frames.loc[frames["split"] != "guard"].sort_values(["sequence", "frame_index"])
    for row in science_frames.itertuples(index=False):
        input_path = input_root / row.product_path
        raw_path = output_root / "raw_predictions" / str(row.sequence) / f"asteris_denoised_{row.filename}"
        with fits.open(input_path, memmap=False) as hdul:
            original = np.asarray(hdul[0].data, dtype=np.float32)
            header = hdul[0].header.copy()
        raw = np.asarray(fits.getdata(raw_path), dtype=np.float32)
        dq = read_dq(raw_path, original.shape)
        if dq is None:
            dq = build_dq(
                original.shape,
                detector_bad=detector_mask,
                no_coverage=~np.isfinite(original) | ~np.isfinite(raw),
            )
        denoised = (original + strength * (raw - original)).astype(np.float32)
        denoised[~np.isfinite(raw)] = original[~np.isfinite(raw)]
        residual = (original - denoised).astype(np.float32)
        denoised = (original - residual).astype(np.float32)
        denoised_path = output_root / "denoised" / str(row.sequence) / f"asteris_denoised_{row.filename}"
        residual_path = output_root / "residuals" / str(row.sequence) / f"asteris_residual_{row.filename}"
        denoised_path.parent.mkdir(parents=True, exist_ok=True)
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and (denoised_path.exists() or residual_path.exists()):
            raise FileExistsError(f"Refusing to overwrite calibrated ASTERIS output for {row.frame_id}")
        for kind, path, data in (
            ("DENOISED", denoised_path, denoised),
            ("RESIDUAL", residual_path, residual),
        ):
            out_header = header.copy()
            out_header["HIERARCH AST MODEL"] = config.model
            out_header["HIERARCH AST KIND"] = kind
            out_header["HIERARCH AST SPLIT"] = str(row.split)
            out_header["HIERARCH AST CKPT"] = checkpoint_hash[:16]
            out_header["HIERARCH AST ALPHA"] = strength
            out_header["HIERARCH AST EQ"] = "DENOISED=INPUT-RESIDUAL"
            write_fits_with_dq(
                path,
                data,
                out_header,
                dq,
                overwrite=overwrite,
                output_verify="exception",
            )
        target = {key: getattr(row, key) for key in ("xc", "yc", "r_ap", "r_in", "r_out")}
        flux_before, snr_before = aperture_flux_snr(original, target)
        flux_after, snr_after = aperture_flux_snr(denoised, target)
        metric_mask = detector_mask | ~np.isfinite(original) | ~np.isfinite(denoised)
        metric_mask[: config.edge_width] = True
        metric_mask[-config.edge_width :] = True
        metric_mask[:, : config.edge_width] = True
        metric_mask[:, -config.edge_width :] = True
        noise_before = neighbor_difference_noise(original, metric_mask)
        noise_after = neighbor_difference_noise(denoised, metric_mask)
        raw_clipped_fraction = (
            float(raw_stats.loc[row.frame_id, "clipped_fraction"])
            if not raw_stats.empty and row.frame_id in raw_stats.index
            else np.nan
        )
        records.append(
            {
                "frame_id": row.frame_id,
                "sequence": str(row.sequence),
                "frame_index": int(row.frame_index),
                "split": str(row.split),
                "filename": row.filename,
                "input_snr": row.input_snr,
                "denoise_strength": strength,
                "denoised_path": denoised_path.relative_to(output_root).as_posix(),
                "residual_path": residual_path.relative_to(output_root).as_posix(),
                "noise_before": noise_before,
                "noise_after": noise_after,
                "noise_ratio": noise_after / max(noise_before, np.finfo(float).eps),
                "aperture_flux_before": flux_before,
                "aperture_flux_after": flux_after,
                "photometry_change_fraction": (flux_after - flux_before) / flux_before,
                "aperture_snr_before": snr_before,
                "aperture_snr_after": snr_after,
                "aperture_snr_ratio": snr_after / snr_before,
                "clipped_fraction": raw_clipped_fraction,
                "equation_max_abs_error_float32": float(
                    np.nanmax(np.abs(denoised - (original - residual)))
                ),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
    statistics = pd.DataFrame(records)
    statistics.to_csv(output_root / "asteris_statistics.csv", index=False, encoding="utf-8-sig")
    return statistics


def calibrate_and_finalize(
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    *,
    config: AsterisConfig = AsterisConfig(),
    overwrite: bool = False,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    strength, calibration = calibrate_prediction_strength(
        input_root, dataset_root, output_root, config=config
    )
    statistics = finalize_calibrated_products(
        input_root,
        dataset_root,
        output_root,
        checkpoint_path,
        config=config,
        overwrite=overwrite,
    )
    return strength, calibration, statistics
