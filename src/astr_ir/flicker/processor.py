"""Robust row/column 1/f stripe correction for the RKZ50 FITS dataset.

The module keeps the scientific workflow explicit so that the companion
notebook can expose every stage independently.  Detector blind maps are used
directly as an exclusion mask; no separate DQ product is created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    median_filter,
    zoom,
)


Direction = Literal["auto", "row", "column"]


@dataclass(frozen=True)
class FlickerConfig:
    """Parameters controlling masking, modelling, and quality gates."""

    direction: Direction = "auto"
    edge_width: int = 24
    background_block_size: int = 64
    background_smooth_sigma_blocks: float = 1.25
    sigma_clip: float = 3.0
    sigma_clip_iters: int = 5
    profile_smooth_size: int = 5
    source_sigma: float = 6.0
    source_dilation: int = 8
    known_source_radius_scale: float = 1.25
    min_direction_score: float = 1.6
    min_relative_improvement: float = 0.30
    max_noise_increase: float = 0.02
    photometry_gate_snr: float = 10.0
    max_photometry_change: float = 0.01

    def validate(self) -> None:
        if self.direction not in {"auto", "row", "column"}:
            raise ValueError("direction must be 'auto', 'row', or 'column'")
        if self.edge_width < 0 or self.background_block_size < 4:
            raise ValueError("edge_width must be >= 0 and block size >= 4")
        if self.profile_smooth_size < 1 or self.profile_smooth_size % 2 == 0:
            raise ValueError("profile_smooth_size must be a positive odd integer")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


@dataclass
class DirectionDiagnostic:
    direction: Literal["row", "column"]
    profile: np.ndarray
    line_scatter: np.ndarray
    valid_count: np.ndarray
    robust_std: float
    noise_floor: float
    signal_std: float
    score: float


@dataclass
class CorrectionResult:
    original: np.ndarray
    corrected: np.ndarray
    flicker_model: np.ndarray
    low_frequency_background: np.ndarray
    residual: np.ndarray
    combined_mask: np.ndarray
    detector_mask: np.ndarray
    source_mask: np.ndarray
    edge_mask: np.ndarray
    row_diagnostic: DirectionDiagnostic
    column_diagnostic: DirectionDiagnostic
    selected_direction: Literal["row", "column"]
    smoothed_profile: np.ndarray
    applied: bool
    status: str
    metrics: dict[str, float | str | bool]


def robust_std(values: np.ndarray) -> float:
    """Return a NaN-safe Gaussian-equivalent MAD."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def load_fits(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load scaled science data and repair non-standard cards in memory only."""

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            hdul.verify("silentfix")
            data = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image, got {data.shape} from {path}")
    return data, header


def load_detector_mask(blind_map_dir: str | Path) -> np.ndarray:
    """Union DeadBlindMap and NoiseBlindMap directly (no redundant DQ array)."""

    blind_map_dir = Path(blind_map_dir)
    masks: list[np.ndarray] = []
    for name in ("DeadBlindMap.tiff", "NoiseBlindMap.tiff"):
        path = blind_map_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing detector blind map: {path}")
        masks.append(np.asarray(Image.open(path)) != 0)
    if masks[0].shape != masks[1].shape:
        raise ValueError("Dead and noise blind maps have different shapes")
    return np.logical_or.reduce(masks)


def load_measurement_table(csv_path: str | Path) -> pd.DataFrame:
    """Load the UTF-8-BOM per-frame target measurement table."""

    return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


def target_record_for_file(table: pd.DataFrame, filename: str) -> dict | None:
    rows = table.loc[table["filename"] == filename]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(f"Expected one target record for {filename}, got {len(rows)}")
    record = rows.iloc[0].to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in record.items()}


def make_edge_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if width <= 0:
        return mask
    width = min(width, min(shape) // 2)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def make_known_source_mask(
    shape: tuple[int, int],
    target: Mapping | None,
    radius_scale: float = 1.25,
) -> np.ndarray:
    """Create a circular target mask from the CSV's 1-based FITS coordinates."""

    mask = np.zeros(shape, dtype=bool)
    if not target or target.get("xc") is None or target.get("yc") is None:
        return mask
    x = float(target["xc"]) - 1.0
    y = float(target["yc"]) - 1.0
    candidates = [12.0]
    if target.get("r_out") is not None:
        candidates.append(float(target["r_out"]) * radius_scale)
    if target.get("fwhm") is not None:
        candidates.append(float(target["fwhm"]) * 3.0)
    radius = max(candidates)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius**2


def make_auto_source_mask(
    image: np.ndarray,
    base_mask: np.ndarray,
    sigma_threshold: float = 6.0,
    dilation: int = 8,
) -> np.ndarray:
    """Detect compact positive sources in a small-scale high-pass image."""

    smooth = gaussian_filter(image, sigma=3.0, mode="reflect")
    highpass = image - smooth
    sigma = robust_std(highpass[~base_mask])
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(image.shape, dtype=bool)
    source = (highpass > sigma_threshold * sigma) & ~base_mask
    if dilation > 0 and np.any(source):
        source = binary_dilation(source, iterations=dilation)
    return source


def combine_masks(
    image: np.ndarray,
    detector_mask: np.ndarray | None,
    target: Mapping | None,
    config: FlickerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine detector blind-map, source, edge, and invalid-pixel masks."""

    shape = image.shape
    if detector_mask is None:
        detector = np.zeros(shape, dtype=bool)
    else:
        detector = np.asarray(detector_mask, dtype=bool)
        if detector.shape != shape:
            raise ValueError(f"Detector mask {detector.shape} != image {shape}")
    edge = make_edge_mask(shape, config.edge_width)
    invalid = ~np.isfinite(image)
    known = make_known_source_mask(shape, target, config.known_source_radius_scale)
    preliminary = detector | edge | invalid | known
    automatic = make_auto_source_mask(
        image,
        preliminary,
        sigma_threshold=config.source_sigma,
        dilation=config.source_dilation,
    )
    source = known | automatic
    combined = detector | edge | invalid | source
    return combined, detector, source, edge


def _fill_missing_grid(grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        raise ValueError("All low-frequency background blocks are masked")
    if np.all(valid):
        return grid
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = grid.copy()
    filled[~valid] = grid[tuple(indices[:, ~valid])]
    return filled


def estimate_low_frequency_background(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int = 64,
    smooth_sigma_blocks: float = 1.25,
) -> np.ndarray:
    """Estimate a robust coarse 2-D background and interpolate to full size."""

    h, w = image.shape
    ny = int(np.ceil(h / block_size))
    nx = int(np.ceil(w / block_size))
    coarse = np.full((ny, nx), np.nan, dtype=np.float64)
    for by in range(ny):
        y0, y1 = by * block_size, min((by + 1) * block_size, h)
        for bx in range(nx):
            x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
            values = image[y0:y1, x0:x1][~mask[y0:y1, x0:x1]]
            values = values[np.isfinite(values)]
            if values.size >= 16:
                coarse[by, bx] = np.median(values)
    coarse = _fill_missing_grid(coarse)
    coarse = gaussian_filter(coarse, sigma=smooth_sigma_blocks, mode="nearest")
    background = zoom(coarse, (h / ny, w / nx), order=3, mode="nearest")
    return background[:h, :w]


def sigma_clipped_profile(
    residual: np.ndarray,
    mask: np.ndarray,
    direction: Literal["row", "column"],
    sigma: float = 3.0,
    maxiters: int = 5,
) -> DirectionDiagnostic:
    """Calculate a robust profile and its expected median uncertainty."""

    axis = 1 if direction == "row" else 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, profile, scatter = sigma_clipped_stats(
            residual,
            mask=mask,
            sigma=sigma,
            maxiters=maxiters,
            axis=axis,
        )
    profile = np.asarray(profile, dtype=np.float64)
    scatter = np.asarray(scatter, dtype=np.float64)
    valid_count = np.sum(~mask & np.isfinite(residual), axis=axis).astype(float)
    profile_std = robust_std(profile)
    uncertainty = 1.2533 * scatter / np.sqrt(np.maximum(valid_count, 1.0))
    finite_uncertainty = uncertainty[np.isfinite(uncertainty) & (uncertainty > 0)]
    noise_floor = float(np.median(finite_uncertainty)) if finite_uncertainty.size else 0.0
    signal_std = float(np.sqrt(max(profile_std**2 - noise_floor**2, 0.0)))
    score = float(profile_std / max(noise_floor, np.finfo(float).eps))
    return DirectionDiagnostic(
        direction=direction,
        profile=profile,
        line_scatter=scatter,
        valid_count=valid_count,
        robust_std=profile_std,
        noise_floor=noise_floor,
        signal_std=signal_std,
        score=score,
    )


def choose_direction(
    row: DirectionDiagnostic,
    column: DirectionDiagnostic,
    requested: Direction,
) -> DirectionDiagnostic:
    if requested == "row":
        return row
    if requested == "column":
        return column
    return row if row.signal_std >= column.signal_std else column


def smooth_profile(profile: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(profile, dtype=np.float64).copy()
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if not np.all(good):
        idx = np.arange(x.size)
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    if size > 1:
        x = median_filter(x, size=size, mode="reflect")
    return x - np.median(x)


def expand_profile(profile: np.ndarray, shape: tuple[int, int], direction: str) -> np.ndarray:
    if direction == "row":
        if len(profile) != shape[0]:
            raise ValueError("Row profile length does not match image height")
        return np.broadcast_to(profile[:, None], shape).copy()
    if direction == "column":
        if len(profile) != shape[1]:
            raise ValueError("Column profile length does not match image width")
        return np.broadcast_to(profile[None, :], shape).copy()
    raise ValueError("direction must be row or column")


def aperture_photometry(image: np.ndarray, target: Mapping | None) -> float:
    """Background-subtracted circular aperture flux using CSV radii."""

    if not target or any(target.get(k) is None for k in ("xc", "yc", "r_ap", "r_in", "r_out")):
        return float("nan")
    x = float(target["xc"]) - 1.0
    y = float(target["yc"]) - 1.0
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    radius = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    aperture = radius <= float(target["r_ap"])
    annulus = (radius >= float(target["r_in"])) & (radius <= float(target["r_out"]))
    annulus_values = image[annulus & np.isfinite(image)]
    if not np.any(aperture) or annulus_values.size == 0:
        return float("nan")
    background = float(np.median(annulus_values))
    return float(np.sum(image[aperture] - background))


def _background_noise(image: np.ndarray, mask: np.ndarray) -> float:
    highpass = image - gaussian_filter(image, sigma=2.0, mode="reflect")
    return robust_std(highpass[~mask])


def correct_flicker(
    image: np.ndarray,
    detector_mask: np.ndarray | None = None,
    target: Mapping | None = None,
    config: FlickerConfig | None = None,
) -> CorrectionResult:
    """Run the complete correction with weak-signal and validation gates."""

    config = config or FlickerConfig()
    config.validate()
    original = np.asarray(image, dtype=np.float64)
    if original.ndim != 2:
        raise ValueError("correct_flicker expects a 2-D image")

    combined, detector, source, edge = combine_masks(original, detector_mask, target, config)
    background = estimate_low_frequency_background(
        original,
        combined,
        block_size=config.background_block_size,
        smooth_sigma_blocks=config.background_smooth_sigma_blocks,
    )
    residual = original - background
    row = sigma_clipped_profile(
        residual, combined, "row", config.sigma_clip, config.sigma_clip_iters
    )
    column = sigma_clipped_profile(
        residual, combined, "column", config.sigma_clip, config.sigma_clip_iters
    )
    selected = choose_direction(row, column, config.direction)
    profile = smooth_profile(selected.profile, config.profile_smooth_size)
    candidate_model = expand_profile(profile, original.shape, selected.direction)
    candidate = original - candidate_model

    before_metric = selected.robust_std
    after_diag = sigma_clipped_profile(
        candidate - background,
        combined,
        selected.direction,
        config.sigma_clip,
        config.sigma_clip_iters,
    )
    candidate_reduction = float(
        1.0 - after_diag.robust_std / max(before_metric, np.finfo(float).eps)
    )
    noise_before = _background_noise(original, combined)
    noise_after_candidate = _background_noise(candidate, combined)
    noise_ratio_candidate = float(noise_after_candidate / max(noise_before, np.finfo(float).eps))
    flux_before = aperture_photometry(original, target)
    flux_after_candidate = aperture_photometry(candidate, target)
    flux_change_candidate = (
        float((flux_after_candidate - flux_before) / flux_before)
        if np.isfinite(flux_before) and flux_before != 0 and np.isfinite(flux_after_candidate)
        else float("nan")
    )

    detected = selected.score >= config.min_direction_score
    improves = candidate_reduction >= config.min_relative_improvement
    noise_ok = noise_ratio_candidate <= 1.0 + config.max_noise_increase
    target_snr = float(target.get("snr", np.nan)) if target else float("nan")
    photometry_gate_active = np.isfinite(target_snr) and target_snr >= config.photometry_gate_snr
    photometry_ok = (
        not photometry_gate_active
        or not np.isfinite(flux_change_candidate)
        or abs(flux_change_candidate) <= config.max_photometry_change
    )

    if not detected:
        applied = False
        status = "not_needed_weak_stripe"
    elif not improves:
        applied = False
        status = "rejected_insufficient_improvement"
    elif not noise_ok:
        applied = False
        status = "rejected_noise_increase"
    elif not photometry_ok:
        applied = False
        status = "rejected_photometry_change"
    else:
        applied = True
        status = "corrected"

    if applied:
        model = candidate_model
        corrected = candidate
        after_metric = after_diag.robust_std
        reduction = candidate_reduction
        noise_after = noise_after_candidate
        noise_ratio = noise_ratio_candidate
        flux_after = flux_after_candidate
        flux_change = flux_change_candidate
    else:
        model = np.zeros_like(original)
        corrected = original.copy()
        after_metric = before_metric
        reduction = 0.0
        noise_after = noise_before
        noise_ratio = 1.0
        flux_after = flux_before
        flux_change = 0.0 if np.isfinite(flux_before) else float("nan")

    metrics: dict[str, float | str | bool] = {
        "direction_requested": config.direction,
        "selected_direction": selected.direction,
        "applied": applied,
        "status": status,
        "row_score": row.score,
        "column_score": column.score,
        "direction_score": selected.score,
        "profile_rstd_before": before_metric,
        "profile_rstd_after": after_metric,
        "relative_reduction": reduction,
        "candidate_relative_reduction": candidate_reduction,
        "background_noise_before": noise_before,
        "background_noise_after": noise_after,
        "background_noise_ratio": noise_ratio,
        "candidate_background_noise_ratio": noise_ratio_candidate,
        "photometry_before": flux_before,
        "photometry_after": flux_after,
        "photometry_change_fraction": flux_change,
        "candidate_photometry_change_fraction": flux_change_candidate,
        "photometry_gate_active": photometry_gate_active,
        "mask_fraction": float(np.mean(combined)),
        "detector_mask_fraction": float(np.mean(detector)),
        "source_mask_fraction": float(np.mean(source)),
        "edge_mask_fraction": float(np.mean(edge)),
    }
    return CorrectionResult(
        original=original,
        corrected=corrected,
        flicker_model=model,
        low_frequency_background=background,
        residual=residual,
        combined_mask=combined,
        detector_mask=detector,
        source_mask=source,
        edge_mask=edge,
        row_diagnostic=row,
        column_diagnostic=column,
        selected_direction=selected.direction,
        smoothed_profile=profile,
        applied=applied,
        status=status,
        metrics=metrics,
    )


def _output_header(
    header: fits.Header,
    result: CorrectionResult,
    product: Literal["corrected", "model"],
) -> fits.Header:
    out = header.copy()
    out["HIERARCH FLK PROD"] = product
    out["HIERARCH FLK DIR"] = result.selected_direction
    out["HIERARCH FLK APPL"] = bool(result.applied)
    out["HIERARCH FLK SCORE"] = round(float(result.metrics["direction_score"]), 6)
    out["HIERARCH FLK REDUC"] = round(float(result.metrics["relative_reduction"]), 6)
    out.add_history("1/f correction implemented by astr_ir.flicker.processor")
    out.add_history("Detector exclusion mask is DeadBlindMap OR NoiseBlindMap; no DQ product.")
    return out


def write_fits_products(
    input_path: str | Path,
    output_dir: str | Path,
    header: fits.Header,
    result: CorrectionResult,
    overwrite: bool = False,
) -> tuple[Path, Path, float]:
    """Write float32 products with the repaired original scientific header."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / f"flicker_corrected_{input_path.name}"
    model_path = output_dir / f"flicker_model_{input_path.name}"
    original32 = result.original.astype(np.float32)
    model32 = result.flicker_model.astype(np.float32)
    corrected32 = original32 - model32
    equation_error = float(np.max(np.abs(corrected32 - (original32 - model32))))
    fits.PrimaryHDU(corrected32, header=_output_header(header, result, "corrected")).writeto(
        corrected_path, overwrite=overwrite, output_verify="silentfix"
    )
    fits.PrimaryHDU(model32, header=_output_header(header, result, "model")).writeto(
        model_path, overwrite=overwrite, output_verify="silentfix"
    )
    return corrected_path, model_path, equation_error


def process_fits_file(
    input_path: str | Path,
    output_dir: str | Path,
    detector_mask: np.ndarray,
    target: Mapping | None,
    config: FlickerConfig,
    overwrite: bool = False,
) -> tuple[CorrectionResult, dict]:
    image, header = load_fits(input_path)
    result = correct_flicker(image, detector_mask=detector_mask, target=target, config=config)
    corrected_path, model_path, equation_error = write_fits_products(
        input_path, output_dir, header, result, overwrite=overwrite
    )
    row = {
        "filename": Path(input_path).name,
        **result.metrics,
        "equation_max_abs_error_float32": equation_error,
        "corrected_path": str(corrected_path),
        "model_path": str(model_path),
    }
    if target:
        row.update(
            {
                "star_id": target.get("star_id"),
                "input_status": target.get("status"),
                "input_snr": target.get("snr"),
            }
        )
    return result, row


def run_batch(
    dataset_root: str | Path,
    output_root: str | Path,
    config: FlickerConfig | None = None,
    sequences: tuple[str, ...] = ("90000002", "90000003"),
    overwrite: bool = False,
    limit_per_sequence: int | None = None,
) -> pd.DataFrame:
    """Process raw FITS files and write products plus flicker_statistics.csv."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    config = config or FlickerConfig()
    detector_mask = load_detector_mask(dataset_root / "盲点表")
    table = load_measurement_table(dataset_root / "单帧检测总表_新方法.csv")
    rows: list[dict] = []
    for sequence in sequences:
        files = sorted((dataset_root / sequence).glob("*.fits"))
        if limit_per_sequence is not None:
            files = files[:limit_per_sequence]
        sequence_output = output_root / sequence
        for index, path in enumerate(files, start=1):
            target = target_record_for_file(table, path.name)
            _, row = process_fits_file(
                path,
                sequence_output,
                detector_mask,
                target,
                config,
                overwrite=overwrite,
            )
            row["sequence"] = sequence
            row["sequence_frame_index"] = index
            row["corrected_path"] = Path(row["corrected_path"]).relative_to(output_root).as_posix()
            row["model_path"] = Path(row["model_path"]).relative_to(output_root).as_posix()
            rows.append(row)
    stats = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / "flicker_statistics.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    return stats


def config_as_dict(config: FlickerConfig) -> dict:
    return asdict(config)
