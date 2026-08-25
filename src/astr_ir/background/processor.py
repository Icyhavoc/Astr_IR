"""Source-masked two-dimensional background subtraction for RKZ50 FITS images.

The implementation adapts the CEERS workflow (Bagley et al. 2023): a coarse
background and clipped annular estimate flatten the image for source finding,
four source-detection tiers build a heavy mask, and a robust gridded model is
interpolated and subtracted.  The science product always obeys, at float32
precision, ``background_subtracted = input - background_model``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import biweight_location, sigma_clip
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
    median_filter,
)
from scipy.signal import fftconvolve


@dataclass
class BackgroundConfig:
    """Parameters for masking, modelling, and conservative quality gates."""

    edge_width: int = 24
    rough_box_size: int = 100
    rough_filter_size: int = 3
    rough_sigma_clip: float = 3.0
    rough_exclude_percentile: float = 90.0
    ring_inner_radius: int = 80
    ring_width: int = 4
    bright_sigma: float = 5.0
    tier_gaussian_sigmas: tuple[float, ...] = (25.0, 15.0, 5.0, 2.0)
    tier_threshold_sigmas: tuple[float, ...] = (3.0, 3.2, 3.8, 5.0)
    tier_min_pixels: tuple[int, ...] = (15, 10, 3, 1)
    tier_dilation_radii: tuple[int, ...] = (24, 16, 10, 6)
    known_source_radius_scale: float = 1.5
    final_box_size: int = 32
    final_filter_size: int = 5
    final_sigma_clip: float = 3.0
    final_exclude_percentile: float = 90.0
    validation_block_size: int = 64
    min_large_scale_reduction: float = 0.10
    max_noise_increase: float = 0.02
    photometry_gate_snr: float = 10.0
    max_photometry_change: float = 0.01

    def validate(self) -> None:
        if self.edge_width < 0:
            raise ValueError("edge_width must be non-negative")
        if min(self.rough_box_size, self.final_box_size) < 4:
            raise ValueError("background box sizes must be at least 4 pixels")
        if self.ring_inner_radius < 1 or self.ring_width < 1:
            raise ValueError("ring radii must be positive")
        if self.rough_sigma_clip <= 0 or self.final_sigma_clip <= 0 or self.bright_sigma <= 0:
            raise ValueError("sigma thresholds must be positive")
        if not 0 <= self.rough_exclude_percentile <= 100:
            raise ValueError("rough_exclude_percentile must be in [0, 100]")
        if not 0 <= self.final_exclude_percentile <= 100:
            raise ValueError("final_exclude_percentile must be in [0, 100]")
        sizes = {
            len(self.tier_gaussian_sigmas),
            len(self.tier_threshold_sigmas),
            len(self.tier_min_pixels),
            len(self.tier_dilation_radii),
        }
        if len(sizes) != 1 or next(iter(sizes)) == 0:
            raise ValueError("all source-tier parameter tuples must have equal non-zero length")
        if any(value <= 0 for value in self.tier_gaussian_sigmas):
            raise ValueError("tier Gaussian sigmas must be positive")
        if any(value <= 0 for value in self.tier_threshold_sigmas):
            raise ValueError("tier thresholds must be positive")
        if any(value < 1 for value in self.tier_min_pixels):
            raise ValueError("tier minimum component sizes must be positive")
        if any(value < 0 for value in self.tier_dilation_radii):
            raise ValueError("tier dilation radii must be non-negative")
        if any(size < 1 or size % 2 == 0 for size in (self.rough_filter_size, self.final_filter_size)):
            raise ValueError("background filter sizes must be positive odd integers")
        if not 0 <= self.min_large_scale_reduction < 1:
            raise ValueError("min_large_scale_reduction must be in [0, 1)")
        if self.validation_block_size < 1 or self.max_noise_increase < 0:
            raise ValueError("validation block size must be positive and noise tolerance non-negative")
        if self.photometry_gate_snr < 0 or not 0 <= self.max_photometry_change < 1:
            raise ValueError("photometry thresholds are outside their valid range")


@dataclass
class BackgroundResult:
    original: np.ndarray
    background_subtracted: np.ndarray
    background_model: np.ndarray
    rough_background: np.ndarray
    ring_background: np.ndarray
    detection_residual: np.ndarray
    detector_mask: np.ndarray
    edge_mask: np.ndarray
    known_source_mask: np.ndarray
    tier_masks: tuple[np.ndarray, ...]
    source_mask: np.ndarray
    combined_mask: np.ndarray
    applied: bool
    status: str
    metrics: dict[str, float | str | bool]


def robust_std(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    center = np.median(x)
    return float(1.4826 * np.median(np.abs(x - center)))


def load_fits(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            hdul.verify("silentfix")
            image = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image, got {image.shape} from {path}")
    return image, header


def load_detector_mask(blind_map_dir: str | Path) -> np.ndarray:
    blind_map_dir = Path(blind_map_dir)
    masks = []
    for name in ("DeadBlindMap.tiff", "NoiseBlindMap.tiff"):
        path = blind_map_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing detector blind map: {path}")
        masks.append(np.asarray(Image.open(path)) != 0)
    if masks[0].shape != masks[1].shape:
        raise ValueError("Dead and noise blind maps have different shapes")
    return masks[0] | masks[1]


def load_measurement_table(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


def raw_filename_from_corrected(filename: str) -> str:
    prefix = "flicker_corrected_"
    return filename[len(prefix) :] if filename.startswith(prefix) else filename


def target_record_for_file(table: pd.DataFrame, filename: str) -> dict | None:
    raw_name = raw_filename_from_corrected(Path(filename).name)
    rows = table.loc[table["filename"] == raw_name]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(f"Expected one target record for {raw_name}, got {len(rows)}")
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
    shape: tuple[int, int], target: Mapping | None, radius_scale: float = 1.5
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not target or target.get("xc") is None or target.get("yc") is None:
        return mask
    x = float(target["xc"]) - 1.0
    y = float(target["yc"]) - 1.0
    radii = [12.0]
    if target.get("r_out") is not None:
        radii.append(float(target["r_out"]) * radius_scale)
    if target.get("fwhm") is not None:
        radii.append(float(target["fwhm"]) * 3.0)
    radius = max(radii)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius**2


def _fill_missing_grid(grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        raise ValueError("All background grid boxes are masked")
    if np.all(valid):
        return grid
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = grid.copy()
    filled[~valid] = grid[tuple(nearest[:, ~valid])]
    return filled


def _robust_location(values: np.ndarray, sigma: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    clipped = sigma_clip(values, sigma=sigma, maxiters=5, masked=True)
    kept = np.asarray(clipped.data[~clipped.mask] if np.ndim(clipped.mask) else clipped.data)
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return float("nan")
    value = float(biweight_location(kept))
    return value if np.isfinite(value) else float(np.median(kept))


def estimate_grid_background(
    image: np.ndarray,
    mask: np.ndarray,
    box_size: int,
    filter_size: int,
    sigma: float,
    exclude_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust block background, median-filtered and spline-interpolated."""

    h, w = image.shape
    ny, nx = int(np.ceil(h / box_size)), int(np.ceil(w / box_size))
    coarse = np.full((ny, nx), np.nan, dtype=np.float64)
    y_centers = np.empty(ny, dtype=float)
    x_centers = np.empty(nx, dtype=float)
    for by in range(ny):
        y0, y1 = by * box_size, min((by + 1) * box_size, h)
        y_centers[by] = 0.5 * (y0 + y1 - 1)
        for bx in range(nx):
            x0, x1 = bx * box_size, min((bx + 1) * box_size, w)
            if by == 0:
                x_centers[bx] = 0.5 * (x0 + x1 - 1)
            block_mask = mask[y0:y1, x0:x1] | ~np.isfinite(image[y0:y1, x0:x1])
            masked_fraction = 100.0 * np.mean(block_mask)
            values = image[y0:y1, x0:x1][~block_mask]
            if masked_fraction <= exclude_percentile and values.size >= 16:
                coarse[by, bx] = _robust_location(values, sigma)
    coarse = _fill_missing_grid(coarse)
    if filter_size > 1:
        coarse = median_filter(coarse, size=filter_size, mode="nearest")
    ky, kx = min(3, ny - 1), min(3, nx - 1)
    spline = RectBivariateSpline(y_centers, x_centers, coarse, kx=ky, ky=kx)
    yy = np.clip(np.arange(h, dtype=float), y_centers[0], y_centers[-1])
    xx = np.clip(np.arange(w, dtype=float), x_centers[0], x_centers[-1])
    background = spline(yy, xx)
    return background, coarse


def _annulus_kernel(inner_radius: int, width: int) -> np.ndarray:
    outer = inner_radius + width
    yy, xx = np.ogrid[-outer : outer + 1, -outer : outer + 1]
    rr2 = xx**2 + yy**2
    kernel = ((rr2 >= inner_radius**2) & (rr2 <= outer**2)).astype(np.float64)
    kernel /= np.sum(kernel)
    return kernel


def estimate_clipped_ring_background(
    image: np.ndarray,
    rough_background: np.ndarray,
    base_mask: np.ndarray,
    inner_radius: int,
    width: int,
    bright_sigma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Efficient clipped annular estimate analogous to CEERS ring filtering.

    CEERS used an exact masked ring median.  For 160 native 1024-square frames,
    the residual is first robustly clipped and bright-source masked, then an
    annular normalized convolution estimates the same large spatial scale
    without an impractical per-pixel 160-pixel median footprint.
    """

    residual = image - rough_background
    valid = ~base_mask & np.isfinite(residual)
    center = float(np.median(residual[valid]))
    scale = robust_std(residual[valid])
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(residual[valid]))
    bright = residual > center + bright_sigma * scale
    valid &= ~bright
    clipped = np.clip(residual - center, -bright_sigma * scale, bright_sigma * scale)
    kernel = _annulus_kernel(inner_radius, width)
    numerator = fftconvolve(np.where(valid, clipped, 0.0), kernel, mode="same")
    denominator = fftconvolve(valid.astype(float), kernel, mode="same")
    ring_component = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.10,
    )
    ring_background = rough_background + center + ring_component
    return ring_background, bright, scale


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx**2 + yy**2 <= radius**2


def _masked_gaussian(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    valid = ~mask & np.isfinite(image)
    numerator = gaussian_filter(np.where(valid, image, 0.0), sigma=sigma, mode="reflect")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="reflect")
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.05)


def _remove_small_components(binary: np.ndarray, min_pixels: int) -> np.ndarray:
    components, count = label(binary)
    if count == 0:
        return np.zeros_like(binary)
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[components]


def make_tiered_source_mask(
    detection_residual: np.ndarray,
    base_mask: np.ndarray,
    initial_source_mask: np.ndarray,
    config: BackgroundConfig,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Build cumulative source masks from broad to compact spatial scales."""

    cumulative = initial_source_mask.copy()
    tiers: list[np.ndarray] = []
    for smooth_sigma, threshold_sigma, min_pixels, dilation_radius in zip(
        config.tier_gaussian_sigmas,
        config.tier_threshold_sigmas,
        config.tier_min_pixels,
        config.tier_dilation_radii,
    ):
        exclusion = base_mask | cumulative
        smoothed = _masked_gaussian(detection_residual, exclusion, smooth_sigma)
        values = smoothed[~exclusion & np.isfinite(smoothed)]
        center = float(np.median(values)) if values.size else 0.0
        scale = robust_std(values)
        if not np.isfinite(scale) or scale <= 0:
            detected = np.zeros_like(cumulative)
        else:
            detected = (smoothed > center + threshold_sigma * scale) & ~exclusion
            detected = _remove_small_components(detected, min_pixels)
        grown = binary_dilation(detected, structure=_disk(dilation_radius)) if np.any(detected) else detected
        cumulative |= grown
        tiers.append(cumulative.copy())
    return cumulative, tuple(tiers)


def aperture_photometry(image: np.ndarray, target: Mapping | None) -> float:
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
    local_background = float(np.median(annulus_values))
    return float(np.sum(image[aperture] - local_background))


def neighbor_difference_noise(image: np.ndarray, mask: np.ndarray) -> float:
    horizontal_ok = ~mask[:, 1:] & ~mask[:, :-1]
    vertical_ok = ~mask[1:, :] & ~mask[:-1, :]
    values = np.concatenate(
        [
            (image[:, 1:] - image[:, :-1])[horizontal_ok],
            (image[1:, :] - image[:-1, :])[vertical_ok],
        ]
    )
    return robust_std(values) / np.sqrt(2.0)


def block_location_scatter(
    image: np.ndarray, mask: np.ndarray, block_size: int, max_mask_fraction: float = 0.20
) -> float:
    h, w = image.shape
    locations = []
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        for x0 in range(0, w, block_size):
            x1 = min(x0 + block_size, w)
            m = mask[y0:y1, x0:x1] | ~np.isfinite(image[y0:y1, x0:x1])
            if np.mean(m) <= max_mask_fraction:
                values = image[y0:y1, x0:x1][~m]
                if values.size >= 16:
                    locations.append(np.median(values))
    return robust_std(np.asarray(locations))


def equivalent_rms_curve(
    image: np.ndarray,
    mask: np.ndarray,
    block_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    max_mask_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """CEERS-style equivalent per-pixel RMS from block-mean scatter."""

    sizes, rms = [], []
    h, w = image.shape
    for size in block_sizes:
        means = []
        for y0 in range(0, h - size + 1, size):
            for x0 in range(0, w - size + 1, size):
                block = image[y0 : y0 + size, x0 : x0 + size]
                m = mask[y0 : y0 + size, x0 : x0 + size] | ~np.isfinite(block)
                if np.mean(m) <= max_mask_fraction:
                    values = block[~m]
                    if values.size >= max(1, int(0.8 * size * size)):
                        means.append(np.mean(values))
        scatter = robust_std(np.asarray(means))
        sizes.append(size)
        rms.append(scatter * size if np.isfinite(scatter) else np.nan)
    return np.asarray(sizes), np.asarray(rms)


def _source_edge_bias(image: np.ndarray, source_mask: np.ndarray, base_mask: np.ndarray) -> float:
    if not np.any(source_mask):
        return float("nan")
    distance = distance_transform_edt(~source_mask)
    ring = (distance > 0) & (distance <= 5) & ~base_mask
    far = (distance >= 15) & (distance <= 30) & ~base_mask
    if np.count_nonzero(ring) < 20 or np.count_nonzero(far) < 20:
        return float("nan")
    return float(np.median(image[ring]) - np.median(image[far]))


def subtract_background(
    image: np.ndarray,
    detector_mask: np.ndarray | None = None,
    target: Mapping | None = None,
    config: BackgroundConfig | None = None,
) -> BackgroundResult:
    config = config or BackgroundConfig()
    config.validate()
    original = np.asarray(image, dtype=np.float64)
    if original.ndim != 2:
        raise ValueError("subtract_background expects a 2-D image")
    invalid = ~np.isfinite(original)
    detector = np.zeros_like(original, dtype=bool) if detector_mask is None else np.asarray(detector_mask, bool)
    if detector.shape != original.shape:
        raise ValueError(f"Detector mask {detector.shape} != image {original.shape}")
    edge = make_edge_mask(original.shape, config.edge_width)
    known = make_known_source_mask(original.shape, target, config.known_source_radius_scale)
    base = detector | edge | invalid

    rough, _ = estimate_grid_background(
        original,
        base | known,
        config.rough_box_size,
        config.rough_filter_size,
        config.rough_sigma_clip,
        config.rough_exclude_percentile,
    )
    ring_background, bright, detection_scale = estimate_clipped_ring_background(
        original,
        rough,
        base | known,
        config.ring_inner_radius,
        config.ring_width,
        config.bright_sigma,
    )
    detection_residual = original - ring_background
    initial_source = known | binary_dilation(bright, structure=_disk(2))
    source, tiers = make_tiered_source_mask(detection_residual, base, initial_source, config)
    combined = base | source
    candidate_model, _ = estimate_grid_background(
        original,
        combined,
        config.final_box_size,
        config.final_filter_size,
        config.final_sigma_clip,
        config.final_exclude_percentile,
    )
    candidate = original - candidate_model

    before_scatter = block_location_scatter(original, combined, config.validation_block_size)
    after_scatter_candidate = block_location_scatter(candidate, combined, config.validation_block_size)
    reduction_candidate = float(
        1.0 - after_scatter_candidate / max(before_scatter, np.finfo(float).eps)
    )
    noise_before = neighbor_difference_noise(original, combined)
    noise_after_candidate = neighbor_difference_noise(candidate, combined)
    noise_ratio_candidate = float(noise_after_candidate / max(noise_before, np.finfo(float).eps))
    flux_before = aperture_photometry(original, target)
    flux_after_candidate = aperture_photometry(candidate, target)
    flux_change_candidate = (
        float((flux_after_candidate - flux_before) / flux_before)
        if np.isfinite(flux_before) and flux_before != 0 and np.isfinite(flux_after_candidate)
        else float("nan")
    )
    target_snr = float(target.get("snr", np.nan)) if target else float("nan")
    photometry_gate_active = np.isfinite(target_snr) and target_snr >= config.photometry_gate_snr
    photometry_verifiable = bool(
        np.isfinite(flux_before) and flux_before != 0 and np.isfinite(flux_after_candidate)
    )
    photometry_ok = bool(
        not photometry_gate_active
        or (photometry_verifiable and abs(flux_change_candidate) <= config.max_photometry_change)
    )
    finite_model = bool(np.all(np.isfinite(candidate_model)))
    improves = np.isfinite(reduction_candidate) and reduction_candidate >= config.min_large_scale_reduction
    noise_ok = np.isfinite(noise_ratio_candidate) and noise_ratio_candidate <= 1 + config.max_noise_increase

    if not finite_model:
        applied, status = False, "rejected_nonfinite_background"
    elif not improves:
        applied, status = False, "rejected_insufficient_improvement"
    elif not noise_ok:
        applied, status = False, "rejected_noise_increase"
    elif photometry_gate_active and not photometry_verifiable:
        applied, status = False, "rejected_photometry_unverifiable"
    elif not photometry_ok:
        applied, status = False, "rejected_photometry_change"
    else:
        applied, status = True, "background_subtracted"

    if applied:
        model = candidate_model
        corrected = candidate
        after_scatter = after_scatter_candidate
        reduction = reduction_candidate
        noise_after = noise_after_candidate
        noise_ratio = noise_ratio_candidate
        flux_after = flux_after_candidate
        flux_change = flux_change_candidate
    else:
        model = np.zeros_like(original)
        corrected = original.copy()
        after_scatter = before_scatter
        reduction = 0.0
        noise_after = noise_before
        noise_ratio = 1.0
        flux_after = flux_before
        flux_change = 0.0 if np.isfinite(flux_before) else float("nan")

    metrics: dict[str, float | str | bool] = {
        "applied": applied,
        "status": status,
        "mask_fraction": float(np.mean(combined)),
        "detector_mask_fraction": float(np.mean(detector)),
        "source_mask_fraction": float(np.mean(source)),
        "edge_mask_fraction": float(np.mean(edge)),
        "detection_residual_rstd": detection_scale,
        "background_median": float(np.median(model[np.isfinite(model)])),
        "candidate_background_median": float(np.median(candidate_model[np.isfinite(candidate_model)])),
        "unmasked_median_before": float(np.median(original[~combined])),
        "unmasked_median_after": float(np.median(corrected[~combined])),
        "candidate_unmasked_median_after": float(np.median(candidate[~combined])),
        "large_scale_scatter_before": before_scatter,
        "large_scale_scatter_after": after_scatter,
        "large_scale_reduction": reduction,
        "candidate_large_scale_reduction": reduction_candidate,
        "high_frequency_noise_before": noise_before,
        "high_frequency_noise_after": noise_after,
        "high_frequency_noise_ratio": noise_ratio,
        "candidate_high_frequency_noise_ratio": noise_ratio_candidate,
        "photometry_before": flux_before,
        "photometry_after": flux_after,
        "photometry_change_fraction": flux_change,
        "candidate_photometry_change_fraction": flux_change_candidate,
        "photometry_gate_active": photometry_gate_active,
        "photometry_verifiable": photometry_verifiable,
        "source_edge_bias_after": _source_edge_bias(corrected, source, base),
        "candidate_source_edge_bias_after": _source_edge_bias(candidate, source, base),
    }
    return BackgroundResult(
        original=original,
        background_subtracted=corrected,
        background_model=model,
        rough_background=rough,
        ring_background=ring_background,
        detection_residual=detection_residual,
        detector_mask=detector,
        edge_mask=edge,
        known_source_mask=known,
        tier_masks=tiers,
        source_mask=source,
        combined_mask=combined,
        applied=applied,
        status=status,
        metrics=metrics,
    )


def _output_header(
    header: fits.Header, result: BackgroundResult, product: str, config: BackgroundConfig
) -> fits.Header:
    out = header.copy()
    out["HIERARCH BKG PROD"] = product
    out["HIERARCH BKG APPL"] = bool(result.applied)
    out["HIERARCH BKG BOX"] = int(config.final_box_size)
    out["HIERARCH BKG REDUC"] = round(float(result.metrics["large_scale_reduction"]), 6)
    out["HIERARCH BKG MASKFR"] = round(float(result.metrics["mask_fraction"]), 6)
    out.add_history("2-D background subtraction implemented by astr_ir.background.processor")
    out.add_history("Input is flicker/flicker_corrected product; flicker_model files are excluded.")
    out.add_history("Science equation: background_subtracted = input - background_model.")
    return out


def write_fits_products(
    input_path: str | Path,
    output_dir: str | Path,
    header: fits.Header,
    result: BackgroundResult,
    config: BackgroundConfig,
    overwrite: bool = False,
) -> tuple[Path, Path, float]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_name = raw_filename_from_corrected(input_path.name)
    corrected_path = output_dir / f"background_subtracted_{raw_name}"
    model_path = output_dir / f"background_model_{raw_name}"
    original32 = result.original.astype(np.float32)
    model32 = result.background_model.astype(np.float32)
    corrected32 = original32 - model32
    fits.PrimaryHDU(
        corrected32, header=_output_header(header, result, "subtracted", config)
    ).writeto(corrected_path, overwrite=overwrite, output_verify="silentfix")
    fits.PrimaryHDU(model32, header=_output_header(header, result, "model", config)).writeto(
        model_path, overwrite=overwrite, output_verify="silentfix"
    )
    written_subtracted = np.asarray(fits.getdata(corrected_path), dtype=np.float32)
    written_model = np.asarray(fits.getdata(model_path), dtype=np.float32)
    expected = original32 - written_model
    if written_subtracted.shape != expected.shape or not np.array_equal(
        np.isfinite(written_subtracted), np.isfinite(expected)
    ):
        raise RuntimeError("Written background products have incompatible shape or finite-pixel masks")
    finite = np.isfinite(expected)
    equation_error = (
        float(np.max(np.abs(written_subtracted[finite] - expected[finite])))
        if np.any(finite)
        else 0.0
    )
    return corrected_path, model_path, equation_error


def discover_input_files(input_root: str | Path, sequence: str) -> list[Path]:
    """Select science products explicitly; never select flicker_model FITS files."""

    return sorted((Path(input_root) / sequence).glob("flicker_corrected_*.fits"))


def process_fits_file(
    input_path: str | Path,
    output_dir: str | Path,
    detector_mask: np.ndarray,
    target: Mapping | None,
    config: BackgroundConfig,
    overwrite: bool = False,
) -> tuple[BackgroundResult, dict]:
    image, header = load_fits(input_path)
    result = subtract_background(image, detector_mask=detector_mask, target=target, config=config)
    corrected_path, model_path, equation_error = write_fits_products(
        input_path, output_dir, header, result, config, overwrite=overwrite
    )
    row = {
        "filename": raw_filename_from_corrected(Path(input_path).name),
        "input_filename": Path(input_path).name,
        **result.metrics,
        "equation_max_abs_error_float32": equation_error,
        "subtracted_path": str(corrected_path),
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
    input_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    config: BackgroundConfig | None = None,
    sequences: tuple[str, ...] | None = None,
    overwrite: bool = False,
    limit_per_sequence: int | None = None,
) -> pd.DataFrame:
    input_root, dataset_root, output_root = (
        Path(input_root).resolve(),
        Path(dataset_root).resolve(),
        Path(output_root).resolve(),
    )
    config = config or BackgroundConfig()
    config.validate()
    detector_mask = load_detector_mask(dataset_root / "盲点表")
    table = load_measurement_table(dataset_root / "单帧检测总表_新方法.csv")
    if sequences is None:
        sequences = tuple(
            path.name
            for path in sorted(input_root.iterdir())
            if path.is_dir() and any(path.glob("flicker_corrected_*.fits"))
        )
    if not sequences:
        raise FileNotFoundError(f"No flicker-corrected sequence directories found under {input_root}")
    rows: list[dict] = []
    for sequence in sequences:
        files = discover_input_files(input_root, sequence)
        if limit_per_sequence is not None:
            files = files[:limit_per_sequence]
        for index, path in enumerate(files, start=1):
            target = target_record_for_file(table, path.name)
            _, row = process_fits_file(
                path,
                output_root / sequence,
                detector_mask,
                target,
                config,
                overwrite=overwrite,
            )
            row["sequence"] = sequence
            row["sequence_frame_index"] = index
            row["subtracted_path"] = Path(row["subtracted_path"]).relative_to(output_root).as_posix()
            row["model_path"] = Path(row["model_path"]).relative_to(output_root).as_posix()
            rows.append(row)
    stats = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / "background_statistics.csv"
    if stats_path.exists():
        existing = pd.read_csv(stats_path, encoding="utf-8-sig", dtype={"sequence": str})
        existing = existing.loc[~existing["sequence"].astype(str).isin(map(str, sequences))]
        stats = pd.concat([existing, stats], ignore_index=True)
        stats = stats.sort_values(["sequence", "sequence_frame_index"]).reset_index(drop=True)
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    return stats


def config_as_dict(config: BackgroundConfig) -> dict:
    return asdict(config)
