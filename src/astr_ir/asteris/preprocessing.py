"""Source-protected 3-sigma clipping and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ClippingResult:
    data: np.ndarray
    valid_mask: np.ndarray
    clipping_mask: np.ndarray
    source_mask: np.ndarray
    low: float
    high: float
    clipped_fraction: float
    source_metrics_before: Mapping[str, float]
    source_metrics_after: Mapping[str, float]


def circular_source_mask(
    shape: tuple[int, int], sources: Iterable[tuple[float, float, float]]
) -> np.ndarray:
    """Build a 2-D mask from zero-based ``(x, y, radius)`` source coordinates."""

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask = np.zeros(shape, dtype=bool)
    for x, y, radius in sources:
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(radius) and radius > 0:
            mask |= (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius) ** 2
    return mask


def build_noise_estimation_mask(
    shape: tuple[int, int],
    detector_mask: np.ndarray,
    source_mask: np.ndarray | None = None,
    edge_width: int = 32,
) -> np.ndarray:
    """Return pixels eligible for noise/clipping statistics."""

    detector_mask = np.asarray(detector_mask, dtype=bool)
    if detector_mask.shape != shape:
        raise ValueError(f"Detector mask shape {detector_mask.shape} does not match {shape}")
    valid = ~detector_mask.copy()
    edge = min(max(int(edge_width), 0), min(shape) // 2)
    if edge:
        valid[:edge] = False
        valid[-edge:] = False
        valid[:, :edge] = False
        valid[:, -edge:] = False
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != shape:
            raise ValueError("source_mask shape does not match image")
        valid &= ~source_mask
    return valid


def _iterative_limits(values: np.ndarray, sigma: float, iterations: int = 5) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 16:
        raise ValueError("Too few finite background pixels for sigma clipping")
    keep = np.ones(values.size, dtype=bool)
    center = float(np.median(values))
    scale = float(np.std(values))
    for _ in range(iterations):
        work = values[keep]
        center = float(np.median(work))
        scale = float(np.std(work))
        if not np.isfinite(scale) or scale <= 0:
            break
        updated = np.abs(values - center) <= float(sigma) * scale
        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 16:
            keep = updated
            break
        keep = updated
    return center - float(sigma) * scale, center + float(sigma) * scale


def source_quality_metrics(
    stack: np.ndarray,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    reference_stack: np.ndarray | None = None,
) -> dict[str, float]:
    """Median peak, aperture-like flux, moment FWHM and SNR across a stack.

    When comparing preprocessing before/after, ``reference_stack`` freezes the
    background center and noise estimate to the pre-clipping data. This prevents
    a changed background estimator from being misreported as changed source flux.
    """

    yy, xx = np.indices(source_mask.shape)
    rows = []
    stack_array = np.asarray(stack, np.float32)
    reference_array = stack_array if reference_stack is None else np.asarray(reference_stack, np.float32)
    for image, reference in zip(stack_array, reference_array):
        background = reference[background_mask & np.isfinite(reference)].astype(np.float64)
        source_valid = source_mask & np.isfinite(image) & np.isfinite(reference)
        source_values = image[source_valid].astype(np.float64)
        if background.size < 16 or source_values.size == 0:
            continue
        center = float(np.median(background))
        noise = float(1.4826 * np.median(np.abs(background - center)))
        signal_image = np.zeros_like(image, dtype=np.float64)
        signal_image[source_valid] = np.clip(image[source_valid].astype(np.float64) - center, 0.0, None)
        flux = float(np.sum(image[source_valid] - center))
        positive_flux = float(np.sum(signal_image))
        if positive_flux > 0:
            xc = float(np.sum(signal_image * xx) / positive_flux)
            yc = float(np.sum(signal_image * yy) / positive_flux)
            radial_variance = float(
                np.sum(signal_image * ((xx - xc) ** 2 + (yy - yc) ** 2))
                / (2.0 * positive_flux)
            )
            fwhm = 2.354820045 * np.sqrt(max(radial_variance, 0.0))
        else:
            fwhm = np.nan
        rows.append(
            (
                float(np.max(source_values) - center),
                flux,
                float(fwhm),
                flux / (noise * np.sqrt(source_values.size)) if noise > 0 else np.nan,
            )
        )
    if not rows:
        return {key: np.nan for key in ("peak", "flux", "fwhm", "snr")}
    values = np.asarray(rows, dtype=np.float64)
    result = {}
    for index, key in enumerate(("peak", "flux", "fwhm", "snr")):
        column = values[:, index]
        result[key] = float(np.nanmedian(column)) if np.isfinite(column).any() else np.nan
    return result


def sigma_clip_stack(
    stack: np.ndarray,
    detector_mask: np.ndarray,
    source_mask: np.ndarray | None = None,
    *,
    sigma: float = 3.0,
    edge_width: int = 32,
    temporal: bool = False,
) -> ClippingResult:
    """Clip a ``(T,H,W)`` stack while protecting known sources.

    Global limits are fitted only on background-estimation pixels.  Clipped
    voxels remain available as bounded model inputs but are removed from the
    loss mask.  Optional temporal clipping is disabled by default because short
    8/16-frame ground-based sequences can confuse seeing changes with outliers.
    """

    array = np.asarray(stack, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("stack must have shape (time, height, width)")
    source = np.zeros(array.shape[1:], dtype=bool) if source_mask is None else np.asarray(source_mask, bool)
    estimation = build_noise_estimation_mask(array.shape[1:], detector_mask, source, edge_width)
    finite = np.isfinite(array)
    values = array[:, estimation & np.all(finite, axis=0)]
    low, high = _iterative_limits(values.ravel(), sigma)
    clipping = finite & estimation[None] & ((array < low) | (array > high))
    if temporal:
        work = np.where(finite, array, np.nan)
        center = np.nanmedian(work, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(work - center[None]), axis=0)
        temporal_clip = (
            finite
            & estimation[None]
            & np.isfinite(mad)[None]
            & (mad > 0)[None]
            & (np.abs(array - center[None]) > float(sigma) * mad[None])
        )
        clipping |= temporal_clip
    clipped = array.copy()
    background_voxels = estimation[None] & finite
    clipped[background_voxels] = np.clip(clipped[background_voxels], low, high)
    base_valid = finite & ~np.asarray(detector_mask, bool)[None]
    valid = base_valid & ~clipping
    clipped[~base_valid] = 0.0
    denominator = max(int(np.count_nonzero(background_voxels)), 1)
    science_source = source & ~np.asarray(detector_mask, bool)
    metrics_before = source_quality_metrics(array, science_source, estimation)
    metrics_after = source_quality_metrics(
        clipped, science_source, estimation, reference_stack=array
    )
    return ClippingResult(
        data=clipped,
        valid_mask=valid,
        clipping_mask=clipping,
        source_mask=source,
        low=float(low),
        high=float(high),
        clipped_fraction=float(np.count_nonzero(clipping) / denominator),
        source_metrics_before=metrics_before,
        source_metrics_after=metrics_after,
    )


def fit_normalization(values: np.ndarray) -> dict[str, float]:
    """Fit mean/std from already-selected training background samples."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = _iterative_limits(values, 3.0)
    kept = values[(values >= low) & (values <= high)]
    mean, std = float(np.mean(kept)), float(np.std(kept))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training normalization scale is not positive")
    return {"mean": mean, "std": std, "fit_low": float(low), "fit_high": float(high)}


def normalize_stack(stack: np.ndarray, valid_mask: np.ndarray, mean: float, std: float) -> np.ndarray:
    normalized = (np.asarray(stack, np.float32) - float(mean)) / float(std)
    normalized[~np.asarray(valid_mask, bool) | ~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def fill_invalid_with_temporal_mean(
    stack: np.ndarray,
    valid_mask: np.ndarray,
    *,
    neutral: float = 0.0,
) -> np.ndarray:
    """Make a finite model input while keeping validity as separate truth.

    Invalid detector samples are replaced only in the transient neural-network
    input.  The replacement is the mean of valid temporal samples at the same
    registered sky pixel; pixels with no temporal coverage receive ``neutral``.
    This prevents a fixed zero-valued blind-map pattern from becoming a learned
    image feature.  Callers must continue to use ``valid_mask`` for losses,
    coadds, photometry, and output DQ.
    """

    values = np.asarray(stack, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("stack and valid_mask must have the same (time, height, width) shape")
    count = valid.sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
    temporal = np.divide(
        total,
        count,
        out=np.full(values.shape[1:], float(neutral), dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)
    filled = np.where(valid, values, temporal[None]).astype(np.float32)
    filled[~np.isfinite(filled)] = float(neutral)
    return filled
