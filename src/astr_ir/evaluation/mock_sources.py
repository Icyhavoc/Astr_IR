"""Reusable PSF injection, blind detection, catalog matching, and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    maximum_filter,
    shift as subpixel_shift,
)
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen scientific choices for a blind injection-recovery experiment."""

    seed: int = 20260821
    psf_size: int = 31
    psf_min_training_snr: float = 10.0
    edge_width: int = 32
    source_exclusion_radius: float = 18.0
    blank_detection_threshold: float = 3.5
    minimum_injection_separation: float = 36.0
    match_radius: float = 2.5
    peak_separation: int = 7
    validation_snrs: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
    test_snrs: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
    validation_sources_per_frame: int = 6
    test_sources_per_frame: int = 8
    test_repeats_per_snr: int = 1
    threshold_grid: tuple[float, ...] = tuple(np.arange(3.5, 7.01, 0.25))
    target_validation_purity: float = 0.90
    bootstrap_iterations: int = 1000

    def validate(self) -> None:
        if self.psf_size < 9 or self.psf_size % 2 != 1:
            raise ValueError("psf_size must be an odd integer >= 9")
        if self.edge_width < self.psf_size // 2:
            raise ValueError("edge_width must be at least half the PSF size")
        if self.match_radius <= 0 or self.peak_separation < 1:
            raise ValueError("matching and peak-separation parameters must be positive")
        if not self.validation_snrs or not self.test_snrs:
            raise ValueError("validation_snrs and test_snrs cannot be empty")
        if min(*self.validation_snrs, *self.test_snrs) <= 0:
            raise ValueError("injected SNR values must be positive")
        if min(self.validation_sources_per_frame, self.test_sources_per_frame) < 1:
            raise ValueError("sources per frame must be positive")
        if self.test_repeats_per_snr < 1 or self.bootstrap_iterations < 100:
            raise ValueError("test repeats must be positive and bootstrap_iterations >= 100")
        if not self.threshold_grid or min(self.threshold_grid) <= 0:
            raise ValueError("threshold_grid must contain positive values")
        if not 0 < self.target_validation_purity <= 1:
            raise ValueError("target_validation_purity must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def robust_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def _extract_centered_cutout(
    image: np.ndarray,
    valid: np.ndarray,
    x: float,
    y: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    half = size // 2
    ix, iy = int(round(x)), int(round(y))
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return None
    cutout = np.asarray(image[y0:y1, x0:x1], dtype=np.float64).copy()
    cut_valid = np.asarray(valid[y0:y1, x0:x1], dtype=bool).copy()
    if cutout.shape != (size, size) or np.mean(cut_valid) < 0.98:
        return None
    dy, dx = iy - y, ix - x
    finite_fill = float(np.median(cutout[cut_valid & np.isfinite(cutout)]))
    shifted = subpixel_shift(
        np.where(cut_valid & np.isfinite(cutout), cutout, finite_fill),
        (dy, dx),
        order=3,
        mode="constant",
        cval=finite_fill,
        prefilter=True,
    )
    shifted_valid = subpixel_shift(
        cut_valid.astype(np.float32),
        (dy, dx),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5
    return shifted, shifted_valid


def build_empirical_psf(
    samples: Iterable[tuple[np.ndarray, np.ndarray, float, float]],
    size: int = 31,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a non-negative, unit-flux empirical PSF from isolated training sources."""

    if size < 9 or size % 2 != 1:
        raise ValueError("size must be an odd integer >= 9")
    yy, xx = np.mgrid[:size, :size]
    center = size // 2
    radius = np.hypot(xx - center, yy - center)
    aperture = radius <= 0.32 * size
    annulus = (radius >= 0.38 * size) & (radius <= 0.48 * size)
    normalized: list[np.ndarray] = []
    rows: list[dict] = []
    for index, (image, valid, x, y) in enumerate(samples):
        extracted = _extract_centered_cutout(image, valid, x, y, size)
        if extracted is None:
            rows.append({"sample_index": index, "accepted": False, "reason": "cutout"})
            continue
        cutout, cut_valid = extracted
        annulus_values = cutout[annulus & cut_valid & np.isfinite(cutout)]
        if annulus_values.size < 24:
            rows.append({"sample_index": index, "accepted": False, "reason": "annulus"})
            continue
        background = float(np.median(annulus_values))
        source = cutout - background
        flux = float(np.sum(source[aperture & cut_valid]))
        peak = float(source[center, center])
        if not np.isfinite(flux) or flux <= 0 or not np.isfinite(peak) or peak <= 0:
            rows.append({"sample_index": index, "accepted": False, "reason": "flux"})
            continue
        source[~cut_valid | ~np.isfinite(source)] = 0.0
        normalized.append(source / flux)
        rows.append(
            {
                "sample_index": index,
                "accepted": True,
                "reason": "accepted",
                "background": background,
                "aperture_flux": flux,
                "normalized_peak": peak / flux,
            }
        )
    if len(normalized) < 4:
        raise RuntimeError(f"At least four valid source cutouts are required; got {len(normalized)}")
    stack = np.stack(normalized)
    psf = np.median(stack, axis=0)
    psf = np.maximum(psf, 0.0)
    psf[~aperture] = 0.0
    total = float(np.sum(psf))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Empirical PSF has invalid total flux")
    psf = (psf / total).astype(np.float32)
    return psf, pd.DataFrame(rows)


def matched_filter_map(
    image: np.ndarray,
    valid: np.ndarray,
    psf: np.ndarray,
    noise_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return matched-filter flux, significance, noise scale, and unit-flux response."""

    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    if not np.any(valid):
        raise ValueError("No valid image pixels")
    center = float(np.median(image[valid]))
    filled = np.where(valid, image - center, 0.0)
    kernel = np.asarray(psf, dtype=np.float64)
    response = float(np.sum(kernel**2))
    if not np.isfinite(response) or response <= 0:
        raise ValueError("PSF matched-filter response must be positive")
    correlation = fftconvolve(filled, kernel[::-1, ::-1], mode="same")
    flux = correlation / response
    effective_noise = valid if noise_mask is None else valid & np.asarray(noise_mask, dtype=bool)
    scale = robust_std(correlation[effective_noise])
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Could not estimate matched-filter noise")
    significance = correlation / scale
    significance[~valid] = np.nan
    flux[~valid] = np.nan
    return flux.astype(np.float32), significance.astype(np.float32), scale, response


def detect_sources(
    image: np.ndarray,
    valid: np.ndarray,
    psf: np.ndarray,
    threshold: float,
    peak_separation: int = 7,
    noise_mask: np.ndarray | None = None,
    detection_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Blind matched-filter peak detection over the allowed image area."""

    flux, significance, scale, _ = matched_filter_map(image, valid, psf, noise_mask=noise_mask)
    allowed = np.asarray(valid, dtype=bool) & np.isfinite(significance)
    if detection_mask is not None:
        allowed &= np.asarray(detection_mask, dtype=bool)
    local_max = maximum_filter(
        np.nan_to_num(significance, nan=-np.inf),
        size=max(3, int(peak_separation)),
        mode="constant",
        cval=-np.inf,
    )
    candidates = allowed & (significance >= float(threshold)) & (significance == local_max)
    y, x = np.nonzero(candidates)
    if x.size == 0:
        catalog = pd.DataFrame(columns=["detection_id", "x", "y", "score", "flux"])
    else:
        order = np.argsort(significance[y, x])[::-1]
        x, y = x[order], y[order]
        catalog = pd.DataFrame(
            {
                "detection_id": np.arange(len(x), dtype=int),
                "x": x.astype(float),
                "y": y.astype(float),
                "score": significance[y, x].astype(float),
                "flux": flux[y, x].astype(float),
            }
        )
    return catalog, flux, significance, scale


def make_evaluation_mask(
    image: np.ndarray,
    valid: np.ndarray,
    psf: np.ndarray,
    edge_width: int,
    known_sources: Sequence[tuple[float, float]] = (),
    source_exclusion_radius: float = 18.0,
    blank_detection_threshold: float = 3.5,
    peak_separation: int = 7,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create an input-derived source-free mask used identically before and after inference."""

    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    half = psf.shape[0] // 2
    mask = binary_erosion(valid, structure=np.ones((2 * half + 1, 2 * half + 1), bool))
    edge = max(int(edge_width), half)
    mask[:edge] = False
    mask[-edge:] = False
    mask[:, :edge] = False
    mask[:, -edge:] = False
    preliminary, _, _, _ = detect_sources(
        image,
        valid,
        psf,
        threshold=blank_detection_threshold,
        peak_separation=peak_separation,
        noise_mask=mask,
        detection_mask=mask,
    )
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    all_sources = list(known_sources) + list(preliminary[["x", "y"]].itertuples(index=False, name=None))
    for x, y in all_sources:
        mask &= (xx - float(x)) ** 2 + (yy - float(y)) ** 2 > source_exclusion_radius**2
    return mask, preliminary


def sample_injection_positions(
    rng: np.random.Generator,
    allowed: np.ndarray,
    count: int,
    minimum_separation: float,
) -> list[tuple[float, float]]:
    candidates_y, candidates_x = np.nonzero(allowed)
    if candidates_x.size == 0:
        raise RuntimeError("No source-free pixels are available for injection")
    positions: list[tuple[float, float]] = []
    for index in rng.permutation(candidates_x.size):
        x = float(candidates_x[index]) + float(rng.uniform(-0.45, 0.45))
        y = float(candidates_y[index]) + float(rng.uniform(-0.45, 0.45))
        if all(np.hypot(x - px, y - py) >= minimum_separation for px, py in positions):
            positions.append((x, y))
            if len(positions) == count:
                return positions
    raise RuntimeError(f"Could only place {len(positions)}/{count} separated mock sources")


def _paste_shifted_psf(image: np.ndarray, psf: np.ndarray, x: float, y: float, flux: float) -> None:
    size = psf.shape[0]
    half = size // 2
    ix, iy = int(round(x)), int(round(y))
    shifted = subpixel_shift(
        psf,
        (y - iy, x - ix),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    shifted = np.maximum(shifted, 0.0)
    shifted /= max(float(np.sum(shifted)), np.finfo(float).eps)
    image[iy - half : iy + half + 1, ix - half : ix + half + 1] += flux * shifted


def inject_sources(
    image: np.ndarray,
    valid: np.ndarray,
    psf: np.ndarray,
    positions: Sequence[tuple[float, float]],
    target_snrs: Sequence[float],
    noise_mask: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Inject unit-normalized empirical PSFs at requested matched-filter SNRs."""

    if len(positions) != len(target_snrs):
        raise ValueError("positions and target_snrs must have the same length")
    _, _, filtered_noise, response = matched_filter_map(
        image, valid, psf, noise_mask=noise_mask
    )
    injected = np.asarray(image, dtype=np.float32).copy()
    rows = []
    for injection_id, ((x, y), target_snr) in enumerate(zip(positions, target_snrs)):
        true_flux = float(target_snr) * filtered_noise / response
        _paste_shifted_psf(injected, psf, x, y, true_flux)
        rows.append(
            {
                "injection_id": int(injection_id),
                "x_true": float(x),
                "y_true": float(y),
                "target_snr": float(target_snr),
                "true_flux": true_flux,
            }
        )
    return injected, pd.DataFrame(rows)


def match_catalogs(
    truth: pd.DataFrame,
    detections: pd.DataFrame,
    radius: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform deterministic one-to-one truth/detection matching."""

    truth = truth.reset_index(drop=True).copy()
    detections = detections.reset_index(drop=True).copy()
    if truth.empty or detections.empty:
        matches = pd.DataFrame(
            columns=["injection_id", "detection_id", "distance", "score", "measured_flux"]
        )
        return matches, truth, detections
    truth_xy = truth[["x_true", "y_true"]].to_numpy(float)
    detection_xy = detections[["x", "y"]].to_numpy(float)
    distances = np.linalg.norm(truth_xy[:, None, :] - detection_xy[None, :, :], axis=2)
    penalty = max(1e6, float(np.nanmax(distances)) + 1e3)
    costs = np.where(distances <= radius, distances, penalty)
    truth_index, detection_index = linear_sum_assignment(costs)
    accepted = costs[truth_index, detection_index] < penalty
    truth_index, detection_index = truth_index[accepted], detection_index[accepted]
    matches = pd.DataFrame(
        {
            "injection_id": truth.iloc[truth_index]["injection_id"].to_numpy(int),
            "detection_id": detections.iloc[detection_index]["detection_id"].to_numpy(int),
            "distance": distances[truth_index, detection_index],
            "score": detections.iloc[detection_index]["score"].to_numpy(float),
            "measured_flux": detections.iloc[detection_index]["flux"].to_numpy(float),
        }
    )
    unmatched_truth = truth.drop(index=truth_index).reset_index(drop=True)
    unmatched_detections = detections.drop(index=detection_index).reset_index(drop=True)
    return matches, unmatched_truth, unmatched_detections


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))

