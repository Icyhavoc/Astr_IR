from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astr_ir.evaluation.mock_sources import (
    EvaluationConfig,
    build_empirical_psf,
    detect_sources,
    inject_sources,
    match_catalogs,
    matched_filter_map,
    sample_injection_positions,
    wilson_interval,
)
from astr_ir.evaluation.pipeline import _interpolated_limit


def gaussian_psf(size: int = 21, sigma: float = 2.0) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    center = size // 2
    psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    return (psf / psf.sum()).astype(np.float32)


def test_config_rejects_even_psf_size():
    with pytest.raises(ValueError, match="odd"):
        EvaluationConfig(psf_size=20).validate()


def test_empirical_psf_is_nonnegative_and_unit_flux():
    psf = gaussian_psf(31, 2.2)
    samples = []
    for seed in range(6):
        image = np.random.default_rng(seed).normal(10.0, 0.01, (65, 65))
        image[17:48, 17:48] += 1000 * psf
        samples.append((image, np.ones_like(image, bool), 32.0, 32.0))
    empirical, diagnostics = build_empirical_psf(samples, size=31)
    assert diagnostics["accepted"].sum() == 6
    assert np.all(empirical >= 0)
    assert np.isclose(empirical.sum(), 1.0, atol=1e-6)
    assert np.unravel_index(np.argmax(empirical), empirical.shape) == (15, 15)


def test_target_snr_matches_injected_filter_score():
    rng = np.random.default_rng(12)
    image = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
    valid = np.ones_like(image, bool)
    psf = gaussian_psf()
    allowed = np.ones_like(valid)
    allowed[:20] = allowed[-20:] = False
    allowed[:, :20] = allowed[:, -20:] = False
    injected, truth = inject_sources(
        image, valid, psf, [(64.0, 64.0)], [7.0], noise_mask=allowed
    )
    _, scores, _, _ = matched_filter_map(injected, valid, psf, noise_mask=allowed)
    local_peak = np.nanmax(scores[61:68, 61:68])
    assert truth.loc[0, "target_snr"] == 7.0
    assert 5.0 < local_peak < 9.0


def test_blind_detector_finds_injected_source_without_truth_position():
    rng = np.random.default_rng(13)
    image = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
    valid = np.ones_like(image, bool)
    psf = gaussian_psf()
    mask = np.ones_like(valid)
    mask[:16] = mask[-16:] = False
    mask[:, :16] = mask[:, -16:] = False
    injected, _ = inject_sources(image, valid, psf, [(73.2, 51.7)], [12.0], mask)
    catalog, _, _, _ = detect_sources(
        injected, valid, psf, threshold=5.0, noise_mask=mask, detection_mask=mask
    )
    assert np.min(np.hypot(catalog["x"] - 73.2, catalog["y"] - 51.7)) < 2.0


def test_catalog_matching_is_one_to_one():
    truth = pd.DataFrame(
        {"injection_id": [0, 1], "x_true": [10.0, 12.0], "y_true": [10.0, 10.0]}
    )
    detections = pd.DataFrame(
        {
            "detection_id": [0],
            "x": [11.0],
            "y": [10.0],
            "score": [8.0],
            "flux": [100.0],
        }
    )
    matches, missed, false = match_catalogs(truth, detections, radius=2.0)
    assert len(matches) == 1
    assert len(missed) == 1
    assert len(false) == 0


def test_position_sampler_respects_separation():
    allowed = np.ones((128, 128), bool)
    positions = sample_injection_positions(
        np.random.default_rng(14), allowed, count=8, minimum_separation=20
    )
    distances = [
        np.hypot(x1 - x2, y1 - y2)
        for index, (x1, y1) in enumerate(positions)
        for x2, y2 in positions[index + 1 :]
    ]
    assert min(distances) >= 20


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(60, 100)
    assert low < 0.6 < high
    assert 0 <= low <= high <= 1


def test_interpolated_completeness_limit():
    metrics = pd.DataFrame(
        {"target_snr": [2.0, 4.0, 6.0], "completeness": [0.1, 0.5, 0.9]}
    )
    assert _interpolated_limit(metrics, "completeness", 0.5) == 4.0
    assert np.isnan(_interpolated_limit(metrics, "completeness", 0.95))
