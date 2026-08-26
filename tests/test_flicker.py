from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from astr_ir.flicker.processor import (
    FlickerConfig,
    aperture_photometry,
    correct_flicker,
    load_detector_mask,
    load_fits,
    make_known_source_mask,
    profile_degradation_metrics,
    write_fits_products,
)


def synthetic_image(direction="row", strength=20.0, seed=7, with_star=False):
    rng = np.random.default_rng(seed)
    h = w = 128
    yy, xx = np.mgrid[:h, :w]
    image = 30000.0 + 0.08 * xx + 0.05 * yy + rng.normal(0, 5, (h, w))
    phase = np.arange(h if direction == "row" else w)
    stripe = strength * (np.sin(2 * np.pi * phase / 19) + 0.35 * np.sin(2 * np.pi * phase / 7))
    image += stripe[:, None] if direction == "row" else stripe[None, :]
    target = None
    if with_star:
        x0, y0, sigma = 63.0, 61.0, 2.2
        image += 1800 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        target = {
            "xc": x0 + 1,
            "yc": y0 + 1,
            "r_ap": 7.0,
            "r_in": 9.0,
            "r_out": 13.0,
            "fwhm": 2.355 * sigma,
            "snr": 100.0,
        }
    return image, target


def config(**changes):
    values = dict(
        edge_width=8,
        background_block_size=16,
        background_smooth_sigma_blocks=1.0,
        profile_smooth_size=5,
        source_dilation=4,
        min_direction_score=1.4,
        min_relative_improvement=0.30,
    )
    values.update(changes)
    return FlickerConfig(**values)


def test_auto_detects_and_corrects_row_stripes():
    image, _ = synthetic_image("row")
    result = correct_flicker(image, config=config())
    assert result.applied
    assert result.selected_direction == "row"
    assert result.metrics["relative_reduction"] >= 0.30


def test_auto_detects_and_corrects_column_stripes():
    image, _ = synthetic_image("column")
    result = correct_flicker(image, config=config())
    assert result.applied
    assert result.selected_direction == "column"
    assert result.metrics["relative_reduction"] >= 0.30


def test_weak_stripe_returns_not_needed_and_zero_model():
    image, _ = synthetic_image("row", strength=0.0)
    result = correct_flicker(image, config=config(min_direction_score=5.0))
    assert not result.applied
    assert result.status == "not_needed_weak_stripe"
    assert np.count_nonzero(result.flicker_model) == 0
    assert np.array_equal(result.corrected, result.original)


def test_corrected_equals_original_minus_model():
    image, _ = synthetic_image("row")
    result = correct_flicker(image, config=config())
    assert np.array_equal(result.corrected, result.original - result.flicker_model)


def test_known_star_is_masked_and_photometry_is_preserved():
    image, target = synthetic_image("row", with_star=True)
    result = correct_flicker(image, target=target, config=config())
    known_mask = make_known_source_mask(image.shape, target)
    assert np.all(result.source_mask[known_mask])
    before = aperture_photometry(result.original, target)
    after = aperture_photometry(result.corrected, target)
    assert abs((after - before) / before) < 0.01


def test_manual_direction_is_respected():
    image, _ = synthetic_image("row")
    result = correct_flicker(image, config=config(direction="column", min_direction_score=0.0, min_relative_improvement=0.0))
    assert result.selected_direction == "column"


def test_local_quality_gate_falls_back_to_unsmoothed_profile():
    image, _ = synthetic_image("row")
    rng = np.random.default_rng(99)
    image = image + rng.normal(0.0, 25.0, image.shape[0])[:, None]
    result = correct_flicker(
        image,
        config=config(
            fallback_profile_smooth_sizes=(1,),
            max_local_worse_fraction=0.0,
            max_local_worse_over_threshold_fraction=0.0,
            max_local_increase_dn=0.0,
        ),
    )
    assert result.applied
    assert result.profile_smooth_size == 1
    assert result.metrics["profile_fallback_used"]
    assert result.metrics["local_worse_lines"] == 0
    assert result.metrics["local_gate_passed"]


def test_high_snr_unverifiable_photometry_is_rejected():
    image, _ = synthetic_image("row")
    result = correct_flicker(image, target={"snr": 100.0}, config=config())
    assert not result.applied
    assert result.status == "rejected_photometry_unverifiable"
    assert np.count_nonzero(result.flicker_model) == 0


def test_profile_degradation_metrics_count_local_regressions():
    metrics = profile_degradation_metrics(
        np.array([-3.0, -1.0, 1.0, 3.0]),
        np.array([-5.0, -1.0, 1.0, 5.0]),
        threshold_dn=1.0,
    )
    assert metrics["local_worse_lines"] == 2
    assert metrics["local_worse_over_threshold_lines"] == 2
    assert metrics["local_max_increase_dn"] == 2.0


def test_dead_and_noise_maps_are_unioned(tmp_path: Path):
    dead = np.zeros((16, 16), dtype=np.uint8)
    noise = np.zeros_like(dead)
    dead[2, 3] = 1
    noise[7, 8] = 1
    noise[2, 3] = 1
    Image.fromarray(dead).save(tmp_path / "DeadBlindMap.tiff")
    Image.fromarray(noise).save(tmp_path / "NoiseBlindMap.tiff")
    combined = load_detector_mask(tmp_path)
    assert combined.sum() == 2
    assert combined[2, 3] and combined[7, 8]


def test_fits_outputs_preserve_science_header_and_float32_equation(tmp_path: Path):
    image, _ = synthetic_image("row")
    input_path = tmp_path / "input.fits"
    header = fits.Header()
    header["EXPOSURE"] = 1.0
    header["TELESCOP"] = "RKZ50"
    header["RA"] = 63.648
    fits.PrimaryHDU(image.astype(np.uint16), header=header).writeto(input_path)
    loaded, loaded_header = load_fits(input_path)
    result = correct_flicker(loaded, config=config())
    corrected_path, model_path, equation_error = write_fits_products(
        input_path, tmp_path / "out", loaded_header, result
    )
    corrected = fits.getdata(corrected_path)
    model = fits.getdata(model_path)
    out_header = fits.getheader(corrected_path)
    with fits.open(corrected_path) as hdul:
        assert hdul["DQ"].data.shape == corrected.shape
    assert corrected.dtype.kind == "f" and corrected.dtype.itemsize == 4
    assert out_header["EXPOSURE"] == 1.0
    assert out_header["TELESCOP"] == "RKZ50"
    assert out_header["RA"] == 63.648
    assert np.array_equal(corrected, loaded.astype(np.float32) - model)
    assert equation_error == 0.0
