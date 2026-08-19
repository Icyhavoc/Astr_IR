from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from astr_ir.background.processor import (
    BackgroundConfig,
    aperture_photometry,
    discover_input_files,
    load_detector_mask,
    make_known_source_mask,
    subtract_background,
    write_fits_products,
)


def synthetic_image(seed=8, with_star=True):
    rng = np.random.default_rng(seed)
    h = w = 192
    yy, xx = np.mgrid[:h, :w]
    background = 30000 + 0.7 * xx + 0.35 * yy + 45 * np.sin(2 * np.pi * xx / 180)
    image = background + rng.normal(0, 5, (h, w))
    target = None
    if with_star:
        x0, y0, sigma = 96.0, 91.0, 2.3
        image += 2200 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        target = {
            "xc": x0 + 1,
            "yc": y0 + 1,
            "r_ap": 7.0,
            "r_in": 10.0,
            "r_out": 15.0,
            "fwhm": 2.355 * sigma,
            "snr": 100.0,
        }
    return image, background, target


def config(**changes):
    values = dict(
        edge_width=8,
        rough_box_size=32,
        rough_filter_size=3,
        ring_inner_radius=20,
        ring_width=3,
        tier_gaussian_sigmas=(8.0, 4.0, 2.0),
        tier_threshold_sigmas=(3.0, 3.5, 5.0),
        tier_min_pixels=(8, 3, 1),
        tier_dilation_radii=(10, 7, 4),
        final_box_size=16,
        final_filter_size=3,
        validation_block_size=24,
        min_large_scale_reduction=0.05,
    )
    values.update(changes)
    return BackgroundConfig(**values)


def test_background_gradient_is_removed():
    image, _, target = synthetic_image()
    result = subtract_background(image, target=target, config=config())
    assert result.applied
    assert result.metrics["large_scale_reduction"] > 0.50
    assert abs(result.metrics["unmasked_median_after"]) < 20


def test_known_star_is_in_source_mask():
    image, _, target = synthetic_image()
    result = subtract_background(image, target=target, config=config())
    known = make_known_source_mask(image.shape, target, config().known_source_radius_scale)
    assert np.all(result.source_mask[known])


def test_aperture_photometry_is_preserved():
    image, _, target = synthetic_image()
    result = subtract_background(image, target=target, config=config())
    before = aperture_photometry(image, target)
    after = aperture_photometry(result.background_subtracted, target)
    assert abs((after - before) / before) < 0.01


def test_float64_equation_is_exact():
    image, _, target = synthetic_image()
    result = subtract_background(image, target=target, config=config())
    assert np.array_equal(result.background_subtracted, result.original - result.background_model)


def test_dead_and_noise_maps_are_unioned(tmp_path: Path):
    dead = np.zeros((16, 16), dtype=np.uint8)
    noise = np.zeros_like(dead)
    dead[2, 3] = 1
    noise[7, 8] = 1
    Image.fromarray(dead).save(tmp_path / "DeadBlindMap.tiff")
    Image.fromarray(noise).save(tmp_path / "NoiseBlindMap.tiff")
    combined = load_detector_mask(tmp_path)
    assert combined.sum() == 2
    assert combined[2, 3] and combined[7, 8]


def test_input_discovery_excludes_flicker_models(tmp_path: Path):
    sequence = tmp_path / "90000002"
    sequence.mkdir()
    (sequence / "flicker_corrected_a.fits").touch()
    (sequence / "flicker_model_a.fits").touch()
    (sequence / "other.fits").touch()
    assert [p.name for p in discover_input_files(tmp_path, "90000002")] == ["flicker_corrected_a.fits"]


def test_quality_gate_returns_zero_model_when_improvement_requirement_is_impossible():
    image, _, target = synthetic_image()
    result = subtract_background(image, target=target, config=config(min_large_scale_reduction=0.999999))
    assert not result.applied
    assert np.count_nonzero(result.background_model) == 0
    assert np.array_equal(result.background_subtracted, result.original)


def test_fits_outputs_preserve_header_and_float32_equation(tmp_path: Path):
    image, _, target = synthetic_image()
    input_path = tmp_path / "flicker_corrected_input.fits"
    header = fits.Header()
    header["EXPOSURE"] = 1.0
    header["TELESCOP"] = "RKZ50"
    header["HIERARCH FLK APPL"] = True
    fits.PrimaryHDU(image.astype(np.float32), header=header).writeto(input_path)
    result = subtract_background(image, target=target, config=config())
    subtracted_path, model_path, error = write_fits_products(
        input_path, tmp_path / "out", header, result, config()
    )
    subtracted = fits.getdata(subtracted_path)
    model = fits.getdata(model_path)
    output_header = fits.getheader(subtracted_path)
    assert output_header["EXPOSURE"] == 1.0
    assert output_header["TELESCOP"] == "RKZ50"
    assert output_header["HIERARCH FLK APPL"]
    assert subtracted.dtype.kind == "f" and subtracted.dtype.itemsize == 4
    assert np.array_equal(subtracted, image.astype(np.float32) - model)
    assert error == 0.0
