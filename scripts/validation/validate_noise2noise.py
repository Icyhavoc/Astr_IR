"""Strictly validate Noise2Noise manifests, FITS equations, and held-out science gates."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def read_float32_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        hdul.verify("exception")
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    if data.ndim != 2 or data.dtype.kind != "f" or data.dtype.itemsize != 4:
        raise RuntimeError(f"Expected a 2-D float32 FITS product: {path} ({data.dtype}, {data.shape})")
    return data.astype(np.float32, copy=False), header


def equation_error(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(f"Shape mismatch for {label}")
    if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        raise RuntimeError(f"Finite-pixel mask mismatch for {label}")
    finite = np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if np.any(finite) else 0.0


def main() -> None:
    background_root = PROJECT_ROOT / "data" / "processed" / "background"
    output_root = PROJECT_ROOT / "data" / "processed" / "noise2noise"
    split = pd.read_csv(
        output_root / "manifests" / "split_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    pairs = pd.read_csv(
        output_root / "manifests" / "pair_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    statistics = pd.read_csv(
        output_root / "noise2noise_statistics.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    expected_split = {
        (sequence, split_name): count
        for sequence in ("90000002", "90000003")
        for split_name, count in (("train", 48), ("guard", 4), ("validation", 12), ("test", 16))
    }
    actual_split = split.groupby(["sequence", "split"]).size().to_dict()
    if actual_split != expected_split:
        raise RuntimeError(f"Unexpected frame split: {actual_split}")
    frame_splits = split.set_index("frame_id")["split"].to_dict()
    if (pairs["lag"] < 2).any():
        raise RuntimeError("Pair manifest contains lag < 2")
    for pair in pairs.itertuples(index=False):
        if frame_splits[pair.frame_a] != pair.split or frame_splits[pair.frame_b] != pair.split:
            raise RuntimeError(f"Cross-split pair detected: {pair.pair_id}")
    expected_denoised = set(statistics["denoised_path"].astype(str))
    expected_residual = set(statistics["residual_path"].astype(str))
    actual_denoised = {
        path.relative_to(output_root).as_posix()
        for path in (output_root / "denoised").rglob("*.fits")
    }
    actual_residual = {
        path.relative_to(output_root).as_posix()
        for path in (output_root / "residual").rglob("*.fits")
    }
    if expected_denoised != actual_denoised or expected_residual != actual_residual:
        raise RuntimeError("Noise2Noise output inventory differs from the statistics table")
    if len(statistics) != 160 or len(actual_denoised) != 160 or len(actual_residual) != 160:
        raise RuntimeError("Expected 160 frames and 320 Noise2Noise FITS products")
    maximum_error = 0.0
    for row in statistics.itertuples(index=False):
        input_image, _ = read_float32_fits(
            background_root / str(row.sequence) / f"background_subtracted_{row.filename}"
        )
        denoised, header = read_float32_fits(output_root / row.denoised_path)
        residual, _ = read_float32_fits(output_root / row.residual_path)
        if header.get("HIERARCH N2N PROD") != "denoised":
            raise RuntimeError(f"Missing N2N product metadata: {row.denoised_path}")
        maximum_error = max(
            maximum_error,
            equation_error(denoised, input_image - residual, row.denoised_path),
        )
    test = statistics.loc[statistics["split"] == "test"]
    weak = test.loc[test["sequence"] == "90000002"]
    high = test.loc[test["sequence"] == "90000003"]
    test_noise_ratio = float(test["noise_ratio"].median())
    weak_snr_ratio = float(weak["aperture_snr_ratio"].median())
    high_flux_change = float(high["photometry_change_fraction"].abs().max())
    print("Noise2Noise strict validation")
    print(f"  frames/products: {len(statistics)}/{len(actual_denoised) + len(actual_residual)}")
    print(f"  float32 equation max error: {maximum_error:g}")
    print(f"  test median noise ratio: {test_noise_ratio:.6f}")
    print(f"  weak-source median aperture-SNR ratio: {weak_snr_ratio:.6f}")
    print(f"  high-SNR max |flux change|: {100 * high_flux_change:.4f}%")
    failures = []
    if maximum_error != 0:
        failures.append("float32 equation")
    if not test_noise_ratio < 1:
        failures.append("test noise reduction")
    if not weak_snr_ratio > 1:
        failures.append("weak-source SNR")
    if not high_flux_change <= 0.01:
        failures.append("high-SNR photometry")
    if failures:
        raise RuntimeError(f"Noise2Noise science validation failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
