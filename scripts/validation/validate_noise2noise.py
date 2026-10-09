"""Validate Noise2Noise split isolation, FITS equations, and DQ propagation."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from astr_ir.dq import read_dq


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "noise2noise")
    args = parser.parse_args()
    background_root, output_root = args.input_root, args.output_root
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
    if split.frame_id.duplicated().any() or set(statistics.frame_id) != set(split.frame_id):
        raise RuntimeError("Split/statistics frame IDs differ or repeat")
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
    if len(statistics) != len(split) or len(actual_denoised) != len(split) or len(actual_residual) != len(split):
        raise RuntimeError("Expected one denoised and one residual product per split frame")
    maximum_error = 0.0
    for row in statistics.itertuples(index=False):
        input_image, _ = read_float32_fits(
            background_root / str(row.sequence) / f"background_subtracted_{row.filename}"
        )
        denoised, header = read_float32_fits(output_root / row.denoised_path)
        residual, _ = read_float32_fits(output_root / row.residual_path)
        if header.get("HIERARCH N2N PROD") != "denoised":
            raise RuntimeError(f"Missing N2N product metadata: {row.denoised_path}")
        input_dq = read_dq(background_root / str(row.sequence) / f"background_subtracted_{row.filename}", input_image.shape)
        output_dq = read_dq(output_root / row.denoised_path, denoised.shape)
        if output_dq is None or (input_dq is not None and np.any((output_dq & input_dq) != input_dq)):
            raise RuntimeError(f"Upstream DQ was not propagated: {row.denoised_path}")
        maximum_error = max(
            maximum_error,
            equation_error(denoised, input_image - residual, row.denoised_path),
        )
    print("Noise2Noise structural validation")
    print(f"  frames/products: {len(statistics)}/{len(actual_denoised) + len(actual_residual)}")
    print(f"  float32 equation max error: {maximum_error:g}")
    if maximum_error != 0:
        raise RuntimeError("Noise2Noise float32 equation failed")
    print("  DQ: upstream flags retained")
    print("  weak-source performance: requires separate blind injection evaluation")


if __name__ == "__main__":
    main()
