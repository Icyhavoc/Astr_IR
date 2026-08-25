"""Validate all generated FITS products and report scientific quality metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.flicker.processor import (  # noqa: E402
    load_fits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-local-profile",
        action="store_true",
        help="Skip the local row/column degradation summary from the statistics table.",
    )
    return parser.parse_args()


def read_product(path: Path) -> np.ndarray:
    """Strictly verify and read a generated FITS product."""

    with fits.open(path, memmap=False) as hdul:
        hdul.verify("exception")
        data = np.asarray(hdul[0].data)
    if data.ndim != 2:
        raise RuntimeError(f"Expected a 2-D FITS product, got {data.shape}: {path}")
    if data.dtype.kind != "f" or data.dtype.itemsize != 4:
        raise RuntimeError(f"Expected a float32 FITS product, got {data.dtype}: {path}")
    return data.astype(np.float32, copy=False)


def equation_max_error(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(f"Equation shape mismatch for {label}: {actual.shape} != {expected.shape}")
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    if not np.array_equal(actual_finite, expected_finite):
        raise RuntimeError(f"Equation finite-pixel mask mismatch for {label}")
    finite = expected_finite
    if not np.any(finite):
        return 0.0
    error = float(np.max(np.abs(actual[finite] - expected[finite])))
    if not np.isfinite(error):
        raise RuntimeError(f"Non-finite equation error for {label}")
    return error


def validate_inventory(root: Path, expected: set[str], stage: str) -> None:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.fits")}
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
        f"{stage} FITS inventory mismatch; "
        f"missing={missing[:5]} ({len(missing)} total), "
        f"extra={extra[:5]} ({len(extra)} total)"
    )


def validate_equations(
    raw_root: Path,
    flicker_root: Path,
    background_root: Path,
    flicker_stats: pd.DataFrame,
    background_stats: pd.DataFrame,
) -> tuple[float, float, int]:
    max_flicker_error = 0.0
    max_background_error = 0.0
    verified_fits = 0
    expected_flicker: set[str] = set()
    expected_background: set[str] = set()

    for row in flicker_stats.itertuples(index=False):
        original, _ = load_fits(raw_root / str(row.sequence) / row.filename)
        original = original.astype(np.float32)
        corrected = read_product(flicker_root / row.corrected_path)
        model = read_product(flicker_root / row.model_path)
        expected_flicker.update((str(row.corrected_path), str(row.model_path)))
        max_flicker_error = max(
            max_flicker_error,
            equation_max_error(corrected, original - model, str(row.corrected_path)),
        )
        verified_fits += 2

    for row in background_stats.itertuples(index=False):
        input_image = read_product(flicker_root / str(row.sequence) / row.input_filename)
        subtracted = read_product(background_root / row.subtracted_path)
        model = read_product(background_root / row.model_path)
        expected_background.update((str(row.subtracted_path), str(row.model_path)))
        max_background_error = max(
            max_background_error,
            equation_max_error(subtracted, input_image - model, str(row.subtracted_path)),
        )
        verified_fits += 2

    if len(expected_flicker) != 2 * len(flicker_stats):
        raise RuntimeError("Duplicate flicker product paths are present in the statistics table")
    if len(expected_background) != 2 * len(background_stats):
        raise RuntimeError("Duplicate background product paths are present in the statistics table")
    validate_inventory(flicker_root, expected_flicker, "Flicker")
    validate_inventory(background_root, expected_background, "Background")
    return max_flicker_error, max_background_error, verified_fits


def audit_local_profiles(
    flicker_stats: pd.DataFrame,
) -> dict[str, float | int | str]:
    required = {
        "local_line_count",
        "local_worse_lines",
        "local_worse_over_threshold_lines",
        "local_max_increase_dn",
        "local_gate_passed",
        "selected_profile_smooth_size",
    }
    missing = sorted(required - set(flicker_stats.columns))
    if missing:
        raise RuntimeError(
            "Flicker statistics predate the local quality gate; rerun flicker processing. "
            f"Missing columns: {missing}"
        )
    frame_table = flicker_stats.loc[flicker_stats["applied"]].copy()
    if frame_table.empty:
        raise RuntimeError("No applied flicker frames are available for local quality audit")
    if not frame_table["local_gate_passed"].astype(bool).all():
        raise RuntimeError("An applied flicker product failed its recorded local quality gate")
    worst = frame_table.loc[frame_table["local_max_increase_dn"].idxmax()]
    return {
        "audited_frames": int(len(frame_table)),
        "profile_size_1_frames": int((frame_table["selected_profile_smooth_size"] == 1).sum()),
        "profile_size_3_frames": int((frame_table["selected_profile_smooth_size"] == 3).sum()),
        "profile_size_5_frames": int((frame_table["selected_profile_smooth_size"] == 5).sum()),
        "median_worse_lines_per_frame": float(frame_table["local_worse_lines"].median()),
        "max_worse_lines_per_frame": int(frame_table["local_worse_lines"].max()),
        "median_lines_over_10_dn_per_frame": float(
            frame_table["local_worse_over_threshold_lines"].median()
        ),
        "max_lines_over_10_dn_per_frame": int(
            frame_table["local_worse_over_threshold_lines"].max()
        ),
        "worst_single_line_increase_dn": float(worst["local_max_increase_dn"]),
        "worst_frame": str(worst["filename"]),
    }


def main() -> None:
    args = parse_args()
    raw_root = PROJECT_ROOT / "data" / "raw" / "our_dataset"
    flicker_root = PROJECT_ROOT / "data" / "processed" / "flicker"
    background_root = PROJECT_ROOT / "data" / "processed" / "background"
    flicker_stats = pd.read_csv(
        flicker_root / "flicker_statistics.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    background_stats = pd.read_csv(
        background_root / "background_statistics.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    raw_count = sum(1 for sequence in raw_root.iterdir() if sequence.is_dir() for _ in sequence.glob("*.fits"))

    flicker_error, background_error, fits_count = validate_equations(
        raw_root, flicker_root, background_root, flicker_stats, background_stats
    )
    flicker_applied = flicker_stats[flicker_stats["applied"]]
    background_applied = background_stats[background_stats["applied"]]
    flicker_high_snr = flicker_stats[flicker_stats["input_snr"] >= 10]
    background_high_snr = background_stats[background_stats["input_snr"] >= 10]

    print("FITS/equations")
    print(f"  raw frames discovered: {raw_count}")
    print(f"  strictly verified FITS: {fits_count}")
    print(f"  flicker equation max error: {flicker_error:g}")
    print(f"  background equation max error: {background_error:g}")
    print("Flicker")
    print(f"  frames/applied: {len(flicker_stats)}/{len(flicker_applied)}")
    print(
        "  profile reduction min/median/max: "
        f"{100*flicker_applied['relative_reduction'].min():.4f}% / "
        f"{100*flicker_applied['relative_reduction'].median():.4f}% / "
        f"{100*flicker_applied['relative_reduction'].max():.4f}%"
    )
    print(f"  max high-frequency ratio: {flicker_applied['background_noise_ratio'].max():.9f}")
    print(
        "  max |SNR>=10 flux change|: "
        f"{100*flicker_high_snr['photometry_change_fraction'].abs().max():.4f}%"
    )
    print("Background")
    print(f"  frames/applied: {len(background_stats)}/{len(background_applied)}")
    print(
        "  large-scale reduction min/median/max: "
        f"{100*background_applied['large_scale_reduction'].min():.4f}% / "
        f"{100*background_applied['large_scale_reduction'].median():.4f}% / "
        f"{100*background_applied['large_scale_reduction'].max():.4f}%"
    )
    print(f"  max high-frequency ratio: {background_applied['high_frequency_noise_ratio'].max():.9f}")
    print(
        "  max |SNR>=10 flux change|: "
        f"{100*background_high_snr['photometry_change_fraction'].abs().max():.4f}%"
    )

    if not args.skip_local_profile:
        print("Local flicker-profile degradation")
        for key, value in audit_local_profiles(flicker_stats).items():
            print(f"  {key}: {value}")

    expected_products = 4 * raw_count
    complete_statistics = len(flicker_stats) == raw_count and len(background_stats) == raw_count
    if (
        fits_count != expected_products
        or not complete_statistics
        or flicker_error != 0
        or background_error != 0
    ):
        raise RuntimeError("Product validation failed")


if __name__ == "__main__":
    main()
