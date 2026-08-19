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
    FlickerConfig,
    correct_flicker,
    load_detector_mask,
    load_fits,
    load_measurement_table,
    sigma_clipped_profile,
    target_record_for_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-local-profile",
        action="store_true",
        help="Skip the slower 160-frame local row/column degradation audit.",
    )
    return parser.parse_args()


def read_product(path: Path) -> np.ndarray:
    """Strictly verify and read a generated FITS product."""

    with fits.open(path, memmap=False) as hdul:
        hdul.verify("exception")
        return np.asarray(hdul[0].data, dtype=np.float32)


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

    for row in flicker_stats.itertuples(index=False):
        original, _ = load_fits(raw_root / str(row.sequence) / row.filename)
        original = original.astype(np.float32)
        corrected = read_product(flicker_root / row.corrected_path)
        model = read_product(flicker_root / row.model_path)
        max_flicker_error = max(
            max_flicker_error,
            float(np.max(np.abs(corrected - (original - model)))),
        )
        verified_fits += 2

    for row in background_stats.itertuples(index=False):
        input_image = read_product(flicker_root / str(row.sequence) / row.input_filename)
        subtracted = read_product(background_root / row.subtracted_path)
        model = read_product(background_root / row.model_path)
        max_background_error = max(
            max_background_error,
            float(np.max(np.abs(subtracted - (input_image - model)))),
        )
        verified_fits += 2

    return max_flicker_error, max_background_error, verified_fits


def audit_local_profiles(
    raw_root: Path,
    flicker_stats: pd.DataFrame,
) -> dict[str, float | int | str]:
    config = FlickerConfig(direction="auto", profile_smooth_size=5)
    detector_mask = load_detector_mask(raw_root / "盲点表")
    measurements = load_measurement_table(raw_root / "单帧检测总表_新方法.csv")
    records: list[dict[str, float | int | str]] = []

    for row in flicker_stats.itertuples(index=False):
        if not bool(row.applied):
            continue
        path = raw_root / str(row.sequence) / row.filename
        image, _ = load_fits(path)
        target = target_record_for_file(measurements, row.filename)
        result = correct_flicker(
            image,
            detector_mask=detector_mask,
            target=target,
            config=config,
        )
        before_diag = (
            result.row_diagnostic
            if result.selected_direction == "row"
            else result.column_diagnostic
        )
        after_diag = sigma_clipped_profile(
            result.corrected - result.low_frequency_background,
            result.combined_mask,
            result.selected_direction,
            sigma=config.sigma_clip,
            maxiters=config.sigma_clip_iters,
        )
        before = before_diag.profile - np.nanmedian(before_diag.profile)
        after = after_diag.profile - np.nanmedian(after_diag.profile)
        increase = np.abs(after) - np.abs(before)
        records.append(
            {
                "filename": row.filename,
                "line_count": int(increase.size),
                "worse_lines": int(np.sum(increase > 0)),
                "worse_over_10_dn": int(np.sum(increase > 10.0)),
                "max_increase_dn": float(np.nanmax(increase)),
            }
        )

    frame_table = pd.DataFrame(records)
    worst = frame_table.loc[frame_table["max_increase_dn"].idxmax()]
    return {
        "audited_frames": int(len(frame_table)),
        "median_worse_lines_per_frame": float(frame_table["worse_lines"].median()),
        "max_worse_lines_per_frame": int(frame_table["worse_lines"].max()),
        "median_lines_over_10_dn_per_frame": float(frame_table["worse_over_10_dn"].median()),
        "max_lines_over_10_dn_per_frame": int(frame_table["worse_over_10_dn"].max()),
        "worst_single_line_increase_dn": float(worst["max_increase_dn"]),
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

    flicker_error, background_error, fits_count = validate_equations(
        raw_root, flicker_root, background_root, flicker_stats, background_stats
    )
    flicker_applied = flicker_stats[flicker_stats["applied"]]
    background_applied = background_stats[background_stats["applied"]]
    flicker_high_snr = flicker_stats[flicker_stats["input_snr"] >= 10]
    background_high_snr = background_stats[background_stats["input_snr"] >= 10]

    print("FITS/equations")
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
        for key, value in audit_local_profiles(raw_root, flicker_stats).items():
            print(f"  {key}: {value}")

    if fits_count != 640 or flicker_error != 0 or background_error != 0:
        raise RuntimeError("Product validation failed")


if __name__ == "__main__":
    main()
