"""Strict validation of the reusable blind mock-source evaluation products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "evaluation" / "noise2noise",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "noise2noise"
        / "manifests"
        / "split_manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.evaluation_root
    required = [
        "evaluation_config.json",
        "empirical_psf.fits",
        "psf_training_diagnostics.csv",
        "validation_threshold_calibration.csv",
        "selected_threshold.json",
        "injection_recovery.csv",
        "blind_detections.csv",
        "trial_metrics.csv",
        "metrics_by_snr.csv",
        "unmodified_test_catalog.csv",
        "evaluation_summary.csv",
        "paired_comparison_by_snr.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Missing evaluation products: {missing}")
    split = pd.read_csv(args.split_manifest, encoding="utf-8-sig", dtype={"sequence": str})
    psf_diagnostics = pd.read_csv(
        root / "psf_training_diagnostics.csv", encoding="utf-8-sig", dtype={"sequence": str}
    )
    calibration = pd.read_csv(root / "validation_threshold_calibration.csv", encoding="utf-8-sig")
    injections = pd.read_csv(
        root / "injection_recovery.csv", encoding="utf-8-sig", dtype={"sequence": str}
    )
    detections = pd.read_csv(
        root / "blind_detections.csv", encoding="utf-8-sig", dtype={"sequence": str}
    )
    trials = pd.read_csv(root / "trial_metrics.csv", encoding="utf-8-sig", dtype={"sequence": str})
    metrics = pd.read_csv(root / "metrics_by_snr.csv", encoding="utf-8-sig")
    comparison = pd.read_csv(root / "paired_comparison_by_snr.csv", encoding="utf-8-sig")
    with (root / "selected_threshold.json").open(encoding="utf-8") as handle:
        threshold = json.load(handle)
    with (root / "evaluation_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    failures = []
    if threshold.get("selection_split") != "validation":
        failures.append("detector threshold was not selected on validation data")
    selected = float(threshold["selected_threshold"])
    selected_rows = calibration.loc[calibration["selected"]]
    if selected_rows.empty or not np.allclose(selected_rows["threshold"], selected):
        failures.append("selected threshold does not match calibration table")
    split_by_frame = split.set_index("frame_id")["split"].to_dict()
    psf_frames = set(psf_diagnostics.loc[psf_diagnostics["accepted"], "frame_id"].astype(str))
    if not psf_frames or any(split_by_frame.get(frame) != "train" for frame in psf_frames):
        failures.append("empirical PSF used non-training frames")
    evaluated_frames = set(injections["frame_id"].astype(str))
    if not evaluated_frames or any(split_by_frame.get(frame) != "test" for frame in evaluated_frames):
        failures.append("reported injection metrics include non-test frames")
    expected_physical = (
        len(evaluated_frames)
        * len(config["test_snrs"])
        * int(config["test_sources_per_frame"])
        * int(config["test_repeats_per_snr"])
    )
    if len(injections) != 2 * expected_physical:
        failures.append(
            f"expected {2 * expected_physical} input/output injection records, got {len(injections)}"
        )
    paired_truth = injections.pivot_table(
        index=["trial_id", "injection_id"],
        columns="method",
        values=["x_true", "y_true", "true_flux", "target_snr"],
        aggfunc="first",
    )
    for field in ("x_true", "y_true", "true_flux", "target_snr"):
        if not {"input", "output"}.issubset(paired_truth[field].columns) or not np.allclose(
            paired_truth[field]["input"], paired_truth[field]["output"], rtol=0, atol=0
        ):
            failures.append(f"input/output do not share identical {field} truth")
    expected_injections = trials.groupby(["trial_id", "method"])[["tp", "fn"]].sum().sum(axis=1)
    recorded_injections = injections.groupby(["trial_id", "method"]).size()
    if not expected_injections.equals(recorded_injections.reindex(expected_injections.index)):
        failures.append("trial TP+FN counts do not equal injection records")
    if injections.duplicated(["trial_id", "method", "injection_id"]).any():
        failures.append("an injected source was recorded more than once per method")
    matched_detection_counts = detections.loc[detections["matched"]].groupby(
        ["trial_id", "method", "detection_id"]
    ).size()
    if not matched_detection_counts.empty and int(matched_detection_counts.max()) != 1:
        failures.append("a detection was matched to multiple injected sources")
    if not metrics["completeness"].between(0, 1).all():
        failures.append("completeness outside [0, 1]")
    if not metrics["purity"].between(0, 1).all():
        failures.append("purity outside [0, 1]")
    if not comparison["mcnemar_exact_p"].between(0, 1).all():
        failures.append("paired-comparison p value outside [0, 1]")
    for row in comparison.itertuples(index=False):
        by_method = metrics.loc[np.isclose(metrics["target_snr"], row.target_snr)].set_index(
            "method"
        )
        expected_gain = by_method.loc["output", "completeness"] - by_method.loc[
            "input", "completeness"
        ]
        if not np.isclose(expected_gain, row.paired_completeness_gain, rtol=0, atol=1e-12):
            failures.append(f"paired gain mismatch at SNR={row.target_snr:g}")
        if row.both_detected + row.neither_detected + row.input_only + row.output_only != row.injections:
            failures.append(f"paired contingency count mismatch at SNR={row.target_snr:g}")
    psf = np.asarray(fits.getdata(root / "empirical_psf.fits"), dtype=float)
    if psf.ndim != 2 or not np.isfinite(psf).all() or np.any(psf < 0):
        failures.append("empirical PSF is invalid")
    if not np.isclose(psf.sum(), 1.0, rtol=0, atol=2e-6):
        failures.append("empirical PSF is not unit normalized")
    snr5 = metrics.loc[np.isclose(metrics["target_snr"], 5.0)].set_index("method")
    if {"input", "output"}.issubset(snr5.index):
        if snr5.loc["output", "completeness"] < snr5.loc["input", "completeness"]:
            failures.append("SNR=5 output completeness is worse than input")
        if snr5.loc["output", "purity"] < 0.90:
            failures.append("SNR=5 output purity is below 90%")
    else:
        failures.append("SNR=5 metrics are incomplete")
    print(f"selected threshold: {selected:.2f}")
    print(f"training PSF cutouts accepted: {len(psf_frames)}")
    print(f"test frames: {len(evaluated_frames)}")
    print(f"injection records: {len(injections)}")
    print(metrics.to_string(index=False))
    if failures:
        raise SystemExit("Validation failures:\n- " + "\n- ".join(failures))
    print("Source-evaluation validation passed.")


if __name__ == "__main__":
    main()
