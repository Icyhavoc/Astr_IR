"""Create a paired 160-versus-400 ASTERIS8 performance report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROCESSED / "evaluation" / "asteris_paper_comparison"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile_rows = []
    metrics = []
    injections = {}
    coadd_tables = []
    for profile in ("160", "400"):
        model_root = PROCESSED / f"asteris_paper_{profile}"
        evaluation_root = PROCESSED / "evaluation" / f"asteris_paper_{profile}"
        history = pd.read_csv(model_root / "checkpoints" / "training_history.csv")
        coadds = pd.read_csv(model_root / "paper_coadd_statistics.csv", dtype={"sequence": str})
        coadds["profile"] = profile
        coadd_tables.append(coadds)
        best = history.loc[history.validation_loss.idxmin()]
        profile_rows.append(
            {
                "profile": profile,
                "dataset_frames": int(profile),
                "training_sequences": 2 if profile == "160" else 5,
                "best_epoch": int(best.epoch),
                "best_validation_loss": float(best.validation_loss),
                "median_coadd_noise_ratio": float(coadds.noise_ratio.median()),
                "coadd_noise_reduction": float(1.0 - coadds.noise_ratio.median()),
            }
        )
        metric = pd.read_csv(evaluation_root / "metrics_by_snr.csv")
        metric["profile"] = profile
        metrics.append(metric)
        injections[profile] = pd.read_csv(evaluation_root / "injection_recovery.csv")
    profile_summary = pd.DataFrame(profile_rows)
    metric_table = pd.concat(metrics, ignore_index=True)
    reported_metrics = (
        "completeness",
        "purity",
        "f1",
        "fp",
        "false_positives_per_frame",
        "median_relative_flux_error",
    )
    output_metrics = metric_table.loc[metric_table.method.eq("output")].pivot(
        index="target_snr", columns="profile", values=list(reported_metrics)
    )
    output_metrics.columns = [f"{metric}_{profile}" for metric, profile in output_metrics.columns]
    output_metrics = output_metrics.reset_index()
    for metric in reported_metrics:
        output_metrics[f"{metric}_delta_400_minus_160"] = (
            output_metrics[f"{metric}_400"] - output_metrics[f"{metric}_160"]
        )
    key = ["split", "sequence", "trial_id", "method", "injection_id", "target_snr"]
    first = injections["160"].loc[
        injections["160"].split.eq("test") & injections["160"].method.eq("output")
    ]
    second = injections["400"].loc[
        injections["400"].split.eq("test") & injections["400"].method.eq("output")
    ]
    paired = first.merge(second, on=key, suffixes=("_160", "_400"), validate="one_to_one")
    if len(paired) != len(first) or len(paired) != len(second):
        raise RuntimeError("The two profiles do not contain the same injection IDs")
    for column in ("x_true", "y_true", "true_flux"):
        if not np.allclose(paired[f"{column}_160"], paired[f"{column}_400"], rtol=0, atol=1e-10):
            raise RuntimeError(f"Paired injection mismatch in {column}")
    paired_rows = []
    for snr, group in paired.groupby("target_snr"):
        only_160 = int((group.detected_160 & ~group.detected_400).sum())
        only_400 = int((~group.detected_160 & group.detected_400).sum())
        paired_rows.append(
            {
                "target_snr": snr,
                "injections": len(group),
                "both_detected": int((group.detected_160 & group.detected_400).sum()),
                "neither_detected": int((~group.detected_160 & ~group.detected_400).sum()),
                "only_160": only_160,
                "only_400": only_400,
                "net_400_detections": only_400 - only_160,
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    coadd_table = pd.concat(coadd_tables, ignore_index=True).pivot(
        index="sequence", columns="profile", values=["noise_before", "noise_after", "noise_ratio"]
    )
    coadd_table.columns = [f"{metric}_{profile}" for metric, profile in coadd_table.columns]
    coadd_table = coadd_table.reset_index()
    coadd_table["noise_ratio_delta_400_minus_160"] = (
        coadd_table["noise_ratio_400"] - coadd_table["noise_ratio_160"]
    )
    profile_summary.to_csv(OUT / "profile_summary.csv", index=False, encoding="utf-8-sig")
    coadd_table.to_csv(OUT / "coadd_noise_by_sequence.csv", index=False, encoding="utf-8-sig")
    output_metrics.to_csv(OUT / "metrics_160_vs_400.csv", index=False, encoding="utf-8-sig")
    paired_summary.to_csv(OUT / "paired_detection_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# ASTERIS8 160-frame versus 400-frame comparison",
        "",
        "Both models use the same official initialization, frozen shared test exposures, injection positions,",
        "detection threshold, preprocessing, and temporal-coadd evaluator.",
        "",
        "## Profile summary",
        "",
        profile_summary.to_markdown(index=False),
        "",
        "## Shared-test coadd noise",
        "",
        coadd_table.to_markdown(index=False),
        "",
        "## Blind source metrics",
        "",
        output_metrics.to_markdown(index=False),
        "",
        "## Direct paired detections",
        "",
        paired_summary.to_markdown(index=False),
    ]
    (OUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(profile_summary.to_string(index=False))
    print(coadd_table.to_string(index=False))
    print(output_metrics.to_string(index=False))
    print(paired_summary.to_string(index=False))


if __name__ == "__main__":
    main()
