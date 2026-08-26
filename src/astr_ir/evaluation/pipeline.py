"""End-to-end, model-agnostic blind mock-source evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import binomtest

from .mock_sources import (
    EvaluationConfig,
    build_empirical_psf,
    detect_sources,
    inject_sources,
    make_evaluation_mask,
    match_catalogs,
    sample_injection_positions,
)


InferenceFunction = Callable[[np.ndarray, np.ndarray, pd.Series], np.ndarray]
MaskFunction = Callable[[pd.Series, np.ndarray], np.ndarray]


def _load_image(input_root: Path, row: pd.Series) -> np.ndarray:
    return np.asarray(fits.getdata(input_root / str(row["product_path"])), dtype=np.float32)


def _serialize_config(config: EvaluationConfig, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)


def _training_psf(
    frames: pd.DataFrame,
    input_root: Path,
    mask_function: MaskFunction,
    config: EvaluationConfig,
    output_root: Path,
) -> np.ndarray:
    from .blind_joint import inspect_frame
    # Equal, deterministic per-sequence sample budget, training frames only.
    # No catalog SNR or target track is allowed to choose the PSF stars.
    training = frames.loc[frames["split"] == "train"].groupby("sequence",sort=True).head(12).copy()
    if training.empty:
        raise RuntimeError("No training frames are available for blind empirical PSF construction")
    samples = []
    metadata = []
    for row in training.itertuples(index=False):
        series = pd.Series(row._asdict())
        image = _load_image(input_root, series)
        valid = mask_function(series, image)
        _,_,_,_,stars,noise = inspect_frame(input_root / str(series["product_path"]), ~valid)
        for x,y,width,flux in stars:
            estimated_snr=flux/(2*np.sqrt(np.pi)*width*noise)
            if estimated_snr >= config.psf_min_training_snr:
                samples.append((image,valid,float(x),float(y)))
                metadata.append((str(series["frame_id"]),str(series["sequence"])))
    psf, diagnostics = build_empirical_psf(samples, size=config.psf_size)
    diagnostics["frame_id"] = [metadata[i][0] for i in diagnostics["sample_index"]]
    diagnostics["sequence"] = [metadata[i][1] for i in diagnostics["sample_index"]]
    diagnostics.to_csv(output_root / "psf_training_diagnostics.csv", index=False, encoding="utf-8-sig")
    fits.PrimaryHDU(psf).writeto(output_root / "empirical_psf.fits", overwrite=True)
    return psf


def _known_sources(row: pd.Series) -> list[tuple[float, float]]:
    """Legacy compatibility: exclusion is now image-derived, never catalog-driven."""
    return []


def _run_trial(
    row: pd.Series,
    image: np.ndarray,
    valid: np.ndarray,
    evaluation_mask: np.ndarray,
    psf: np.ndarray,
    inference: InferenceFunction,
    snrs: list[float],
    threshold: float,
    config: EvaluationConfig,
    rng: np.random.Generator,
    trial_id: str,
    split: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    positions = sample_injection_positions(
        rng,
        evaluation_mask,
        count=len(snrs),
        minimum_separation=config.minimum_injection_separation,
    )
    injected, truth = inject_sources(
        image,
        valid,
        psf,
        positions,
        target_snrs=snrs,
        noise_mask=evaluation_mask,
    )
    output = np.asarray(inference(injected, valid, row), dtype=np.float32)
    if output.shape != image.shape:
        raise ValueError(f"Inference changed image shape {image.shape} -> {output.shape}")
    catalogs: dict[str, pd.DataFrame] = {}
    for method, candidate in (("input", injected), ("output", output)):
        catalog, _, _, _ = detect_sources(
            candidate,
            valid,
            psf,
            threshold=threshold,
            peak_separation=config.peak_separation,
            noise_mask=evaluation_mask,
            detection_mask=evaluation_mask,
        )
        catalogs[method] = catalog
    injection_rows: list[dict] = []
    detection_rows: list[dict] = []
    trial_rows: list[dict] = []
    for method, catalog in catalogs.items():
        matches, missed, false = match_catalogs(truth, catalog, config.match_radius)
        match_by_injection = matches.set_index("injection_id") if not matches.empty else None
        for source in truth.itertuples(index=False):
            matched = match_by_injection is not None and int(source.injection_id) in match_by_injection.index
            match = match_by_injection.loc[int(source.injection_id)] if matched else None
            measured_flux = float(match["measured_flux"]) if matched else np.nan
            injection_rows.append(
                {
                    "split": split,
                    "frame_id": str(row["frame_id"]),
                    "sequence": str(row["sequence"]),
                    "trial_id": trial_id,
                    "method": method,
                    "injection_id": int(source.injection_id),
                    "target_snr": float(source.target_snr),
                    "x_true": float(source.x_true),
                    "y_true": float(source.y_true),
                    "true_flux": float(source.true_flux),
                    "detected": bool(matched),
                    "score": float(match["score"]) if matched else np.nan,
                    "measured_flux": measured_flux,
                    "relative_flux_error": (
                        (measured_flux - float(source.true_flux)) / float(source.true_flux)
                        if matched else np.nan
                    ),
                    "astrometric_error": float(match["distance"]) if matched else np.nan,
                }
            )
        matched_ids = set(matches["detection_id"].astype(int)) if not matches.empty else set()
        for detection in catalog.itertuples(index=False):
            detection_rows.append(
                {
                    "split": split,
                    "frame_id": str(row["frame_id"]),
                    "sequence": str(row["sequence"]),
                    "trial_id": trial_id,
                    "method": method,
                    "detection_id": int(detection.detection_id),
                    "x": float(detection.x),
                    "y": float(detection.y),
                    "score": float(detection.score),
                    "flux": float(detection.flux),
                    "matched": int(detection.detection_id) in matched_ids,
                }
            )
        tp, fn, fp = len(matches), len(missed), len(false)
        trial_rows.append(
            {
                "split": split,
                "frame_id": str(row["frame_id"]),
                "sequence": str(row["sequence"]),
                "trial_id": trial_id,
                "method": method,
                "target_snr": float(snrs[0]) if len(set(snrs)) == 1 else np.nan,
                "threshold": float(threshold),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "detections": int(len(catalog)),
            }
        )
    return injection_rows, detection_rows, trial_rows


def _calibrate_threshold(
    frames: pd.DataFrame,
    input_root: Path,
    mask_function: MaskFunction,
    inference: InferenceFunction,
    psf: np.ndarray,
    config: EvaluationConfig,
    output_root: Path,
) -> tuple[float, pd.DataFrame]:
    validation = frames.loc[frames["split"] == "validation"].copy()
    if validation.empty:
        raise RuntimeError("A validation split is required for detector-threshold calibration")
    rng = np.random.default_rng(config.seed + 1000)
    cached: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    sources_per_level = max(1, config.validation_sources_per_frame // len(config.validation_snrs))
    for _, row in validation.iterrows():
        image = _load_image(input_root, row)
        valid = mask_function(row, image)
        evaluation_mask, _ = make_evaluation_mask(
            image,
            valid,
            psf,
            config.edge_width,
            known_sources=_known_sources(row),
            source_exclusion_radius=config.source_exclusion_radius,
            blank_detection_threshold=config.blank_detection_threshold,
            peak_separation=config.peak_separation,
        )
        snrs = list(config.validation_snrs) * sources_per_level
        positions = sample_injection_positions(
            rng, evaluation_mask, len(snrs), config.minimum_injection_separation
        )
        injected, truth = inject_sources(
            image, valid, psf, positions, snrs, noise_mask=evaluation_mask
        )
        output = np.asarray(inference(injected, valid, row), dtype=np.float32)
        for method, candidate in (("input", injected), ("output", output)):
            catalog, _, _, _ = detect_sources(
                candidate,
                valid,
                psf,
                threshold=min(config.threshold_grid),
                peak_separation=config.peak_separation,
                noise_mask=evaluation_mask,
                detection_mask=evaluation_mask,
            )
            catalog["method"] = method
            catalog["frame_id"] = str(row["frame_id"])
            truth_copy = truth.copy()
            truth_copy["frame_id"] = str(row["frame_id"])
            cached.append((method, truth_copy, catalog))
    rows = []
    for threshold in config.threshold_grid:
        totals = {method: {"tp": 0, "fn": 0, "fp": 0} for method in ("input", "output")}
        for method, truth, full_catalog in cached:
            catalog = full_catalog.loc[full_catalog["score"] >= threshold]
            matches, missed, false = match_catalogs(truth, catalog, config.match_radius)
            totals[method]["tp"] += len(matches)
            totals[method]["fn"] += len(missed)
            totals[method]["fp"] += len(false)
        method_scores = []
        minimum_purity = 1.0
        for method, counts in totals.items():
            tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]
            completeness = tp / max(tp + fn, 1)
            purity = tp / max(tp + fp, 1)
            f1 = 2 * completeness * purity / max(completeness + purity, np.finfo(float).eps)
            method_scores.append(f1)
            minimum_purity = min(minimum_purity, purity)
            rows.append(
                {
                    "threshold": float(threshold),
                    "method": method,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "completeness": completeness,
                    "purity": purity,
                    "f1": f1,
                    "minimum_method_purity": np.nan,
                    "mean_method_f1": np.nan,
                    "passes_purity_gate": False,
                }
            )
        for row_out in rows[-2:]:
            row_out["minimum_method_purity"] = minimum_purity
            row_out["mean_method_f1"] = float(np.mean(method_scores))
            row_out["passes_purity_gate"] = minimum_purity >= config.target_validation_purity
    table = pd.DataFrame(rows)
    choices = table.drop_duplicates("threshold")
    passing = choices.loc[choices["passes_purity_gate"]]
    pool = passing if not passing.empty else choices
    selected = float(pool.sort_values(["mean_method_f1", "threshold"]).iloc[-1]["threshold"])
    table["selected"] = np.isclose(table["threshold"], selected)
    table.to_csv(output_root / "validation_threshold_calibration.csv", index=False, encoding="utf-8-sig")
    with (output_root / "selected_threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_threshold": selected,
                "selection_split": "validation",
                "target_minimum_purity": config.target_validation_purity,
                "purity_gate_satisfied": bool(not passing.empty),
            },
            handle,
            indent=2,
        )
    return selected, table


def _cluster_bootstrap_interval(
    group: pd.DataFrame,
    numerator: str,
    other: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap complete frames within each sequence to preserve within-frame dependence."""

    rng = np.random.default_rng(seed)
    by_frame = (
        group.groupby(["sequence", "frame_id"], as_index=False)[[numerator, other]].sum()
    )
    sequence_groups = [part.reset_index(drop=True) for _, part in by_frame.groupby("sequence")]
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        numerator_total = 0
        denominator_total = 0
        for part in sequence_groups:
            indices = rng.integers(0, len(part), size=len(part))
            sampled = part.iloc[indices]
            numerator_total += int(sampled[numerator].sum())
            denominator_total += int(sampled[numerator].sum() + sampled[other].sum())
        values[iteration] = numerator_total / max(denominator_total, 1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _summarize_test(trials: pd.DataFrame, injections: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    rows = []
    test_trials = trials.loc[trials["split"] == "test"]
    test_injections = injections.loc[injections["split"] == "test"]
    for (method, target_snr), group in test_trials.groupby(["method", "target_snr"], dropna=True):
        tp, fn, fp = int(group["tp"].sum()), int(group["fn"].sum()), int(group["fp"].sum())
        completeness = tp / max(tp + fn, 1)
        purity = tp / max(tp + fp, 1)
        f1 = 2 * completeness * purity / max(completeness + purity, np.finfo(float).eps)
        stable_seed = config.seed + int(round(float(target_snr) * 100)) + (0 if method == "input" else 1)
        c_low, c_high = _cluster_bootstrap_interval(
            group, "tp", "fn", config.bootstrap_iterations, stable_seed
        )
        p_low, p_high = _cluster_bootstrap_interval(
            group, "tp", "fp", config.bootstrap_iterations, stable_seed + 100_000
        )
        recovered = test_injections.loc[
            (test_injections["method"] == method)
            & np.isclose(test_injections["target_snr"], float(target_snr))
            & test_injections["detected"]
        ]
        rows.append(
            {
                "method": method,
                "target_snr": float(target_snr),
                "injected": tp + fn,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "completeness": completeness,
                "completeness_ci_low": c_low,
                "completeness_ci_high": c_high,
                "purity": purity,
                "purity_ci_low": p_low,
                "purity_ci_high": p_high,
                "false_discovery_rate": 1.0 - purity,
                "false_positives_per_frame": fp / max(group["frame_id"].nunique(), 1),
                "f1": f1,
                "median_relative_flux_error": float(recovered["relative_flux_error"].median()),
                "mad_relative_flux_error": float(
                    1.4826
                    * np.median(
                        np.abs(
                            recovered["relative_flux_error"]
                            - recovered["relative_flux_error"].median()
                        )
                    )
                ) if not recovered.empty else np.nan,
                "median_astrometric_error_pixels": float(recovered["astrometric_error"].median()),
                "threshold": float(group["threshold"].iloc[0]),
                "confidence_interval": "stratified frame-cluster bootstrap 95%",
            }
        )
    return pd.DataFrame(rows).sort_values(["target_snr", "method"]).reset_index(drop=True)


def _unmodified_catalogs(
    frames: pd.DataFrame,
    input_root: Path,
    mask_function: MaskFunction,
    inference: InferenceFunction,
    psf: np.ndarray,
    threshold: float,
    config: EvaluationConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in frames.loc[frames["split"] == "test"].iterrows():
        image = _load_image(input_root, row)
        valid = mask_function(row, image)
        evaluation_mask, _ = make_evaluation_mask(
            image,
            valid,
            psf,
            config.edge_width,
            known_sources=_known_sources(row),
            source_exclusion_radius=config.source_exclusion_radius,
            blank_detection_threshold=config.blank_detection_threshold,
            peak_separation=config.peak_separation,
        )
        output = np.asarray(inference(image, valid, row), dtype=np.float32)
        catalogs = {}
        for method, candidate in (("input", image), ("output", output)):
            catalogs[method], _, _, _ = detect_sources(
                candidate,
                valid,
                psf,
                threshold=threshold,
                peak_separation=config.peak_separation,
                noise_mask=evaluation_mask,
                detection_mask=evaluation_mask,
            )
        matches, _, new_output = match_catalogs(
            catalogs["input"].rename(columns={"detection_id": "injection_id", "x": "x_true", "y": "y_true"}),
            catalogs["output"],
            config.match_radius,
        )
        matched_output = set(matches["detection_id"].astype(int)) if not matches.empty else set()
        for method, catalog in catalogs.items():
            for detection in catalog.itertuples(index=False):
                rows.append(
                    {
                        "frame_id": str(row["frame_id"]),
                        "sequence": str(row["sequence"]),
                        "method": method,
                        "detection_id": int(detection.detection_id),
                        "x": float(detection.x),
                        "y": float(detection.y),
                        "score": float(detection.score),
                        "flux": float(detection.flux),
                        "new_relative_to_input": bool(
                            method == "output" and int(detection.detection_id) not in matched_output
                        ),
                    }
                )
        if new_output.empty and catalogs["output"].empty:
            continue
    return pd.DataFrame(rows)


def _interpolated_limit(group: pd.DataFrame, metric: str, target: float) -> float:
    ordered = group.sort_values("target_snr")
    x = ordered["target_snr"].to_numpy(float)
    y = np.maximum.accumulate(ordered[metric].to_numpy(float))
    if y.size == 0 or target < y[0] or target > y[-1]:
        return float("nan")
    index = int(np.searchsorted(y, target, side="left"))
    if index == 0 or np.isclose(y[index], y[index - 1]):
        return float(x[index])
    fraction = (target - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def _evaluation_summary(metrics: pd.DataFrame, unmodified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    limits: dict[tuple[str, float], float] = {}
    for target in (0.5, 0.9):
        for method, group in metrics.groupby("method"):
            limit = _interpolated_limit(group, "completeness", target)
            limits[(method, target)] = limit
            rows.append(
                {
                    "metric": f"snr_at_{int(target * 100)}pct_completeness_{method}",
                    "value": limit,
                }
            )
        input_limit = limits.get(("input", target), np.nan)
        output_limit = limits.get(("output", target), np.nan)
        rows.append(
            {
                "metric": f"snr_limit_improvement_at_{int(target * 100)}pct_completeness",
                "value": input_limit - output_limit,
            }
        )
    for snr in sorted(metrics["target_snr"].unique()):
        subset = metrics.loc[np.isclose(metrics["target_snr"], snr)].set_index("method")
        if {"input", "output"}.issubset(subset.index):
            rows.append(
                {
                    "metric": f"completeness_gain_snr_{snr:g}",
                    "value": float(
                        subset.loc["output", "completeness"]
                        - subset.loc["input", "completeness"]
                    ),
                }
            )
    if unmodified.empty:
        input_count = output_count = new_count = 0
    else:
        input_count = int((unmodified["method"] == "input").sum())
        output_count = int((unmodified["method"] == "output").sum())
        new_count = int(unmodified["new_relative_to_input"].fillna(False).sum())
    rows.extend(
        [
            {"metric": "unmodified_input_blank_region_detections", "value": input_count},
            {"metric": "unmodified_output_blank_region_detections", "value": output_count},
            {"metric": "unmodified_output_new_candidates", "value": new_count},
        ]
    )
    return pd.DataFrame(rows)


def _paired_comparison(injections: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    keys = ["sequence", "frame_id", "trial_id", "injection_id", "target_snr"]
    paired = injections.pivot_table(
        index=keys,
        columns="method",
        values="detected",
        aggfunc="first",
    ).reset_index()
    if not {"input", "output"}.issubset(paired.columns):
        raise RuntimeError("Paired input/output recovery records are incomplete")
    paired["input"] = paired["input"].astype(bool)
    paired["output"] = paired["output"].astype(bool)
    rows = []
    for target_snr, group in paired.groupby("target_snr"):
        input_only = int((group["input"] & ~group["output"]).sum())
        output_only = int((~group["input"] & group["output"]).sum())
        both = int((group["input"] & group["output"]).sum())
        neither = int((~group["input"] & ~group["output"]).sum())
        group = group.copy()
        group["gain"] = group["output"].astype(int) - group["input"].astype(int)
        by_frame = group.groupby(["sequence", "frame_id"], as_index=False).agg(
            gain_sum=("gain", "sum"), injections=("gain", "size")
        )
        sequence_groups = [part.reset_index(drop=True) for _, part in by_frame.groupby("sequence")]
        rng = np.random.default_rng(config.seed + 500_000 + int(round(float(target_snr) * 100)))
        bootstrap = np.empty(config.bootstrap_iterations, dtype=float)
        for iteration in range(config.bootstrap_iterations):
            gain_sum = 0
            count = 0
            for part in sequence_groups:
                indices = rng.integers(0, len(part), size=len(part))
                sampled = part.iloc[indices]
                gain_sum += int(sampled["gain_sum"].sum())
                count += int(sampled["injections"].sum())
            bootstrap[iteration] = gain_sum / max(count, 1)
        discordant = input_only + output_only
        p_value = (
            float(binomtest(min(input_only, output_only), discordant, p=0.5).pvalue)
            if discordant > 0
            else 1.0
        )
        rows.append(
            {
                "target_snr": float(target_snr),
                "injections": len(group),
                "both_detected": both,
                "neither_detected": neither,
                "input_only": input_only,
                "output_only": output_only,
                "paired_completeness_gain": float(group["gain"].mean()),
                "gain_ci_low": float(np.quantile(bootstrap, 0.025)),
                "gain_ci_high": float(np.quantile(bootstrap, 0.975)),
                "mcnemar_exact_p": p_value,
                "confidence_interval": "stratified frame-cluster bootstrap 95%",
            }
        )
    return pd.DataFrame(rows)


def run_mock_source_evaluation(
    frames: pd.DataFrame,
    input_root: str | Path,
    output_root: str | Path,
    inference: InferenceFunction,
    mask_function: MaskFunction,
    config: EvaluationConfig | None = None,
) -> dict[str, pd.DataFrame | float | np.ndarray]:
    """Run validation-calibrated blind detection and frozen held-out testing."""

    config = config or EvaluationConfig()
    config.validate()
    input_root, output_root = Path(input_root), Path(output_root)
    _serialize_config(config, output_root)
    required = {"frame_id", "sequence", "split", "product_path", "input_snr", "track_x", "track_y"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"Frame manifest is missing columns: {sorted(missing)}")
    if not {"train", "validation", "test"}.issubset(set(frames["split"])):
        raise ValueError("Frame manifest must contain train, validation, and test splits")
    psf = _training_psf(frames, input_root, mask_function, config, output_root)
    selected_threshold, calibration = _calibrate_threshold(
        frames, input_root, mask_function, inference, psf, config, output_root
    )
    rng = np.random.default_rng(config.seed + 2000)
    injection_rows: list[dict] = []
    detection_rows: list[dict] = []
    trial_rows: list[dict] = []
    test = frames.loc[frames["split"] == "test"].copy()
    for _, row in test.iterrows():
        image = _load_image(input_root, row)
        valid = mask_function(row, image)
        evaluation_mask, _ = make_evaluation_mask(
            image,
            valid,
            psf,
            config.edge_width,
            known_sources=_known_sources(row),
            source_exclusion_radius=config.source_exclusion_radius,
            blank_detection_threshold=config.blank_detection_threshold,
            peak_separation=config.peak_separation,
        )
        for target_snr in config.test_snrs:
            for repeat in range(config.test_repeats_per_snr):
                trial_id = f"{row['frame_id']}:snr{target_snr:g}:r{repeat}"
                sources = [float(target_snr)] * config.test_sources_per_frame
                injected, detected, trial = _run_trial(
                    row,
                    image,
                    valid,
                    evaluation_mask,
                    psf,
                    inference,
                    sources,
                    selected_threshold,
                    config,
                    rng,
                    trial_id,
                    "test",
                )
                injection_rows.extend(injected)
                detection_rows.extend(detected)
                trial_rows.extend(trial)
        print(f"mock-source evaluation {row['frame_id']}", flush=True)
    injections = pd.DataFrame(injection_rows)
    detections = pd.DataFrame(detection_rows)
    trials = pd.DataFrame(trial_rows)
    metrics = _summarize_test(trials, injections, config)
    unmodified = _unmodified_catalogs(
        frames, input_root, mask_function, inference, psf, selected_threshold, config
    )
    summary = _evaluation_summary(metrics, unmodified)
    comparison = _paired_comparison(injections, config)
    injections.to_csv(output_root / "injection_recovery.csv", index=False, encoding="utf-8-sig")
    detections.to_csv(output_root / "blind_detections.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(output_root / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_root / "metrics_by_snr.csv", index=False, encoding="utf-8-sig")
    unmodified.to_csv(output_root / "unmodified_test_catalog.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_root / "paired_comparison_by_snr.csv", index=False, encoding="utf-8-sig")
    return {
        "psf": psf,
        "selected_threshold": selected_threshold,
        "calibration": calibration,
        "injections": injections,
        "detections": detections,
        "trials": trials,
        "metrics": metrics,
        "unmodified": unmodified,
        "summary": summary,
        "comparison": comparison,
    }
