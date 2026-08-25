"""Blind mock-source evaluation using genuinely independent exposure stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from astr_ir.evaluation.mock_sources import (
    EvaluationConfig,
    detect_sources,
    inject_sources,
    make_evaluation_mask,
    match_catalogs,
    sample_injection_positions,
)
from astr_ir.evaluation.pipeline import _summarize_test, _training_psf
from astr_ir.noise2noise.dataset import load_detector_mask

from .paper_pipeline import (
    _load_registered_stack,
    denoise_registered_exposures,
    load_paper_model,
)


@dataclass(frozen=True)
class PaperEvaluationConfig:
    threshold: float = 4.0
    repeats: int = 4
    sources_per_snr_per_coadd: int = 2
    seed: int = 20260825
    bootstrap_iterations: int = 1000


def _rows_from_catalogs(
    truth: pd.DataFrame,
    catalogs: dict[str, pd.DataFrame],
    *,
    sequence: str,
    split: str,
    trial_id: str,
    threshold: float,
    match_radius: float,
) -> tuple[list[dict], list[dict]]:
    unique_snrs = truth["target_snr"].dropna().unique()
    if len(unique_snrs) != 1:
        raise ValueError("Each detection trial must contain exactly one target SNR")
    target_snr = float(unique_snrs[0])
    injection_rows, trial_rows = [], []
    for method, catalog in catalogs.items():
        matches, missed, false = match_catalogs(truth, catalog, match_radius)
        match_by_source = matches.set_index("injection_id") if not matches.empty else None
        for source in truth.itertuples(index=False):
            matched = match_by_source is not None and int(source.injection_id) in match_by_source.index
            match = match_by_source.loc[int(source.injection_id)] if matched else None
            measured_flux = float(match["measured_flux"]) if matched else np.nan
            injection_rows.append(
                {
                    "split": split,
                    # Treat each independently generated injection trial as a
                    # cluster for uncertainty estimation.
                    "frame_id": trial_id,
                    "sequence": sequence,
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
                        if matched
                        else np.nan
                    ),
                    "astrometric_error": float(match["distance"]) if matched else np.nan,
                }
            )
        trial_rows.append(
            {
                "split": split,
                "frame_id": trial_id,
                "sequence": sequence,
                "trial_id": trial_id,
                "method": method,
                "target_snr": target_snr,
                "threshold": threshold,
                "tp": len(matches),
                "fn": len(missed),
                "fp": len(false),
                "detections": len(catalog),
            }
        )
    return injection_rows, trial_rows


def run_paper_mock_evaluation(
    input_root: str | Path,
    dataset_root: str | Path,
    model_output_root: str | Path,
    evaluation_root: str | Path,
    checkpoint_path: str | Path,
    *,
    sequences: Sequence[str] = ("90000002", "90000003"),
    device: str | None = None,
    config: PaperEvaluationConfig = PaperEvaluationConfig(),
) -> dict[str, pd.DataFrame | float]:
    """Inject each source into every distinct held-out exposure before ASTERIS8."""

    input_root, dataset_root = Path(input_root), Path(dataset_root)
    model_output_root, evaluation_root = Path(model_output_root), Path(evaluation_root)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(
        model_output_root / "manifests" / "split_manifest.csv",
        encoding="utf-8-sig",
        dtype={"sequence": str},
    )
    frames = frames.loc[frames.sequence.isin(map(str, sequences))].copy()
    detector_mask = load_detector_mask(dataset_root)
    model, _, model_config = load_paper_model(checkpoint_path, device=device)
    device_name = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    science = EvaluationConfig(
        seed=config.seed,
        test_sources_per_frame=len(EvaluationConfig().test_snrs) * config.sources_per_snr_per_coadd,
        test_repeats_per_snr=config.repeats,
        bootstrap_iterations=config.bootstrap_iterations,
    )

    def single_mask(row: pd.Series, image: np.ndarray) -> np.ndarray:
        return ~detector_mask & np.isfinite(image)

    psf = _training_psf(frames, input_root, single_mask, science, evaluation_root)
    injection_rows, trial_rows = [], []
    rng = np.random.default_rng(config.seed)
    # The threshold is fixed before evaluation, so no validation-set tuning is
    # performed here.  Restrict injections to the frozen test exposures.
    for split_name in ("test",):
        for sequence, group in frames.loc[frames.split.eq(split_name)].groupby("sequence", sort=True):
            group = group.sort_values("frame_index").reset_index(drop=True)
            physical, valid = _load_registered_stack(group, input_root, detector_mask)
            base_input, _, base_valid, _ = denoise_registered_exposures(
                physical, valid, model, model_config, device=device_name
            )
            source_available = bool(group.source_measurement_available.astype(bool).any())
            known = (
                [(float(group.reference_x.iloc[0]), float(group.reference_y.iloc[0]))]
                if source_available
                else []
            )
            evaluation_mask, _ = make_evaluation_mask(
                base_input,
                base_valid,
                psf,
                science.edge_width,
                known_sources=known,
                source_exclusion_radius=science.source_exclusion_radius,
            )
            for repeat in range(config.repeats):
                for target_snr in science.test_snrs:
                    positions = sample_injection_positions(
                        rng,
                        evaluation_mask,
                        count=config.sources_per_snr_per_coadd,
                        minimum_separation=science.minimum_injection_separation,
                    )
                    injected_coadd, truth = inject_sources(
                        base_input,
                        base_valid,
                        psf,
                        positions,
                        target_snrs=[float(target_snr)] * config.sources_per_snr_per_coadd,
                        noise_mask=evaluation_mask,
                    )
                    delta = (injected_coadd - base_input).astype(np.float32)
                    injected_stack = physical + delta[None]
                    input_result, output_result, result_valid, _ = denoise_registered_exposures(
                        injected_stack, valid, model, model_config, device=device_name
                    )
                    catalogs = {}
                    for method, candidate in (("input", input_result), ("output", output_result)):
                        catalog, _, _, _ = detect_sources(
                            candidate,
                            result_valid,
                            psf,
                            threshold=config.threshold,
                            peak_separation=science.peak_separation,
                            noise_mask=evaluation_mask,
                            detection_mask=evaluation_mask,
                        )
                        catalogs[method] = catalog
                    trial_id = f"{sequence}:{split_name}:snr{target_snr:g}:r{repeat}"
                    injected_rows, trials = _rows_from_catalogs(
                        truth,
                        catalogs,
                        sequence=str(sequence),
                        split=split_name,
                        trial_id=trial_id,
                        threshold=config.threshold,
                        match_radius=science.match_radius,
                    )
                    injection_rows.extend(injected_rows)
                    trial_rows.extend(trials)
                    print(f"{trial_id} complete", flush=True)
    injections = pd.DataFrame(injection_rows)
    trials = pd.DataFrame(trial_rows)
    test_injections = injections.loc[injections.split.eq("test")].copy()
    test_trials = trials.loc[trials.split.eq("test")].copy()
    metrics = _summarize_test(test_trials, test_injections, science)
    injections.to_csv(evaluation_root / "injection_recovery.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(evaluation_root / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(evaluation_root / "metrics_by_snr.csv", index=False, encoding="utf-8-sig")
    with (evaluation_root / "paper_evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"paper": asdict(config), "science": science.to_dict(), "sequences": list(sequences)},
            handle,
            indent=2,
        )
    return {"threshold": config.threshold, "injections": injections, "trials": trials, "metrics": metrics}
