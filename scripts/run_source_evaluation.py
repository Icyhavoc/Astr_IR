"""Run model-agnostic blind mock-source evaluation for a trained image model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.evaluation import EvaluationConfig, run_mock_source_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("noise2noise", "asteris"), default="noise2noise")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "background",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "our_dataset",
    )
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "noise2noise",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "evaluation" / "noise2noise",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--test-sources-per-frame", type=int, default=8)
    parser.add_argument("--test-repeats-per-snr", type=int, default=1)
    return parser.parse_args()


def _noise2noise_adapter(args: argparse.Namespace):
    from astr_ir.noise2noise.dataset import load_detector_mask
    from astr_ir.noise2noise.processor import (
        Noise2NoiseConfig,
        denoise_array,
        load_calibrated_strength,
        load_manifests,
        load_model,
    )

    checkpoint = args.checkpoint or args.model_output_root / "checkpoints" / "best_checkpoint.pt"
    frames, _, scales = load_manifests(args.model_output_root)
    detector_mask = load_detector_mask(args.dataset_root)
    model, _ = load_model(checkpoint, device=args.device)
    strength = load_calibrated_strength(args.model_output_root)
    model_config = Noise2NoiseConfig()

    def mask_function(row: pd.Series, image: np.ndarray) -> np.ndarray:
        if detector_mask.shape != image.shape:
            raise ValueError(
                f"Detector mask shape {detector_mask.shape} does not match {image.shape}"
            )
        return ~detector_mask & np.isfinite(image)

    def inference(image: np.ndarray, valid: np.ndarray, row: pd.Series) -> np.ndarray:
        denoised, _ = denoise_array(
            image,
            valid,
            model,
            scales[str(row["sequence"])],
            tile_size=model_config.inference_tile_size,
            overlap=model_config.inference_overlap,
            strength=strength,
        )
        return denoised

    return frames, mask_function, inference


def _asteris_adapter(args: argparse.Namespace):
    """Expose ASTERIS through the same blind-evaluation callable interface."""

    from astr_ir.asteris.inference import denoise_array
    from astr_ir.asteris.processor import (
        AsterisConfig,
        load_calibrated_strength,
        load_manifests,
        load_model,
    )
    from astr_ir.noise2noise.dataset import load_detector_mask

    checkpoint = args.checkpoint or args.model_output_root / "checkpoints" / "best_checkpoint.pt"
    frames, _, normalizations = load_manifests(args.model_output_root)
    detector_mask = load_detector_mask(args.dataset_root)
    model, state = load_model(checkpoint, device=args.device)
    saved = state.get("config", {})
    config = AsterisConfig(**{key: value for key, value in saved.items() if key in AsterisConfig.__dataclass_fields__})
    strength = load_calibrated_strength(args.model_output_root)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    def mask_function(row: pd.Series, image: np.ndarray) -> np.ndarray:
        if detector_mask.shape != image.shape:
            raise ValueError(
                f"Detector mask shape {detector_mask.shape} does not match {image.shape}"
            )
        return ~detector_mask & np.isfinite(image)

    def inference(image: np.ndarray, valid: np.ndarray, row: pd.Series) -> np.ndarray:
        return denoise_array(
            image,
            valid,
            model,
            normalizations[str(row["sequence"])],
            patch_t=config.patch_t,
            device=device,
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            strength=strength,
        )

    return frames, mask_function, inference


def main() -> None:
    args = parse_args()
    if args.model == "noise2noise":
        frames, mask_function, inference = _noise2noise_adapter(args)
    elif args.model == "asteris":
        frames, mask_function, inference = _asteris_adapter(args)
    else:
        raise ValueError(f"Unsupported model adapter: {args.model}")
    config = replace(
        EvaluationConfig(),
        test_sources_per_frame=args.test_sources_per_frame,
        test_repeats_per_snr=args.test_repeats_per_snr,
    )
    results = run_mock_source_evaluation(
        frames,
        args.input_root,
        args.evaluation_root,
        inference,
        mask_function,
        config=config,
    )
    print(f"validation-selected threshold: {results['selected_threshold']:.2f}")
    print(results["metrics"].to_string(index=False))
    print("Generate figures in a clean CPU process with scripts/plot_source_evaluation.py")


if __name__ == "__main__":
    main()
