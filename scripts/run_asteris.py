"""Run the notebook-backed ASTERIS preparation, training, inference and evaluation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.asteris.processor import (
    AsterisConfig,
    calibrate_and_finalize,
    prepare_manifests,
    run_inference,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "infer", "calibrate", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "our_dataset")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "asteris")
    parser.add_argument("--model", choices=("asteris4", "asteris8"), default="asteris4")
    parser.add_argument("--patch-t", type=int)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-samples-per-epoch", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--f-maps", type=int, default=24)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--temporal-clip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_t = args.patch_t or (4 if args.model == "asteris4" else 8)
    config = AsterisConfig(
        model=args.model,
        patch_t=patch_t,
        patch_size=args.patch_size,
        epochs=args.epochs,
        train_samples_per_epoch=args.train_samples_per_epoch,
        validation_samples=args.validation_samples,
        batch_size=args.batch_size,
        f_maps=args.f_maps,
        temporal_clip=args.temporal_clip,
    )
    checkpoint = args.checkpoint or args.output_root / "checkpoints" / "best_checkpoint.pt"
    if args.stage in {"prepare", "all"}:
        split, windows, stats = prepare_manifests(
            args.input_root, args.dataset_root, args.output_root, config=config
        )
        print(split.groupby(["sequence", "split"]).size().to_string())
        print(windows.loc[windows["usable"]].groupby(["sequence", "split"]).size().to_string())
        print("train-only normalization", stats)
    if args.stage in {"train", "all"}:
        checkpoint, history = train_model(
            args.input_root,
            args.dataset_root,
            args.output_root,
            config=config,
            device=args.device,
            resume=args.resume,
        )
        print(f"best checkpoint: {checkpoint}")
        print(history.tail().to_string(index=False))
    if args.stage in {"infer", "all"}:
        statistics = run_inference(
            args.input_root,
            args.dataset_root,
            args.output_root,
            checkpoint,
            config=config,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(statistics.groupby(["sequence", "split"]).size().to_string())
    if args.stage in {"calibrate", "all"}:
        strength, calibration, statistics = calibrate_and_finalize(
            args.input_root,
            args.dataset_root,
            args.output_root,
            checkpoint,
            config=config,
            overwrite=args.overwrite,
        )
        print(f"selected validation-calibrated ASTERIS strength: {strength:.2f}")
        print(calibration.to_string(index=False))
        print(statistics.groupby(["sequence", "split"]).size().to_string())
    if args.stage in {"evaluate", "all"}:
        command = [
            sys.executable,
            "scripts/run_source_evaluation.py",
            "--model",
            "asteris",
            "--model-output-root",
            str(args.output_root),
            "--evaluation-root",
            str(PROJECT_ROOT / "data" / "processed" / "evaluation" / "asteris"),
        ]
        if args.device:
            command.extend(["--device", args.device])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
