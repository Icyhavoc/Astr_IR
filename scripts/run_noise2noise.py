"""Prepare, train, infer, and evaluate the self-supervised Noise2Noise baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.noise2noise.processor import (
    Noise2NoiseConfig,
    prepare_manifests,
    run_inference,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "infer", "all"), required=True)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "our_dataset")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "noise2noise")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-samples-per-epoch", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Frozen residual subtraction strength; default is full strength (1.0).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.strength <= 1:
        raise ValueError("--strength must be in [0, 1]")
    config = Noise2NoiseConfig(
        epochs=args.epochs,
        train_samples_per_epoch=args.train_samples_per_epoch,
        validation_samples=args.validation_samples,
        batch_size=args.batch_size,
    )
    checkpoint = args.checkpoint or args.output_root / "checkpoints" / "best_checkpoint.pt"
    if args.stage in {"prepare", "all"}:
        split, pairs, scales = prepare_manifests(
            args.input_root,
            args.dataset_root,
            args.output_root,
            config=config,
        )
        print("split counts")
        print(split.groupby(["sequence", "split"]).size().to_string())
        print("usable pair counts")
        print(pairs.loc[pairs["usable"]].groupby(["sequence", "split"]).size().to_string())
        print("normalization scales", scales)
    if args.stage in {"train", "all"}:
        checkpoint, history = train_model(
            args.input_root,
            args.dataset_root,
            args.output_root,
            config=config,
            device=args.device,
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
            strength=args.strength,
        )
        print(statistics.groupby(["sequence", "split"]).size().to_string())


if __name__ == "__main__":
    main()
