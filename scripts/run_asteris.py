"""Main ASTERIS entry: paper-style multi-exposure ASTERIS8 (160/400 profiles)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.asteris.paper_pipeline import (
    PaperAsterisConfig,
    prepare_paper_dataset,
    run_paper_inference,
    train_paper_model,
)


PROFILES = {
    "160": ("90000002", "90000003"),
    "400": ("90000002", "90000003", "90000004", "90000005_1", "90000005_2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "infer", "all"), default="all")
    parser.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "our_dataset"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--samples-per-sequence", type=int, default=64)
    parser.add_argument("--validation-samples-per-sequence", type=int, default=16)
    parser.add_argument("--inference-tile-size", type=int, default=128)
    parser.add_argument("--inference-overlap", type=int, default=16)
    parser.add_argument("--random-initialization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluation-sequences",
        nargs="+",
        default=["90000002", "90000003"],
        help="Held-out sequences used for the primary 160-vs-400 comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = PROFILES[args.profile]
    output_root = args.output_root or (
        PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{args.profile}"
    )
    config = replace(
        PaperAsterisConfig(),
        epochs=args.epochs,
        patch_size=args.patch_size,
        samples_per_sequence=args.samples_per_sequence,
        validation_samples_per_sequence=args.validation_samples_per_sequence,
        inference_tile_size=args.inference_tile_size,
        inference_overlap=args.inference_overlap,
        initialize_from_official=not args.random_initialization,
    )
    if args.stage in {"prepare", "all"}:
        split, stacks = prepare_paper_dataset(
            args.input_root,
            args.dataset_root,
            output_root,
            sequences=sequences,
            config=config,
        )
        print("frame split")
        print(split.groupby(["sequence", "split"]).size().to_string())
        print("paper stacks")
        print(stacks[["sequence", "split", "frames"]].to_string(index=False))
    checkpoint = args.checkpoint or output_root / "checkpoints" / "best_checkpoint.pt"
    if args.stage in {"train", "all"}:
        checkpoint, history = train_paper_model(output_root, config=config, device=args.device)
        print(f"best checkpoint: {checkpoint}")
        print(history.tail().to_string(index=False))
    if args.stage in {"infer", "all"}:
        stats = run_paper_inference(
            args.input_root,
            args.dataset_root,
            output_root,
            checkpoint,
            evaluation_sequences=args.evaluation_sequences,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
