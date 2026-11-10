"""Main RKZ50 ASTERIS entry with independent-field splits and blind selection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.asteris.paper_pipeline import (
    PaperAsterisConfig,
    RKZ50_FIELD_ASSIGNMENTS,
    prepare_paper_dataset,
    run_paper_inference,
    select_checkpoint_by_detection,
    selected_checkpoint,
    train_paper_model,
)


SEQUENCES = tuple(RKZ50_FIELD_ASSIGNMENTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=("compact4", "full8"), default="compact4")
    parser.add_argument("--stage", choices=("prepare", "train", "select", "infer"), required=True)
    parser.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "our_dataset"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection-table", type=Path,
                        help="Frozen catalog-free validation injection metrics for --stage select")
    parser.add_argument("--max-false-peaks-per-image", type=float, default=0.5)
    parser.add_argument("--flux-response-tolerance", type=float, default=0.05)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--samples-per-sequence", type=int)
    parser.add_argument("--validation-samples-per-sequence", type=int)
    parser.add_argument("--inference-tile-size", type=int)
    parser.add_argument("--inference-overlap", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluation-sequences",
        nargs="+",
        default=["90000005_1", "90000005_2"],
        help="Frozen test-field visits; both defaults are the same independent sky field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (
        PROJECT_ROOT / "data" / "processed" / "asteris_rkz50_field_split"
    )
    base = PaperAsterisConfig()
    if args.network == "full8":
        base = replace(base, model="asteris8", patch_t=8, patch_size=128, f_maps=24,
                       num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                       heads=(1, 2, 4, 8), samples_per_sequence=64,
                       validation_samples_per_sequence=16, inference_tile_size=128)
    config = replace(
        base,
        epochs=args.epochs,
        patch_size=args.patch_size or base.patch_size,
        samples_per_sequence=args.samples_per_sequence or base.samples_per_sequence,
        validation_samples_per_sequence=(args.validation_samples_per_sequence
                                         or base.validation_samples_per_sequence),
        inference_tile_size=args.inference_tile_size or base.inference_tile_size,
        inference_overlap=(args.inference_overlap if args.inference_overlap is not None
                           else base.inference_overlap),
    )
    if args.stage == "prepare":
        split, stacks = prepare_paper_dataset(
            args.input_root,
            args.dataset_root,
            output_root,
            sequences=SEQUENCES,
            config=config,
            field_assignments=RKZ50_FIELD_ASSIGNMENTS,
        )
        print("frame split")
        print(split.groupby(["sequence", "split"]).size().to_string())
        print("paper stacks")
        print(stacks[["sequence", "split", "frames"]].to_string(index=False))
    if args.stage == "train":
        checkpoint, history = train_paper_model(output_root, config=config, device=args.device)
        print(f"last checkpoint: {checkpoint}")
        print("formal checkpoint remains pending blind-injection detection selection")
        print(history.tail().to_string(index=False))
    if args.stage == "select":
        if args.selection_table is None:
            raise ValueError("--stage select requires --selection-table")
        checkpoint = select_checkpoint_by_detection(
            output_root, args.selection_table,
            max_false_peaks_per_image=args.max_false_peaks_per_image,
            flux_response_tolerance=args.flux_response_tolerance,
        )
        print(f"detection-selected checkpoint: {checkpoint}")
    if args.stage == "infer":
        checkpoint = args.checkpoint or selected_checkpoint(output_root)
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
