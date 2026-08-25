"""Batch background subtraction for flicker-corrected FITS files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.background.processor import BackgroundConfig, run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "flicker")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "our_dataset")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "background")
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Sequence folders to process; default discovers all flicker-corrected folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-sequence", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_batch(
        args.input_root,
        args.dataset_root,
        args.output_root,
        config=BackgroundConfig(),
        sequences=tuple(args.sequences) if args.sequences else None,
        overwrite=args.overwrite,
        limit_per_sequence=args.limit_per_sequence,
    )
    print(stats.groupby(["sequence", "status"]).size().to_string())
    print(f"\nStatistics: {args.output_root / 'background_statistics.csv'}")


if __name__ == "__main__":
    main()
