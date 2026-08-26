"""Run genuine multi-exposure injection/recovery for a paper-style ASTERIS8 model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.asteris.paper_evaluation import PaperEvaluationConfig, run_paper_mock_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("160", "400"), required=True)
    parser.add_argument("--device")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--sources-per-snr", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()
    model_root = PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{args.profile}"
    evaluation_root = PROJECT_ROOT / "data" / "processed" / "evaluation" / f"asteris_paper_{args.profile}"
    results = run_paper_mock_evaluation(
        PROJECT_ROOT / "data" / "processed" / "background",
        PROJECT_ROOT / "data" / "raw" / "our_dataset",
        model_root,
        evaluation_root,
        model_root / "checkpoints" / "best_checkpoint.pt",
        device=args.device,
        config=PaperEvaluationConfig(
            threshold=args.threshold,
            repeats=args.repeats,
            sources_per_snr_per_coadd=args.sources_per_snr,
        ),
    )
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
