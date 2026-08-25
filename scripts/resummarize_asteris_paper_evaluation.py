"""Rebuild paper-evaluation metrics from saved trial-level recovery tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.evaluation.mock_sources import EvaluationConfig  # noqa: E402
from astr_ir.evaluation.pipeline import _summarize_test  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("160", "400"), required=True)
    args = parser.parse_args()
    root = PROJECT_ROOT / "data" / "processed" / "evaluation" / f"asteris_paper_{args.profile}"
    with (root / "paper_evaluation_config.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    config = EvaluationConfig(**saved["science"])
    injections = pd.read_csv(root / "injection_recovery.csv", dtype={"sequence": str})
    trials = pd.read_csv(root / "trial_metrics.csv", dtype={"sequence": str})
    # Older outputs used one frame ID for every coadd repeat.  Trial IDs are
    # the actual independent Monte Carlo clusters and are stable across profiles.
    injections["frame_id"] = injections["trial_id"]
    trials["frame_id"] = trials["trial_id"]
    metrics = _summarize_test(trials, injections, config)
    injections.to_csv(root / "injection_recovery.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(root / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(root / "metrics_by_snr.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
