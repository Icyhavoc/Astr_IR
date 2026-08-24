"""Generate mock-source evaluation figures without importing a GPU model runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.evaluation.visualization import (
    plot_completeness_purity,
    plot_empirical_psf,
    plot_photometric_accuracy,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "evaluation" / "noise2noise",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=PROJECT_ROOT / "figures" / "evaluation_output" / "noise2noise",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.evaluation_root / "metrics_by_snr.csv", encoding="utf-8-sig")
    injections = pd.read_csv(args.evaluation_root / "injection_recovery.csv", encoding="utf-8-sig")
    psf = np.asarray(fits.getdata(args.evaluation_root / "empirical_psf.fits"), dtype=float)
    paths = [
        save_figure(plot_empirical_psf(psf), args.figure_root / "empirical_psf.png"),
        save_figure(
            plot_completeness_purity(metrics),
            args.figure_root / "completeness_purity.png",
        ),
        save_figure(
            plot_photometric_accuracy(injections),
            args.figure_root / "photometric_accuracy.png",
        ),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
