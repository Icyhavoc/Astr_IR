"""Attach real 2MASS/Gaia sources to ASTERIS paper coadds and make overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astr_ir.asteris.catalog_overlay import run_catalog_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-catalogs", action="store_true", help="Ignore cached Gaia/2MASS query results")
    args = parser.parse_args()
    results = run_catalog_overlay(PROJECT_ROOT, refresh_catalogs=args.refresh_catalogs)
    for sequence, result in results.items():
        print(
            f"{sequence}: weak={result['weak_sources']}, catalog={result['catalog_sources_in_field']}, "
            f"WCS RMS={result['anchor_rms_pix']:.3f} px, scale={result['pixel_scale_arcsec']:.4f} arcsec/px"
        )
        print(f"  {result['output_dir']}")
        print(f"  {result['overview']}")
        print(f"  {result['cutouts']}")


if __name__ == "__main__":
    main()
