"""Display-only 2MASS overlays for one science product at each pipeline stage.

Catalog coordinates never enter processing, training, masking, or source selection.
The FITS headers lack a usable WCS, so the display transform is independently
fitted to three identified stars in each image.  Its anchor residual is not an
independent astrometric accuracy estimate.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from astr_ir.asteris.catalog_overlay import CALIBRATIONS, solve_wcs


DISPLAY_SEQUENCE = "90000003"
DISPLAY_SOURCE = "05194363-1308496"  # Selected by 2MASS K and central field position.


def plot_stage_starmap(
    science_fits: str | Path,
    catalog_csv: str | Path,
    *,
    stage: str,
    output_png: str | Path | None = None,
):
    """Show an unmodified science FITS with catalog-predicted positions.

    The main panel marks 2MASS K<12 sources; the inset shows a fixed, fainter
    K=12.715 catalog location.  Neither the circles nor display stretching
    imply a detection.  The returned figure can be displayed in a notebook.
    """
    science_fits = Path(science_fits)
    catalog_csv = Path(catalog_csv)
    catalog = pd.read_csv(catalog_csv, dtype={"designation": "string"})
    with fits.open(science_fits, memmap=False) as hdul:
        hdul.verify("silentfix")  # Old DATE-OBS formatting; in-memory only.
        header = hdul[0].header
        if str(header.get("OBJECT", "")) != DISPLAY_SEQUENCE:
            raise ValueError(
                f"Expected OBJECT={DISPLAY_SEQUENCE}, got {header.get('OBJECT')!r}: {science_fits}"
            )
        if abs(float(header.get("RA", np.nan)) - 79.894) > 0.3 or abs(
            float(header.get("DE", np.nan)) + 13.177
        ) > 0.3:
            raise ValueError(f"FITS pointing does not match 90000003: {science_fits}")
        image = np.asarray(hdul[0].data, dtype=float)
        dq = np.asarray(hdul["DQ"].data, dtype=bool) if "DQ" in hdul else None
    if image.shape != (1024, 1024):
        raise ValueError(f"Expected 90000003 1024x1024 image, got {image.shape}")
    valid = np.isfinite(image) if dq is None else np.isfinite(image) & ~dq
    if not np.any(valid):
        raise ValueError(f"No valid science pixels: {science_fits}")

    # Fit only for display.  This does not add or change any FITS WCS cards.
    wcs, _, solution = solve_wcs(DISPLAY_SEQUENCE, np.where(valid, image, np.nan), catalog)
    sky = SkyCoord(catalog["ra"].to_numpy() * u.deg, catalog["dec"].to_numpy() * u.deg)
    x, y = wcs.world_to_pixel(sky)  # zero-based matplotlib/FITS-array pixels
    bright = (
        np.isfinite(catalog["k_m"].to_numpy())
        & (catalog["k_m"].to_numpy() < 12.0)
        & (x >= 20) & (x < 1004) & (y >= 20) & (y < 1004)
    )
    anchor_names = {a.designation for a in CALIBRATIONS[DISPLAY_SEQUENCE].anchors}
    anchor = catalog["designation"].isin(anchor_names).to_numpy()
    faint_index = catalog.index[catalog["designation"] == DISPLAY_SOURCE]
    if len(faint_index) != 1:
        raise ValueError(f"Missing fixed display source {DISPLAY_SOURCE}")
    fx, fy = float(x[faint_index[0]]), float(y[faint_index[0]])
    faint_k = float(catalog.loc[faint_index[0], "k_m"])

    # Percentile stretch changes display only; pixel values and FITS stay intact.
    lo, hi = np.percentile(image[valid], [5.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    shown = np.ma.array(image, mask=~valid)
    fig, (field, detail) = plt.subplots(1, 2, figsize=(14, 6.5),
                                       gridspec_kw={"width_ratios": [1.4, 1]})
    for ax in (field, detail):
        ax.imshow(shown, origin="lower", cmap="gray", vmin=lo, vmax=hi,
                  interpolation="nearest")
        ax.set_xlabel("x (zero-based pixel)")
        ax.set_ylabel("y (zero-based pixel)")
    field.scatter(x[bright & ~anchor], y[bright & ~anchor], s=110,
                  facecolors="none", edgecolors="orange", linewidths=1.2,
                  label="2MASS K<12 (holdout positions)")
    field.scatter(x[bright & anchor], y[bright & anchor], s=130,
                  facecolors="none", edgecolors="cyan", marker="s",
                  linewidths=1.4, label="WCS fit anchors (3)")
    field.add_patch(Circle((fx, fy), 12, fill=False, color="lime", linewidth=1.5))
    field.set_title("Full field: catalog-predicted positions")
    field.legend(loc="lower right", fontsize=8, framealpha=0.85)
    detail.set_xlim(fx - 80, fx + 80)
    detail.set_ylim(fy - 80, fy + 80)
    detail.add_patch(Circle((fx, fy), 12, fill=False, color="lime", linewidth=1.5))
    detail.set_title(
        f"Fixed 2MASS location: {DISPLAY_SOURCE}\n"
        f"K={faint_k:.3f}; circle is not a detection"
    )
    fig.suptitle(
        f"{stage} | {science_fits.name}\n"
        f"90000003 · 2MASS fp_psc · fitted rotation {solution['rotation_deg']:.2f}° · "
        f"3-anchor RMS {solution['anchor_rms_pix']:.2f} px · independent accuracy unverified",
        fontsize=11,
    )
    fig.tight_layout()
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=135, metadata={"Software": "astr_ir stage_starmap"})
    return fig, solution
