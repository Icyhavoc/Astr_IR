"""Catalog-backed astrometry and weak-source overlays for ASTERIS paper coadds.

The RKZ50 FITS headers contain pointing coordinates in the non-standard
``RA``/``DE`` cards, but no WCS rotation.  This module completes the WCS from
2MASS calibration stars visible in the registered input coadds, cross-matches
the infrared catalog to Gaia DR3, and writes non-destructive annotated copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
IRSA_GATOR_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"


@dataclass(frozen=True)
class AstrometricAnchor:
    designation: str
    rough_x: float
    rough_y: float


@dataclass(frozen=True)
class SequenceCalibration:
    target_designation: str
    anchors: tuple[AstrometricAnchor, ...]


# The rough image coordinates only select the local stellar peak.  Final
# centroids and the WCS are recomputed from the coadd on every run.
CALIBRATIONS: dict[str, SequenceCalibration] = {
    "90000002": SequenceCalibration(
        target_designation="04143544+2221048",
        anchors=(
            AstrometricAnchor("04143544+2221048", 475.92, 558.00),
            AstrometricAnchor("04144941+2218105", 724.00, 513.00),
            AstrometricAnchor("04143040+2227066", 237.00, 824.00),
        ),
    ),
    "90000003": SequenceCalibration(
        target_designation="05193452-1310364",
        anchors=(
            AstrometricAnchor("05193452-1310364", 477.86, 408.00),
            AstrometricAnchor("05192508-1307438", 288.00, 491.00),
            AstrometricAnchor("05193913-1305383", 403.00, 686.00),
        ),
    ),
}


def _download_text(url: str, parameters: dict[str, str], *, post: bool = False) -> str:
    encoded = urlencode(parameters).encode("utf-8")
    request = Request(url, data=encoded if post else None)
    if not post:
        request = Request(f"{url}?{encoded.decode('utf-8')}")
    request.add_header("User-Agent", "astr-ir-catalog-overlay/1.0")
    with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed archive URLs
        return response.read().decode("utf-8")


def query_2mass(
    ra_deg: float,
    dec_deg: float,
    cache_path: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve the 2MASS All-Sky Point Source Catalog around one pointing."""

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype={"designation": "string"})
    text = _download_text(
        IRSA_GATOR_URL,
        {
            "spatial": "cone",
            "catalog": "fp_psc",
            "objstr": f"{ra_deg:.8f} {dec_deg:.8f}",
            "radius": "20",
            "radunits": "arcmin",
            "outfmt": "1",
        },
    )
    catalog = ascii.read(text, format="ipac").to_pandas()
    catalog["designation"] = catalog["designation"].astype("string")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(cache_path, index=False)
    return catalog


def query_gaia_dr3(
    ra_deg: float,
    dec_deg: float,
    cache_path: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve Gaia DR3 astrometry and broad-band photometry."""

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype={"source_id": "string"})
    query = f"""
        SELECT source_id, ra, dec, pmra, pmdec, ref_epoch,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, 0.32)
        )
        AND phot_g_mean_mag < 21.0
    """
    text = _download_text(
        GAIA_TAP_URL,
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        post=True,
    )
    catalog = pd.read_csv(StringIO(text), dtype={"source_id": "string"})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(cache_path, index=False)
    return catalog


def _high_pass(image: np.ndarray) -> tuple[np.ndarray, float]:
    valid = np.isfinite(image)
    filled = np.where(valid, image, np.nanmedian(image))
    filtered = gaussian_filter(filled - gaussian_filter(filled, 12.0), 1.2)
    median = float(np.median(filtered[valid]))
    sigma = float(1.4826 * np.median(np.abs(filtered[valid] - median)))
    return filtered, max(sigma, np.finfo(float).eps)


def _local_centroid(filtered: np.ndarray, x: float, y: float, radius: int = 7) -> tuple[float, float]:
    x0, y0 = int(round(x)), int(round(y))
    ylo, yhi = max(0, y0 - radius), min(filtered.shape[0], y0 + radius + 1)
    xlo, xhi = max(0, x0 - radius), min(filtered.shape[1], x0 + radius + 1)
    cutout = filtered[ylo:yhi, xlo:xhi]
    peak_y, peak_x = np.unravel_index(np.nanargmax(cutout), cutout.shape)
    peak_x += xlo
    peak_y += ylo
    rr = 4
    cy0, cy1 = max(0, peak_y - rr), min(filtered.shape[0], peak_y + rr + 1)
    cx0, cx1 = max(0, peak_x - rr), min(filtered.shape[1], peak_x + rr + 1)
    patch = filtered[cy0:cy1, cx0:cx1]
    weights = np.clip(patch - np.nanpercentile(patch, 25), 0.0, None)
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return float(peak_x), float(peak_y)
    return float((weights * xx).sum() / weights.sum()), float((weights * yy).sum() / weights.sum())


def _fit_similarity(
    sky_arcsec: np.ndarray,
    image_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit x=aE-bN+tx, y=bE+aN+ty and return A, t, residuals."""

    rows, values = [], []
    for (east, north), (x, y) in zip(sky_arcsec, image_pixels, strict=True):
        rows.extend([[east, -north, 1.0, 0.0], [north, east, 0.0, 1.0]])
        values.extend([x, y])
    a, b, tx, ty = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
    matrix = np.array([[a, -b], [b, a]], dtype=float)
    translation = np.array([tx, ty], dtype=float)
    predicted = sky_arcsec @ matrix.T + translation
    residuals = np.linalg.norm(predicted - image_pixels, axis=1)
    return matrix, translation, residuals


def solve_wcs(
    sequence: str,
    image: np.ndarray,
    catalog_2mass: pd.DataFrame,
) -> tuple[WCS, pd.DataFrame, dict[str, float | list[dict[str, float | str]]]]:
    """Solve a TAN WCS from the configured, catalog-verified 2MASS anchors."""

    calibration = CALIBRATIONS[sequence]
    filtered, _ = _high_pass(image)
    indexed = catalog_2mass.set_index(catalog_2mass["designation"].astype(str))
    target = indexed.loc[calibration.target_designation]
    target_coord = SkyCoord(float(target.ra) * u.deg, float(target.dec) * u.deg)
    sky, measured, rows = [], [], []
    for anchor in calibration.anchors:
        source = indexed.loc[anchor.designation]
        coordinate = SkyCoord(float(source.ra) * u.deg, float(source.dec) * u.deg)
        east, north = target_coord.spherical_offsets_to(coordinate)
        x, y = _local_centroid(filtered, anchor.rough_x, anchor.rough_y)
        sky.append((east.arcsec, north.arcsec))
        measured.append((x, y))
        rows.append(
            {
                "designation": anchor.designation,
                "ra_deg": float(source.ra),
                "dec_deg": float(source.dec),
                "x_measured": x,
                "y_measured": y,
            }
        )
    sky_array = np.asarray(sky)
    measured_array = np.asarray(measured)
    matrix, translation, residuals = _fit_similarity(sky_array, measured_array)
    for row, residual in zip(rows, residuals, strict=True):
        row["fit_residual_pix"] = float(residual)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [float(target.ra), float(target.dec)]
    wcs.wcs.crpix = translation + 1.0  # FITS convention is one-based.
    wcs.wcs.cd = np.linalg.inv(matrix) / 3600.0
    scale_arcsec = 1.0 / math.sqrt(abs(np.linalg.det(matrix)))
    rotation_deg = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    solution = {
        "sequence": sequence,
        "reference_catalog": "2MASS All-Sky Point Source Catalog (fp_psc)",
        "target_designation": calibration.target_designation,
        "crval_ra_deg": float(target.ra),
        "crval_dec_deg": float(target.dec),
        "crpix1_fits": float(translation[0] + 1.0),
        "crpix2_fits": float(translation[1] + 1.0),
        "pixel_scale_arcsec": float(scale_arcsec),
        "rotation_deg": float(rotation_deg),
        "anchor_rms_pix": float(np.sqrt(np.mean(residuals**2))),
        "anchor_max_residual_pix": float(residuals.max()),
        "anchors": rows,
    }
    return wcs, pd.DataFrame(rows), solution


def _propagate_gaia(gaia: pd.DataFrame, epoch: float = 2026.20) -> SkyCoord:
    dt = epoch - gaia["ref_epoch"].fillna(2016.0).to_numpy(float)
    dec = gaia["dec"].to_numpy(float)
    pmra = gaia["pmra"].fillna(0.0).to_numpy(float)
    pmdec = gaia["pmdec"].fillna(0.0).to_numpy(float)
    ra_epoch = gaia["ra"].to_numpy(float) + pmra * dt / (3.6e6 * np.cos(np.deg2rad(dec)))
    dec_epoch = dec + pmdec * dt / 3.6e6
    return SkyCoord(ra_epoch * u.deg, dec_epoch * u.deg)


def crossmatch_catalogs(catalog_2mass: pd.DataFrame, gaia: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest Gaia DR3 source, using proper motion at both epochs."""

    result = catalog_2mass.copy()
    two_mass = SkyCoord(result.ra.to_numpy(float) * u.deg, result.dec.to_numpy(float) * u.deg)
    gaia_2000 = _propagate_gaia(gaia, 2000.0)
    index, separation, _ = two_mass.match_to_catalog_sky(gaia_2000)
    matched = separation.arcsec <= 2.0
    gaia_2026 = _propagate_gaia(gaia, 2026.20)
    result["gaia_source_id"] = ""
    result["gaia_match_sep_arcsec"] = np.nan
    result["gaia_g_mag"] = np.nan
    result["gaia_bp_rp"] = np.nan
    result["gaia_ra_2026_deg"] = np.nan
    result["gaia_dec_2026_deg"] = np.nan
    # Pixel projection stays in the same 2MASS/J2000 frame used by the WCS
    # anchors.  Gaia epoch-2026 positions are retained as separate metadata;
    # mixing epochs here would displace high-proper-motion sources.
    result["catalog_ra_deg"] = result.ra.to_numpy(float)
    result["catalog_dec_deg"] = result.dec.to_numpy(float)
    positions = np.where(matched)[0]
    result.loc[positions, "gaia_source_id"] = gaia.iloc[index[matched]].source_id.astype(str).to_numpy()
    result.loc[positions, "gaia_match_sep_arcsec"] = separation.arcsec[matched]
    result.loc[positions, "gaia_g_mag"] = gaia.iloc[index[matched]].phot_g_mean_mag.to_numpy()
    result.loc[positions, "gaia_bp_rp"] = gaia.iloc[index[matched]].bp_rp.to_numpy()
    result.loc[positions, "gaia_ra_2026_deg"] = gaia_2026.ra.deg[index[matched]]
    result.loc[positions, "gaia_dec_2026_deg"] = gaia_2026.dec.deg[index[matched]]
    return result


def _aperture_metrics(
    image: np.ndarray,
    filtered: np.ndarray,
    global_sigma: float,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    radius_max = 14
    x0, y0 = int(round(x)), int(round(y))
    xlo, xhi = max(0, x0 - radius_max), min(image.shape[1], x0 + radius_max + 1)
    ylo, yhi = max(0, y0 - radius_max), min(image.shape[0], y0 + radius_max + 1)
    patch = image[ylo:yhi, xlo:xhi]
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    radius = np.hypot(xx - x, yy - y)
    aperture = (radius <= 5.0) & np.isfinite(patch)
    annulus = (radius >= 8.0) & (radius <= 13.0) & np.isfinite(patch)
    if aperture.sum() < 20 or annulus.sum() < 80:
        return np.nan, np.nan, np.nan
    background = float(np.median(patch[annulus]))
    sigma = float(1.4826 * np.median(np.abs(patch[annulus] - background)))
    flux = float(np.sum(patch[aperture] - background))
    snr = flux / (max(sigma, np.finfo(float).eps) * math.sqrt(int(aperture.sum())))
    cut = filtered[max(0, y0 - 3) : y0 + 4, max(0, x0 - 3) : x0 + 4]
    peak_snr = float(np.nanmax(cut) / global_sigma)
    return flux, snr, peak_snr


def build_source_table(
    catalog: pd.DataFrame,
    wcs: WCS,
    images: dict[str, np.ndarray],
    sequence: str,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    coordinates = SkyCoord(
        catalog.catalog_ra_deg.to_numpy(float) * u.deg,
        catalog.catalog_dec_deg.to_numpy(float) * u.deg,
    )
    x, y = wcs.world_to_pixel(coordinates)
    table = catalog.copy()
    table["x_pix_0based"] = x
    table["y_pix_0based"] = y
    inside = (x >= 18) & (x <= images["input"].shape[1] - 19) & (y >= 18) & (y <= images["input"].shape[0] - 19)
    table = table.loc[inside].copy().reset_index(drop=True)
    for method, image in images.items():
        filtered, global_sigma = _high_pass(image)
        metrics = [
            _aperture_metrics(image, filtered, global_sigma, row.x_pix_0based, row.y_pix_0based)
            for row in table.itertuples()
        ]
        table[f"{method}_aperture_flux"] = [item[0] for item in metrics]
        table[f"{method}_aperture_snr"] = [item[1] for item in metrics]
        table[f"{method}_peak_snr"] = [item[2] for item in metrics]

    calibration = CALIBRATIONS[sequence]
    table["is_target"] = table.designation.astype(str) == calibration.target_designation
    anchor_ids = {anchor.designation for anchor in calibration.anchors}
    table["is_astrometric_anchor"] = table.designation.astype(str).isin(anchor_ids)
    seq_rows = manifest.loc[manifest.sequence.astype(str) == sequence]
    median_single_snr = float(seq_rows.input_snr.median())
    table["target_median_single_frame_snr"] = np.where(table.is_target, median_single_snr, np.nan)

    k_mag = pd.to_numeric(table.k_m, errors="coerce")
    weak = (
        (k_mag <= 14.5)
        & (table.input_peak_snr >= 1.8)
        & (table.input_peak_snr <= 15.0)
        & (~table.is_astrometric_anchor)
    )
    weak |= table.is_target & (median_single_snr < 10.0)
    table["is_weak_source"] = weak
    weak_indices = table.index[weak].tolist()
    weak_indices.sort(key=lambda i: (not bool(table.loc[i, "is_target"]), float(k_mag.loc[i])))
    weak_indices = weak_indices[:12]
    table["weak_label"] = ""
    for number, index in enumerate(weak_indices, 1):
        table.loc[index, "weak_label"] = f"W{number:02d}"
    table["is_weak_source"] = table.weak_label != ""
    return table.sort_values(["is_weak_source", "k_m"], ascending=[False, True]).reset_index(drop=True)


def _write_regions(table: pd.DataFrame, output_dir: Path, sequence: str) -> None:
    weak = table.loc[table.is_weak_source]
    image_lines = ["# Region file format: DS9 version 4.1", "global color=cyan width=2 font='helvetica 10 bold'", "image"]
    sky_lines = ["# Region file format: DS9 version 4.1", "global color=cyan width=2 font='helvetica 10 bold'", "fk5"]
    for row in weak.itertuples():
        color = "yellow" if row.is_target else "cyan"
        image_lines.append(
            f"circle({row.x_pix_0based + 1:.3f},{row.y_pix_0based + 1:.3f},9) # color={color} text={{{row.weak_label}}}"
        )
        sky_lines.append(
            f"circle({row.catalog_ra_deg:.8f},{row.catalog_dec_deg:.8f},6\") # color={color} text={{{row.weak_label}}}"
        )
    (output_dir / f"{sequence}_weak_sources_image.reg").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (output_dir / f"{sequence}_weak_sources_fk5.reg").write_text("\n".join(sky_lines) + "\n", encoding="utf-8")


def _write_annotated_fits(
    source_path: Path,
    destination: Path,
    wcs: WCS,
    table: pd.DataFrame,
) -> None:
    with fits.open(source_path) as hdul:
        primary = fits.PrimaryHDU(np.asarray(hdul[0].data), header=hdul[0].header.copy())
        inherited_extensions = [hdu.copy() for hdu in hdul[1:] if hdu.name != "CATALOG"]
    for key, value in wcs.to_header(relax=True).items():
        primary.header[key] = value
    primary.header["CATREF"] = ("2MASS+GaiaDR3", "Astrometric reference catalogs")
    primary.header.add_history("WCS solved from catalog-verified 2MASS anchors; image pixels unchanged.")
    columns = [
        "weak_label", "designation", "gaia_source_id", "catalog_ra_deg", "catalog_dec_deg",
        "j_m", "h_m", "k_m", "x_pix_0based", "y_pix_0based", "input_aperture_snr",
        "asteris160_aperture_snr", "asteris400_aperture_snr", "is_target", "is_weak_source",
    ]
    catalog = table[columns].copy()
    for column in ("weak_label", "designation", "gaia_source_id"):
        catalog[column] = catalog[column].fillna("").astype(str)
    catalog_hdu = fits.BinTableHDU(Table.from_pandas(catalog, index=False), name="CATALOG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, *inherited_extensions, catalog_hdu]).writeto(destination, overwrite=True)


def _display_limits(images: list[np.ndarray]) -> tuple[float, float]:
    sample = np.concatenate([image[np.isfinite(image)][::20] for image in images])
    return float(np.percentile(sample, 1.0)), float(np.percentile(sample, 99.7))


def plot_overview(
    images: dict[str, np.ndarray],
    table: pd.DataFrame,
    sequence: str,
    output_path: Path,
) -> None:
    weak = table.loc[table.is_weak_source]
    methods = [("input", "Input coadd"), ("asteris160", "ASTERIS8 — 160 frames"), ("asteris400", "ASTERIS8 — 400 frames")]
    vmin, vmax = _display_limits([images[key] for key, _ in methods])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    for ax, (key, title) in zip(axes, methods, strict=True):
        ax.imshow(images[key], origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for row in weak.itertuples():
            color = "#ffd43b" if row.is_target else "#00e5ff"
            ax.add_patch(Circle((row.x_pix_0based, row.y_pix_0based), 11, fill=False, color=color, linewidth=1.4))
            ax.text(row.x_pix_0based + 9, row.y_pix_0based + 9, row.weak_label, color=color, fontsize=8, weight="bold")
        ax.set_title(title)
        ax.set_xlabel("x [pixel, 0-based]")
    axes[0].set_ylabel("y [pixel, 0-based]")
    fig.suptitle(f"{sequence}: catalog-confirmed weak-source positions (2MASS / Gaia DR3)", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cutouts(
    images: dict[str, np.ndarray],
    table: pd.DataFrame,
    sequence: str,
    output_path: Path,
) -> None:
    weak = table.loc[table.is_weak_source]
    methods = [("input", "Input"), ("asteris160", "ASTERIS 160"), ("asteris400", "ASTERIS 400")]
    if weak.empty:
        return
    half = 18
    fig, axes = plt.subplots(len(weak), 3, figsize=(8.2, 2.45 * len(weak)), squeeze=False, constrained_layout=True)
    for row_index, row in enumerate(weak.itertuples()):
        x, y = int(round(row.x_pix_0based)), int(round(row.y_pix_0based))
        cuts = [images[key][y - half : y + half + 1, x - half : x + half + 1] for key, _ in methods]
        combined = np.concatenate([cut[np.isfinite(cut)] for cut in cuts])
        vmin, vmax = np.percentile(combined, [2.0, 99.5])
        for column, ((key, title), cut) in enumerate(zip(methods, cuts, strict=True)):
            ax = axes[row_index, column]
            ax.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.add_patch(Circle((half, half), 6, fill=False, color="#00e5ff", linewidth=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{row.weak_label}\nK={row.k_m:.2f}")
    fig.suptitle(f"{sequence}: same-stretch cutouts at real catalog positions", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_catalog_overlay(
    project_root: str | Path,
    *,
    sequences: tuple[str, ...] = ("90000002", "90000003"),
    refresh_catalogs: bool = False,
) -> dict[str, dict[str, Path | float | int]]:
    """Build catalog tables, WCS FITS copies, DS9 regions, and comparison plots."""

    root = Path(project_root)
    evaluation_root = root / "data" / "processed" / "evaluation" / "asteris_paper_catalog"
    figure_root = root / "figures" / "asteris_paper_output" / "catalog_overlays"
    manifest = pd.read_csv(root / "data" / "processed" / "asteris_paper_400" / "manifests" / "split_manifest.csv")
    results: dict[str, dict[str, Path | float | int]] = {}
    for sequence in sequences:
        raw_path = next((root / "data" / "raw" / "our_dataset" / sequence).glob("*.fits"))
        header = fits.getheader(raw_path, ignore_missing_end=True)
        ra_deg, dec_deg = float(header["RA"]), float(header["DE"])
        sequence_root = evaluation_root / sequence
        cache_root = evaluation_root / "catalog_cache"
        two_mass = query_2mass(ra_deg, dec_deg, cache_root / f"2mass_{sequence}.csv", refresh=refresh_catalogs)
        gaia = query_gaia_dr3(ra_deg, dec_deg, cache_root / f"gaia_dr3_{sequence}.csv", refresh=refresh_catalogs)

        paths = {
            "input": root / "data" / "processed" / "asteris_paper_400" / "coadds" / sequence / f"input_coadd_{sequence}.fits",
            "asteris160": root / "data" / "processed" / "asteris_paper_160" / "coadds" / sequence / f"asteris8_coadd_{sequence}.fits",
            "asteris400": root / "data" / "processed" / "asteris_paper_400" / "coadds" / sequence / f"asteris8_coadd_{sequence}.fits",
        }
        images = {name: fits.getdata(path).astype(float) for name, path in paths.items()}
        wcs, anchors, solution = solve_wcs(sequence, images["input"], two_mass)
        merged = crossmatch_catalogs(two_mass, gaia)
        source_table = build_source_table(merged, wcs, images, sequence, manifest)
        sequence_root.mkdir(parents=True, exist_ok=True)
        source_table.to_csv(sequence_root / "catalog_sources.csv", index=False)
        source_table.loc[source_table.is_weak_source].to_csv(sequence_root / "weak_sources.csv", index=False)
        anchors.to_csv(sequence_root / "astrometric_anchors.csv", index=False)
        (sequence_root / "astrometric_solution.json").write_text(json.dumps(solution, indent=2), encoding="utf-8")
        _write_regions(source_table, sequence_root, sequence)

        for profile, key in (("160", "asteris160"), ("400", "asteris400")):
            _write_annotated_fits(
                paths[key],
                sequence_root / f"asteris_paper_{profile}_asteris8_coadd_{sequence}_catalog.fits",
                wcs,
                source_table,
            )
        overview = figure_root / f"{sequence}_catalog_overlay.png"
        cutouts = figure_root / f"{sequence}_weak_source_cutouts.png"
        plot_overview(images, source_table, sequence, overview)
        plot_cutouts(images, source_table, sequence, cutouts)
        results[sequence] = {
            "weak_sources": int(source_table.is_weak_source.sum()),
            "catalog_sources_in_field": int(len(source_table)),
            "anchor_rms_pix": float(solution["anchor_rms_pix"]),
            "pixel_scale_arcsec": float(solution["pixel_scale_arcsec"]),
            "output_dir": sequence_root,
            "overview": overview,
            "cutouts": cutouts,
        }
    return results
