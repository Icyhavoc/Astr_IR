"""Small FITS data-quality helpers shared by science-product writers."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np


DQ_DO_NOT_USE = np.uint16(1 << 0)
DQ_DETECTOR_BAD = np.uint16(1 << 1)
DQ_NO_COVERAGE = np.uint16(1 << 2)
DQ_PARTIAL_COVERAGE = np.uint16(1 << 3)


def build_dq(
    shape: tuple[int, int],
    *,
    detector_bad: np.ndarray | None = None,
    no_coverage: np.ndarray | None = None,
    partial_coverage: np.ndarray | None = None,
) -> np.ndarray:
    """Build a uint16 DQ image without changing the corresponding science data."""

    dq = np.zeros(shape, dtype=np.uint16)

    def checked(value: np.ndarray | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        mask = np.asarray(value, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{name} mask {mask.shape} does not match science image {shape}")
        return mask

    detector = checked(detector_bad, "detector_bad")
    missing = checked(no_coverage, "no_coverage")
    partial = checked(partial_coverage, "partial_coverage")
    if detector is not None:
        dq[detector] |= DQ_DO_NOT_USE | DQ_DETECTOR_BAD
    if missing is not None:
        dq[missing] |= DQ_DO_NOT_USE | DQ_NO_COVERAGE
    if partial is not None:
        dq[partial] |= DQ_PARTIAL_COVERAGE
    return dq


def write_fits_with_dq(
    path: str | Path,
    data: np.ndarray,
    header: fits.Header,
    dq: np.ndarray,
    *,
    overwrite: bool = False,
    output_verify: str = "exception",
) -> None:
    """Write unchanged science pixels in PRIMARY plus a machine-readable DQ image."""

    science = np.asarray(data)
    quality = np.asarray(dq, dtype=np.uint16)
    if science.shape != quality.shape:
        raise ValueError(f"DQ shape {quality.shape} does not match science data {science.shape}")
    primary_header = header.copy()
    primary_header["HIERARCH DQ EXT"] = ("DQ", "Data-quality image extension")
    dq_hdu = fits.ImageHDU(data=quality, name="DQ")
    dq_hdu.header["DQBIT0"] = ("DO_NOT_USE", "Pixel must not enter science estimates")
    dq_hdu.header["DQBIT1"] = ("DETECTOR_BAD", "Dead/noisy detector blind-map pixel")
    dq_hdu.header["DQBIT2"] = ("NO_COVERAGE", "No valid measurement contributes")
    dq_hdu.header["DQBIT3"] = ("PARTIAL", "Some contributing measurements are invalid")
    fits.HDUList([fits.PrimaryHDU(science, header=primary_header), dq_hdu]).writeto(
        path,
        overwrite=overwrite,
        output_verify=output_verify,
    )


def read_dq(path: str | Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Read a DQ extension when present, returning ``None`` for legacy products."""

    with fits.open(path, memmap=False) as hdul:
        if "DQ" not in hdul:
            return None
        dq = np.asarray(hdul["DQ"].data, dtype=np.uint16)
    if dq.shape != shape:
        raise ValueError(f"DQ shape {dq.shape} does not match science data {shape}")
    return dq

