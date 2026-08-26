"""Mask-normalized bilinear translation; invalid detector values never enter sums."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from astr_ir.dq import read_dq, DQ_DO_NOT_USE


def science_valid(path, image, detector_mask):
    valid = np.isfinite(image) & ~np.asarray(detector_mask, bool)
    dq = read_dq(path, image.shape)
    if dq is not None:
        valid &= (dq & DQ_DO_NOT_USE) == 0
    return valid


def masked_gaussian(image, valid, sigma):
    valid = np.asarray(valid, bool) & np.isfinite(image)
    numerator = gaussian_filter(np.where(valid, image, 0.0), sigma, mode="reflect")
    weight = gaussian_filter(valid.astype(float), sigma, mode="reflect")
    return np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 0.05)


def masked_shift(image, valid, offset, *, variance=None, min_support=0.5):
    """Return image, valid mask, fractional support and propagated diagonal variance.

    Output[y,x] samples input[y-dy,x-dx]. The variance uses squared normalized
    interpolation weights. Neighbouring output pixels remain correlated; this
    diagonal map does NOT describe that covariance. No extrapolation at edges.
    """
    image = np.asarray(image, dtype=float)
    valid = np.asarray(valid, bool) & np.isfinite(image)
    if image.ndim != 2 or valid.shape != image.shape:
        raise ValueError("Expected equal-shaped 2-D image and validity mask")
    if not 0 < min_support <= 1 or not np.all(np.isfinite(offset)):
        raise ValueError("Invalid support threshold or translation")
    var = None if variance is None else np.broadcast_to(np.asarray(variance, float), image.shape)
    if var is not None:
        valid &= np.isfinite(var) & (var >= 0)
    h, w = image.shape
    sy = np.arange(h, dtype=float) - float(offset[0])
    sx = np.arange(w, dtype=float) - float(offset[1])
    iy, ix = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = sy - iy, sx - ix
    numerator = np.zeros_like(image)
    support = np.zeros_like(image)
    varnum = np.zeros_like(image) if var is not None else None
    for oy, wy in ((0, 1-fy), (1, fy)):
        for ox, wx in ((0, 1-fx), (1, fx)):
            jy, jx = iy+oy, ix+ox
            inside = ((jy >= 0) & (jy < h))[:, None] & ((jx >= 0) & (jx < w))[None, :]
            index = np.ix_(np.clip(jy, 0, h-1), np.clip(jx, 0, w-1))
            ok = inside & valid[index]
            weight = wy[:, None] * wx[None, :] * ok
            numerator += weight * np.where(ok, image[index], 0)
            support += weight
            if var is not None:
                varnum += weight**2 * np.where(ok, var[index], 0)
    full_domain = ((sy >= 0) & (sy <= h-1))[:, None] & ((sx >= 0) & (sx <= w-1))[None, :]
    good = (support >= min_support) & full_domain
    result = np.divide(numerator, support, out=np.full_like(image, np.nan), where=good)
    outvar = None if var is None else np.divide(varnum, support**2, out=np.full_like(image, np.nan), where=good)
    return result.astype(np.float32), good, support.astype(np.float32), outvar
