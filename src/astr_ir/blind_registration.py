"""Image-only blind registration shared by science stages and evaluation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from scipy.fft import rfftn, irfftn

from astr_ir.background.processor import neighbor_difference_noise
from astr_ir.flicker.processor import robust_std, make_edge_mask
from astr_ir.registration import load_science, masked_gaussian

def fit_gaussian_pixels(cut, mask):
    """Small scalar-sum Gaussian fit, safe alongside torch's Windows OpenMP.

    Amplitude and background are solved analytically for each trial center/width;
    bounded coordinate search avoids loading a second BLAS/OpenMP runtime.
    """
    radius = cut.shape[0]//2
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    x,y,z=xx[mask],yy[mask],cut[mask]
    zm=float(np.mean(z))
    def trial(position):
        dx,dy,width=position
        g=np.exp(-((x-dx)**2+(y-dy)**2)/(2*width**2))
        gm=float(np.mean(g))
        a=max(0,float(np.sum((g-gm)*(z-zm))/max(np.sum((g-gm)**2),1e-20)))
        b=zm-a*gm
        residual=a*g+b-z
        return float(np.sum(residual**2)),a,b,residual
    position=np.array([0.,0.,2.])
    best=trial(position)
    for step in (1.,.5,.25,.125,.0625,.03125,.015625):
        for _ in range(2):
            for axis in range(3):
                for direction in (-1,1):
                    candidate=position.copy()
                    candidate[axis]+=direction*step
                    candidate=np.clip(candidate,[-2,-2,.7],[2,2,5])
                    fit=trial(candidate)
                    if fit[0]<best[0]:
                        best,position=fit,candidate
    return np.array([best[1],position[0],position[1],position[2],best[2]]),best[3]


def inspect_frame(path, detector):
    image, header, valid, _ = load_science(path, detector)
    return inspect_array(image, valid, header)


def inspect_array(image, valid, header=None):
    """The same image-only feature extraction for read-only in-memory trials."""
    image=np.asarray(image,float)
    valid = np.asarray(valid,bool) & np.isfinite(image) & ~make_edge_mask(image.shape, 24)
    smooth = masked_gaussian(image, valid, 16)
    highpass = np.where(valid, image - smooth, 0)
    feature = masked_gaussian(highpass, valid, 1.3)
    scale = robust_std(feature[valid])
    peaks = valid & (feature > 7 * scale) & (feature == maximum_filter(feature, 15))
    candidates = np.argwhere(peaks)
    candidates = sorted(candidates, key=lambda p: feature[tuple(p)], reverse=True)[:60]
    stars = []
    radius = 8
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    for y, x in candidates:
        cut = image[y-radius:y+radius+1, x-radius:x+radius+1]
        mask = valid[y-radius:y+radius+1, x-radius:x+radius+1]
        if cut.shape != xx.shape or mask.mean() < 0.95:
            continue
        center = float(np.median(cut[mask]))
        amplitude = float(cut[mask].max()-center)
        if amplitude <= 0:
            continue
        parameters, residual = fit_gaussian_pixels(cut,mask)
        a, dx, dy, width, b = parameters
        noise = robust_std(residual)
        fit_snr = a*np.sqrt(np.pi)*width/max(noise, 1e-12)
        if 0.8 < width < 4.8 and a > 3*noise and fit_snr >= 12:
            stars.append((x+dx, y+dy, width, 2*np.pi*a*width**2))
    stars = np.asarray(stars, float).reshape(-1,4)
    # Suppress detector pattern and random noise in the registration features.
    feature = np.where(valid, np.clip(feature-3*scale, 0, 40*scale), 0)
    feature *= np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    sigma = neighbor_difference_noise(image, ~valid)
    if not np.isfinite(sigma) or sigma <= 0 or np.count_nonzero(feature) < 20:
        raise ValueError("Insufficient blind registration/noise information")
    return image, valid, header, feature, stars, sigma


def feature_translation(reference, moving):
    """FFT correlation + parabolic peak, avoiding threaded BLAS in torch processes."""
    correlation = irfftn(rfftn(reference)*np.conj(rfftn(moving)), s=reference.shape)
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    offset = np.asarray(peak, float)
    for axis, length in enumerate(correlation.shape):
        before, after = list(peak), list(peak)
        before[axis] = (before[axis]-1)%length
        after[axis] = (after[axis]+1)%length
        left, center, right = correlation[tuple(before)], correlation[peak], correlation[tuple(after)]
        curvature = left-2*center+right
        delta = 0.5*(left-right)/curvature if curvature < 0 else 0
        if offset[axis] > length//2:
            offset[axis] -= length
        offset[axis] += np.clip(delta,-0.5,0.5)
    norm = np.sum(reference**2)*np.sum(moving**2)
    error = np.sqrt(max(0,1-correlation[peak]**2/max(norm,1e-30)))
    return offset, error


def register_features(reference_feature, reference_stars, feature, stars):
    offset, error = feature_translation(reference_feature, feature)
    if not np.all(np.isfinite(offset)) or np.max(np.abs(offset)) > 64:
        raise ValueError(f"Unreliable blind translation: {offset}")
    pairs = np.empty((0,2), int)
    scatter = np.nan
    if len(reference_stars) and len(stars):
        distance, index = cKDTree(reference_stars[:,:2]).query(stars[:,:2] + offset[::-1])
        candidates = np.flatnonzero(distance < 3)
        # Require unique matches; no known source is selected as an anchor.
        used, pairs_list = set(), []
        for k in sorted(candidates, key=lambda k: distance[k]):
            if int(index[k]) not in used:
                pairs_list.append((k, index[k]))
                used.add(int(index[k]))
        pairs = np.asarray(pairs_list, int).reshape(-1,2)
        if len(pairs) >= 3:
            delta = reference_stars[pairs[:,1],:2] - stars[pairs[:,0],:2]
            center = np.median(delta, axis=0)
            good = np.linalg.norm(delta-center, axis=1) < 1.0
            pairs = pairs[good]
            if len(pairs) >= 3:
                delta = delta[good]
                offset = np.median(delta, axis=0)[::-1]
                scatter = float(np.sqrt(np.mean((delta-offset[::-1])**2)))
    if len(pairs) < 3 and (not np.isfinite(error) or error > 0.8):
        raise ValueError(f"Blind registration failed: matches={len(pairs)}, phase error={error}")
    throughput = 1.0
    if len(pairs) >= 3:
        ratios = stars[pairs[:,0],3] / reference_stars[pairs[:,1],3]
        throughput = float(np.median(ratios))
        if not 0.2 <= throughput <= 5:
            raise ValueError(f"Implausible transparency ratio {throughput}")
    return offset, throughput, dict(matched_stars=len(pairs), registration_rms=scatter, phase_error=float(error), transparency_fallback=len(pairs)<3)
