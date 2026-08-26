"""Catalog-free, native-pixel multi-exposure matched-filter detection.

Implements the statistic of Zackay & Ofek, arXiv:1512.06872, eqs. 27/29,
with spatially varying DQ support. A Gaussian PSF and relative transparency
are estimated from automatically detected stars. This is a detection baseline,
not Paper II's Fourier proper coadd, nor a claim of calibrated tail probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.fft import rfftn, irfftn

from astr_ir.background.processor import neighbor_difference_noise
from astr_ir.flicker.processor import load_fits, load_detector_mask, robust_std, make_edge_mask
from astr_ir.registration import masked_gaussian, masked_shift, science_valid
from astr_ir.dq import build_dq


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
    image, header = load_fits(path)
    return inspect_array(image,science_valid(path,image,detector),header)


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


def gaussian_psf(sigma, fraction=(0,0)):
    radius = int(np.ceil(5*sigma))+1
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-((yy+fraction[0])**2+(xx+fraction[1])**2)/(2*sigma**2))
    return kernel/kernel.sum()


def exposure_statistic(image, valid, noise, psf_sigma, offset=(0,0), throughput=1, *, fit_local_background=True, psf_kernel=None):
    """Evaluate a shifted PSF on native pixels, without resampling noisy data.

    Numerator=sum(F*P*I/V); information=sum(F²*P²/V). By default a
    local constant background is marginalized using the weighted normal
    equations for [PSF, constant]; its variance cost is included. Fractional
    translation is in the PSF kernel, followed by an INTEGER map translation.
    Thus no bilinear noise covariance is omitted from the detection variance.
    Preprocessing-induced/temporal correlations still require empirical checks.
    """
    integer = np.floor(offset).astype(int)
    fraction=np.asarray(offset)-integer
    if psf_kernel is None:
        kernel = gaussian_psf(psf_sigma, fraction)
    else:
        from .weak_detection import shifted_kernel
        kernel = shifted_kernel(psf_kernel,-fraction[0],-fraction[1])
    noise=np.asarray(noise,float)
    if not np.isfinite(noise).all() or np.any(noise<=0):
        raise ValueError('Noise must be positive and finite')
    valid=np.asarray(valid,bool) & np.isfinite(image)
    center = np.median(image[valid])
    ivar = valid.astype(float)/noise**2
    n = throughput*fftconvolve(np.where(valid, image-center, 0)*ivar, kernel[::-1,::-1], mode="same")
    d = throughput**2*fftconvolve(ivar, (kernel**2)[::-1,::-1], mode="same")
    if fit_local_background:
        window=np.ones_like(kernel)
        c=fftconvolve(ivar,window,mode="same")
        b=fftconvolve(ivar,kernel[::-1,::-1],mode="same")
        total=fftconvolve(np.where(valid,image-center,0)*ivar,window,mode="same")
        n-=throughput*np.divide(b*total,c,out=np.zeros_like(n),where=c>0)
        d-=throughput**2*np.divide(b*b,c,out=np.zeros_like(d),where=c>0)
    support = fftconvolve(valid.astype(float), kernel[::-1,::-1], mode="same")
    ok = (support >= 0.8) & (d > 0)
    n, good, _, _ = masked_shift(n, ok, integer, min_support=1)
    d, _, _, _ = masked_shift(d, ok, integer, min_support=1)
    return np.where(good,n,0).astype(float), np.where(good,d,0).astype(float), good


def empirical_score(numerator, information, valid):
    nominal = np.divide(numerator, np.sqrt(information), out=np.full_like(numerator,np.nan), where=valid & (information>0))
    # Catalog-free robust central-distribution calibration; not a 5-sigma FAP claim.
    values = nominal[valid & np.isfinite(nominal)]
    center = float(np.median(values))
    scale = robust_std(values)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot calibrate empty/degenerate detection map")
    return (nominal-center)/scale, nominal, center, scale


def detect_blind(score, valid, threshold=5.0, min_distance=5):
    work = np.where(valid & np.isfinite(score), score, -np.inf)
    peaks = (work >= threshold) & (work == maximum_filter(work, 2*min_distance+1))
    positions = np.argwhere(peaks)
    positions = sorted(positions, key=lambda p: work[tuple(p)], reverse=True)
    return pd.DataFrame([dict(x=int(x), y=int(y), snr_empirical=float(work[y,x])) for y,x in positions], columns=["x","y","snr_empirical"])


def wait_for_upstream(progress_path, filename, *, timeout=1800):
    """Wait for a completed-file audit row, never consume an old in-place FITS."""
    start=time.monotonic()
    while True:
        try:
            table=pd.read_csv(progress_path)
            input_name='flicker_corrected_'+filename.removeprefix('background_subtracted_')
            rows=table.loc[table['input_filename']==input_name]
            if len(rows)==1 and float(rows.iloc[0]['equation_max_abs_error_float32'])==0:
                return
        except (FileNotFoundError,pd.errors.EmptyDataError,pd.errors.ParserError,KeyError,ValueError):
            pass  # Writer may currently be replacing its progress CSV.
        if time.monotonic()-start>=timeout:
            raise TimeoutError(f"No completed upstream product for {filename} in {progress_path}")
        time.sleep(5)


def run_sequence(input_root, dataset_root, output_root, sequence, *, limit=None, progress_path=None):
    input_root, dataset_root, output_root = map(Path, (input_root,dataset_root,output_root))
    files = sorted((input_root/sequence).glob("background_subtracted_*.fits"))
    if limit is not None:
        files = files[:limit]
    if len(files) < 2:
        raise ValueError("Joint detection requires >=2 frames")
    destination = output_root/sequence
    destination.mkdir(parents=True, exist_ok=True)
    detector = load_detector_mask(dataset_root/"盲点表")
    if progress_path is not None:
        wait_for_upstream(progress_path,files[0].name)
    reference = inspect_frame(files[0],detector)
    shape = reference[0].shape
    numerator = np.zeros((2,*shape),float)
    information = np.zeros_like(numerator)
    coverage = np.zeros((2,*shape),np.uint16)
    mean_num, mean_den = np.zeros(shape), np.zeros(shape)
    mean_count = np.zeros(shape,np.uint16)
    diagnostics = []
    for index,path in enumerate(files):
        if progress_path is not None:
            wait_for_upstream(progress_path,path.name)
        image,valid,header,feature,stars,noise = reference if index==0 else inspect_frame(path,detector)
        offset,throughput,diag = register_features(reference[3],reference[4],feature,stars)
        sigma = float(np.median(stars[:,2])) if len(stars)>=3 else 2.5
        n,d,ok = exposure_statistic(image,valid,noise,sigma,offset,throughput)
        half = index%2
        numerator[half] += n
        information[half] += d
        coverage[half] += ok
        aligned,good,_,var = masked_shift(image-np.median(image[valid]),valid,offset,variance=noise**2)
        ivar = np.divide(throughput**2,var,out=np.zeros(shape),where=good & (var>0))
        mean_num += np.where(good,aligned/throughput,0)*ivar
        mean_den += ivar
        mean_count += good
        diagnostics.append(dict(filename=path.name, frame_index=index, alignment_dy=offset[0], alignment_dx=offset[1], noise=noise, psf_sigma=sigma, psf_stars=len(stars), psf_fallback=len(stars)<3, throughput=throughput, **diag))
        if (index+1)%10==0:
            print(f"blind detection {sequence}: {index+1}/{len(files)}",flush=True)
    n,d = numerator.sum(axis=0),information.sum(axis=0)
    count = coverage.sum(axis=0)
    good = (count>=np.ceil(0.8*len(files))) & (d>0)
    score,nominal,center,scale = empirical_score(n,d,good)
    flux = np.divide(n,d,out=np.full(shape,np.nan),where=good)
    mean = np.divide(mean_num,mean_den,out=np.full(shape,np.nan),where=mean_den>0)
    halves = [empirical_score(numerator[k], information[k], coverage[k]>=np.ceil(0.8*len(files[k::2])))[0] for k in (0,1)]
    half_flux = np.divide(numerator,information,out=np.full_like(numerator,np.nan),where=information>0)
    null_variance = np.divide(1,information,out=np.full_like(information,np.inf),where=information>0).sum(axis=0)
    null = (half_flux[0]-half_flux[1])/np.sqrt(null_variance)
    sources = detect_blind(score,good)
    negative = detect_blind(-score,good)
    for name,array in (("snr_even",halves[0]),("snr_odd",halves[1]),("flux",flux), ("coverage",count)):
        sources[name] = [float(array[int(row.y),int(row.x)]) for row in sources.itertuples()]
    sources["both_halves_ge3"] = (sources.snr_even>=3) & (sources.snr_odd>=3)
    sources.to_csv(destination/"blind_sources.csv",index=False,encoding="utf-8-sig")
    negative.to_csv(destination/"negative_peaks.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(destination/"frame_diagnostics.csv",index=False,encoding="utf-8-sig")
    summary = dict(sequence=sequence, frames=len(files), detections_ge5=len(sources), detections_both_halves_ge3=int(sources.both_halves_ge3.sum()), negative_peaks_ge5=len(negative), nominal_center=center, empirical_noise_scale=scale, shared_structure_warning=bool(scale>2), null_robust_std=robust_std(null[good]), catalog_used=False, coordinate_system="zero-based pixels of FIRST exposure; not legacy ASTERIS coadd WCS", method="Zackay-Ofek I native-pixel Gaussian PSF statistic + local background nuisance", significance="central-distribution MAD calibration, NOT calibrated false-alarm probability", psf_fallback_frames=sum(r["psf_fallback"] for r in diagnostics), transparency_fallback_frames=sum(r["transparency_fallback"] for r in diagnostics))
    (destination/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    base_header = reference[2].copy()
    base_header["NCOMBINE"] = len(files)
    base_header["CATUSED"] = False
    base_header["REFIMAGE"] = files[0].name
    base_header["HIERARCH JOINT METHOD"] = "Zackay-Ofek I approx, native pixel PSF"
    base_header["HIERARCH JOINT NOISESCL"] = scale
    base_header["HIERARCH JOINT LOCALBKG"] = True
    base_header.add_history("No catalog or known-target coordinates enter processing.")
    base_header.add_history("FIRST exposure pixel reference. No inherited catalog-overlay WCS.")
    for name,array in (("joint_score",score),("joint_nominal_score",nominal),("joint_flux",flux),("weighted_coadd",mean),("odd_even_null",null)):
        mask = np.isfinite(array) & (good if name!="weighted_coadd" else (mean_count>0))
        product_count=mean_count if name=="weighted_coadd" else count
        dq = build_dq(shape,no_coverage=~mask,partial_coverage=mask & (product_count<len(files)))
        header = base_header.copy()
        header["BUNIT"] = "DN" if name in {"joint_flux","weighted_coadd"} else "dimensionless"
        fits.HDUList([fits.PrimaryHDU(np.where(mask,array,np.nan).astype(np.float32),header), fits.ImageHDU(dq,name="DQ"), fits.ImageHDU(product_count.astype(np.uint16),name="COVERAGE"), fits.ImageHDU(d.astype(np.float32),name="INFORMATION"), fits.ImageHDU(mean_count,name="MEAN_COVERAGE"),fits.ImageHDU(mean_den.astype(np.float32),name="MEAN_WEIGHT")]).writeto(destination/f"{name}_{sequence}.fits",overwrite=True,output_verify="silentfix")
    regions = ["# Region file format: DS9 version 4.1", "image"]
    regions += [f"circle({r.x+1},{r.y+1},6) # color=green text={{{r.snr_empirical:.1f}}}" for r in sources.itertuples()]
    (destination/"blind_sources.reg").write_text("\n".join(regions)+"\n",encoding="utf-8")
    return summary
