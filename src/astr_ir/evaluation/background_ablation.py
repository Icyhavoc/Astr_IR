"""Frozen, paired background experiments; catalog coordinates never enter here.

Selection and final testing are separate commands/seeds. All branches use the
same exposures, registration, injection positions and detector implementation.
Source-free simulations constrain a *model-conditional* false-peak budget, not
the false-positive rate of a real sky with unmodelled instrumental structure.
"""
from dataclasses import asdict, replace
from pathlib import Path
import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from astropy.io import fits

from astr_ir.background.processor import BackgroundConfig, subtract_background
from astr_ir.background.sequence import subtract_sequence
from astr_ir.flicker.processor import correct_flicker, load_detector_mask, load_fits
from astr_ir.registration import science_valid, masked_shift
from .blind_joint import inspect_array, register_features
from .weak_detection import DetectionConfig, Exposure, estimate_psf, analyze_exposures, _fit_group
from .stage_recovery import (RecoveryConfig, inject_native, sample_truth, _stage_exposures,
                             _match_unique, select_threshold_from_null)

BACKGROUND_METHODS = {
    'old64_single': (64, 5, False),
    'new32_single': (32, 1, False),
    'new32_two': (32, 1, True),
}

_FRAME_POOL = None


def _frame_map(function, arguments):
    """Independent-frame CPU parallelism only; preserve input/output ordering."""
    global _FRAME_POOL
    workers = int(os.environ.get('ASTR_IR_FRAME_WORKERS', '1'))
    if workers <= 1:
        return list(map(function, arguments))
    if _FRAME_POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        _FRAME_POOL = ProcessPoolExecutor(max_workers=workers)
    return list(_FRAME_POOL.map(function, arguments))


def _flicker_frame(arguments):
    image, mask = arguments
    return np.where(mask, correct_flicker(image, detector_mask=~mask).corrected, np.nan).astype(np.float32)


def _background_frame(arguments):
    image, mask, config = arguments
    return np.where(mask, subtract_background(image, detector_mask=~mask,
                    config=config).background_subtracted, np.nan).astype(np.float32)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_field(root, manifest, sequence, phase, limit):
    table = pd.read_csv(manifest, usecols=['sequence', 'filename', 'frame_index', 'split'],
                        dtype={'sequence': str})
    rows = table.loc[table.sequence.eq(sequence) & table.split.eq(phase)].sort_values('frame_index')
    if len(rows) < limit or limit < 8:
        raise ValueError('At least eight explicitly frozen frames required')
    rows = rows.iloc[:limit]
    detector = load_detector_mask(Path(root)/'盲点表')
    files, images, masks = [], [], []
    for row in rows.itertuples():
        if Path(row.filename).name != row.filename or Path(sequence).name != sequence:
            raise ValueError('Unsafe input name')
        path = Path(root)/sequence/row.filename
        image, _ = load_fits(path)
        files.append(path)
        images.append(image.astype(np.float32))
        masks.append(science_valid(path, image, detector))
    images, masks = np.stack(images), np.stack(masks)
    flicker = flicker_stack(images, masks)
    reference = inspect_array(flicker[0], masks[0])
    offsets, psfs, registration = [], [], []
    detection = DetectionConfig()
    for image, valid in zip(flicker, masks):
        inspected = inspect_array(image, valid)
        offset, _, diagnostic = register_features(reference[3], reference[4], inspected[3], inspected[4])
        offsets.append(offset)
        registration.append(diagnostic)
        psfs.append(estimate_psf(image, valid, inspected[4], detection)[0])
    return images, masks, np.asarray(offsets), psfs, reference[4], files, registration


def flicker_stack(raw, valid):
    return np.stack(_frame_map(_flicker_frame, zip(raw, valid)))


def background_stack(flicker, valid, offsets, method):
    box, filt, two = BACKGROUND_METHODS[method]
    config = BackgroundConfig(final_box_size=box, final_filter_size=filt)
    if two:
        return subtract_sequence(flicker, valid, offsets, config)[0]
    return np.stack(_frame_map(_background_frame, ((image, mask, config) for image, mask in zip(flicker, valid))))


def registered_stack(images, masks, offsets):
    aligned, valid = [], []
    for image, mask, offset in zip(images, masks, offsets):
        data, good, _, _ = masked_shift(image, mask, offset)
        aligned.append(data)
        valid.append(good)
    return np.asarray(aligned, np.float32), np.asarray(valid, bool)


def paper_input(images, valid):
    # Exactly the input-control image used by denoise_registered_exposures.
    from astr_ir.asteris.paper_pipeline import paper_sigma_clip, _masked_mean, _restore_clip_residual
    clipped, residual, mask, _ = paper_sigma_clip(images, valid)
    image = _masked_mean(clipped, mask) + _restore_clip_residual(residual)
    good = mask.any(axis=0) & np.isfinite(image)
    return np.where(good, image, np.nan).astype(np.float32), good


def products(raw, valid, offsets, methods, models=None):
    """No injection truth; PSF fitting and processing only see images/masks."""
    flicker = flicker_stack(raw, valid)
    result = {}
    for method in methods:
        processed = background_stack(flicker, valid, offsets, method)
        aligned, masks = registered_stack(processed, valid, offsets)
        image, good = paper_input(aligned, masks)
        result[method+'_input'] = (image[None], good[None], np.zeros((1, 2)))
        if models is not None:
            import torch
            from astr_ir.asteris.paper_pipeline import denoise_registered_exposures
            model, config = models[method]
            with torch.inference_mode():
                control, denoised, good, _ = denoise_registered_exposures(aligned, masks, model, config, device='cuda')
            if not np.allclose(control, image, equal_nan=True):
                raise AssertionError('Paper control differs from model input coadd')
            result[method+'_asteris'] = (denoised[None], good[None], np.zeros((1, 2)))
        del processed, aligned, masks
    return result


def detect(product, threshold=5.):
    config = DetectionConfig(threshold=float(threshold))
    exposures = _stage_exposures(product, config)
    return exposures, analyze_exposures(exposures, config)['sources']


def accepted(table, threshold):
    return table.loc[table.accepted.astype(bool) & (table.snr_empirical >= threshold)].reset_index(drop=True)


def score_trial(points, flux, original, detected, base_exposures, new_exposures, threshold):
    original, detected = accepted(original, threshold), accepted(detected, threshold)
    xy = points[['x', 'y']].to_numpy()
    found, existing = detected[['x', 'y']].to_numpy(), original[['x', 'y']].to_numpy()
    matches = _match_unique(xy, found, 2.5)
    prior = _match_unique(xy, existing, 2.5)
    associated = _match_unique(existing, found, 2.5)
    used = {j for j, _ in matches.values()} | {j for j, _ in associated.values()}
    delta = [Exposure(b.image-a.image, a.valid & b.valid, a.noise, a.offset, a.throughput, a.psf)
             for a, b in zip(base_exposures, new_exposures)]
    rows = []
    for i, row in enumerate(points.to_dict('records')):
        try:
            response = float(_fit_group(delta, xy[i:i+1])[0][0]/flux)
        except ValueError:
            response = float('nan')
        match = matches.get(i)
        rows.append(dict(**row, injected_flux=flux, recovered=match is not None,
                         preexisting_match=i in prior, paired_flux_response=response,
                         localization_error=match[1] if match else np.nan,
                         measured_flux=float(detected.iloc[match[0]].flux) if match else np.nan))
    summary = dict(injected=len(points), eligible_new=len(points)-len(prior),
                   recovered_new=len(set(matches)-set(prior)), baseline_candidates=len(original),
                   injected_candidates=len(detected), new_unmatched_candidates=len(found)-len(used),
                   median_flux_response=float(np.nanmedian([r['paired_flux_response'] for r in rows])))
    return rows, summary


def null_raw(raw, valid, rng):
    """Known source-free parametric sky: smooth gradient, independent Gaussian
    pixel noise and stochastic row offsets. No real-image cutouts are recycled.
    Noise/level fitted only from the supplied validation exposures. This does
    not model real PSF wings, persistence or all correlated detector artifacts.
    """
    from astr_ir.flicker.processor import robust_std
    from astr_ir.background.processor import neighbor_difference_noise
    yy, xx = np.indices(raw.shape[1:], dtype=np.float32)
    h, w = yy.shape
    output = []
    for image, mask in zip(raw, valid):
        sigma = neighbor_difference_noise(image, ~mask)
        level = float(np.nanmedian(image[mask]))
        # Smooth nuisance sky has the measured robust large-scale amplitude.
        amplitude = min(float(robust_std(image[mask])), 4*sigma)
        smooth = amplitude*(.3*xx/w + .2*yy/h + .15*np.sin(2*np.pi*xx/w))
        rows = rng.normal(0, .1*sigma, (h, 1))
        noise = rng.normal(0, sigma, image.shape)
        output.append(np.where(mask, level+smooth+rows+noise, np.nan).astype(np.float32))
    return np.stack(output)


def run_field(root, manifest, output, sequence, *, phase='validation', limit=8,
              methods=tuple(BACKGROUND_METHODS), seed=2026082601, repeats=2,
              sources=12, fluxes=(3000., 6000., 12000.), null_repeats=4,
              thresholds=None, models=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Fresh experiment required: {output}')
    output.mkdir(parents=True)
    started = time.monotonic()
    def progress(message):
        print(f'{sequence} [{time.monotonic()-started:.0f}s] {message}', flush=True)
        save_json(output/'progress.json', dict(message=message, elapsed=time.monotonic()-started, complete=False))
    progress('Loading frozen '+phase+' frames')
    raw, valid, offsets, psfs, stars, files, registration = load_field(root, manifest, sequence, phase, limit)
    hashes = {str(p): sha256(p) for p in files}
    detection_floor = 5.
    base_products = products(raw, valid, offsets, methods, models)
    null_scores = {name: [] for name in base_products}
    # Calibrate before injection trials; the threshold never sees injection truth.
    if thresholds is None:
        for repeat in range(null_repeats):
            progress(f'Null validation {repeat+1}/{null_repeats}')
            simulated = null_raw(raw, valid, np.random.default_rng(seed+70000+repeat))
            for name, product in products(simulated, valid, offsets, methods, models).items():
                _, table = detect(product, detection_floor)
                table.to_csv(output/f'null_{repeat}_{name}.csv', index=False)
                null_scores[name].append(accepted(table, detection_floor).snr_empirical.tolist())
        thresholds = {}
        for name, scores in null_scores.items():
            threshold, rate = select_threshold_from_null(scores, np.arange(5., 15.5, .5), .5)
            thresholds[name] = threshold
        save_json(output/'thresholds.json', dict(thresholds=thresholds, null_scores=null_scores,
            simulations=null_repeats, max_false_per_image=.5, model_conditional=True))
    base = {}
    for name, product in base_products.items():
        progress('Baseline '+name)
        base[name] = detect(product, detection_floor)
        base[name][1].to_csv(output/f'baseline_{name}.csv', index=False)
    recovery = RecoveryConfig(seed=seed, sources_per_trial=sources, fluxes=tuple(fluxes), repeats=repeats)
    rng = np.random.default_rng(seed)
    rows, summaries = [], []
    for repeat in range(repeats):
        positions = sample_truth(valid, offsets, stars, recovery, rng)
        positions.to_csv(output/f'truth_{repeat}.csv', index=False)
        for flux in fluxes:
            progress(f'Injection {repeat+1}/{repeats}, flux={flux:g}')
            injected = inject_native(raw, valid, offsets, psfs, positions.assign(flux=flux), rng)
            for name, product in products(injected, valid, offsets, methods, models).items():
                progress(f'Fit {name}, repeat={repeat}, flux={flux:g}')
                exposures, table = detect(product, detection_floor)
                table.to_csv(output/f'injected_{repeat}_{flux:g}_{name}.csv', index=False)
                detail, summary = score_trial(positions, flux, base[name][1], table,
                                              base[name][0], exposures, thresholds[name])
                meta = dict(sequence=sequence, phase=phase, method=name, repeat=repeat,
                            flux=flux, threshold=thresholds[name])
                rows.extend(dict(**meta, **row) for row in detail)
                summaries.append(dict(**meta, **summary))
                pd.DataFrame(rows).to_csv(output/'source_recovery.csv', index=False)
                pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False)
                del exposures, table
            gc.collect()
    for path in files:
        if sha256(path) != hashes[str(path)]:
            raise RuntimeError('Raw input changed during experiment')
    save_json(output/'report.json', dict(sequence=sequence, phase=phase, limit=limit, seed=seed,
        repeats=repeats, sources=sources, fluxes=list(fluxes), input_sha256=hashes,
        thresholds=thresholds, methods=list(methods), offsets_yx=offsets.tolist(),
        registration=registration, catalog_used=False, deterministic_injection=True,
        model_evaluated=models is not None, elapsed=time.monotonic()-started,
        caveats=['Null rate is conditional on simplified source-free Gaussian + row-noise simulations.',
                 'Unmatched real-sky candidates are not confirmed false detections.',
                 'Flux levels reuse positions; do not count them as independent position samples.',
                 'All detection uses paper input or output coadds, not native joint exposure detection.']))
    save_json(output/'progress.json', dict(message='Complete', complete=True, elapsed=time.monotonic()-started))
    return pd.DataFrame(rows), pd.DataFrame(summaries)
