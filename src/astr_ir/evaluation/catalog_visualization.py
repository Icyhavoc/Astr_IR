"""Offline, display-only overlays of the previously frozen real-source catalog.

Never queried by preprocessing, training or detection. Only PNG/CSV/JSON under
figures are written; science FITS and catalog caches remain read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy import ndimage

from .blind_joint import inspect_frame, register_features


def read_display_image(path):
    """Mask invalid science pixels without filling or modifying the FITS."""
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=float).copy()
        header = hdul[0].header.copy()
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D science image: {path}")
        valid = np.isfinite(image)
        if 'DQ' in hdul:
            if hdul['DQ'].data.shape != image.shape:
                raise ValueError(f"DQ shape mismatch: {path}")
            valid &= (hdul['DQ'].data & 1) == 0
        image[~valid] = np.nan
    if not valid.any():
        raise ValueError(f"No valid pixels: {path}")
    return image, header


def transfer_positions(table, offset_yx, image):
    """Translate catalog coordinates, never snap them to a detected peak.

offset_yx shifts the old reference image INTO the new reference grid, hence
x_new=x_old+dx and y_new=y_old+dy. All pixel coordinates are zero-based.
"""
    offset = np.asarray(offset_yx, float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('Expected finite (dy, dx)')
    result = table.copy()
    result['x_plot'] = result.x_pix_0based.astype(float) + offset[1]
    result['y_plot'] = result.y_pix_0based.astype(float) + offset[0]
    x, y = result.x_plot.to_numpy(), result.y_plot.to_numpy()
    h, w = image.shape
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    good = np.zeros(len(result), bool)
    indices = np.flatnonzero(inside)
    good[indices] = np.isfinite(image[np.rint(y[indices]).astype(int), np.rint(x[indices]).astype(int)])
    result['in_frame'] = inside
    result['valid_at_position'] = good
    return result


def estimate_display_translation(new_reference, old_reference):
    """Image-only registration; catalog rows are deliberately not arguments."""
    shape = fits.getdata(new_reference).shape
    if fits.getdata(old_reference).shape != shape:
        raise ValueError('Reference image dimensions differ')
    # These are already registered coadds: use their DQ, not native blindmap again.
    detector = np.zeros(shape, bool)
    new = inspect_frame(new_reference, detector)
    old = inspect_frame(old_reference, detector)
    offset, _, diagnostic = register_features(new[3], new[4], old[3], old[4])
    rms = diagnostic['registration_rms']
    if diagnostic['matched_stars'] < 6 or not np.isfinite(rms) or rms > 0.75:
        raise ValueError(f'Insufficient astrometric transfer for catalog display: {diagnostic}')
    diagnostic['registration_rms_2d_pix'] = float(np.sqrt(2)*rms)
    diagnostic['dy_old_to_current'] = float(offset[0])
    diagnostic['dx_old_to_current'] = float(offset[1])
    return offset, diagnostic


def _normalization(image, score=False):
    if score:
        return plt.Normalize(vmin=-2, vmax=8)
    values = image[np.isfinite(image)]
    lo, hi = np.percentile(values, (1, 99.7)) if values.size else (0, 1)
    return PowerNorm(gamma=0.5, vmin=float(lo), vmax=float(max(hi, lo+1e-6)), clip=True)


def _display_cmap():
    """Opaque neutral gray for missing data, independent of the figure background."""
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('#808080', alpha=1.0)
    return cmap


def interpolate_display_holes(image, *, max_hole_pixels=16):
    """Return a cosmetic COPY and fill mask; never infer values from the catalog.

Only enclosed, 8-connected invalid components of at most max_hole_pixels are
eligible. Values come from Gaussian-weighted ORIGINAL valid neighbors within
three pixels (sigma=1); filled pixels never become support for another fill.
Valid science pixels, large gaps and components touching an edge are unchanged.
The result is not suitable for measurement, detection or model input.
"""
    if not isinstance(max_hole_pixels, (int, np.integer)) or max_hole_pixels < 1:
        raise ValueError('max_hole_pixels must be a positive integer')
    original = np.asarray(image, dtype=float)
    if original.ndim != 2:
        raise ValueError('Expected a 2-D display image')
    result = original.copy()
    valid = np.isfinite(original)
    result[~valid] = np.nan
    labels, count = ndimage.label(~valid, structure=np.ones((3,3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    eligible = (sizes > 0) & (sizes <= max_hole_pixels)
    eligible[0] = False
    boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))
    eligible[boundary] = False
    fill = eligible[labels]
    if fill.any():
        support = ndimage.gaussian_filter(valid.astype(float), sigma=1, radius=3, mode='constant', cval=0)
        numerator = ndimage.gaussian_filter(np.where(valid,original,0), sigma=1, radius=3, mode='constant', cval=0)
        fill &= support > 1e-8
        result[fill] = numerator[fill]/support[fill]
    return result, fill


def _mark(ax, row, radius, label=False):
    x, y = row.x_plot, row.y_plot
    color = '#00e5ff' if row.valid_at_position else '#ffad42'
    ax.add_patch(Circle((x,y), radius, fill=False, color=color, linewidth=1.1))
    if label:
        ax.annotate(row.weak_label, (x,y), xytext=(7,7), textcoords='offset points',
                    color=color, fontsize=9, weight='bold', clip_on=True)


def _overview(image, positions, title, path, *, score, dpi, display_image=None):
    interpolated = display_image is not None
    shown = image if display_image is None else display_image
    fig, ax = plt.subplots(figsize=(9,8.7), layout='constrained')
    artist = ax.imshow(np.ma.masked_invalid(shown), origin='lower', cmap=_display_cmap(),
                       interpolation='nearest', norm=_normalization(image,score))
    for row in positions.loc[positions.in_frame].itertuples():
        _mark(ax,row,9,label=True)
    mode = 'DISPLAY-ONLY INTERPOLATED' if interpolated else 'MASKED: invalid pixels gray'
    ax.set(title=f'{title}\n{mode}', xlabel='x [pixel, zero-based]', ylabel='y [pixel, zero-based]')
    fig.colorbar(artist, ax=ax, fraction=0.04, label='Empirical score (not calibrated sigma)' if score else 'DN (sqrt display)')
    note = ('Small holes filled for display, NOT measurements; remaining gaps gray.' if interpolated
            else 'Gray = DQ DO_NOT_USE / non-finite; no detector-mask overlay or interpolation.')
    fig.supxlabel('Catalog positions, NOT detections | cyan/orange: ORIGINAL pixel valid/invalid\n'
                  f'{note}\nStretch from original valid data; no peak snapping or source protection.', fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _cutout_page(products, positions, labels, sequence, path, *, half, dpi, interpolated=False):
    fig, axes = plt.subplots(len(labels),len(products), figsize=(12,2.65*len(labels)+0.8), squeeze=False, layout='constrained')
    for column, (key, product) in enumerate(products.items()):
        table = positions[key].set_index('weak_label')
        image = product['image']
        shown = product['display_image'] if interpolated else image
        h,w = image.shape
        for index,label in enumerate(labels):
            ax = axes[index,column]
            row = table.loc[label]
            if not row.in_frame:
                ax.text(0.5,0.5,'Outside image',ha='center',transform=ax.transAxes)
            else:
                x,y = int(round(row.x_plot)), int(round(row.y_plot))
                xlo,xhi,ylo,yhi = max(0,x-half),min(w,x+half+1),max(0,y-half),min(h,y+half+1)
                cut = image[ylo:yhi,xlo:xhi]
                ax.imshow(np.ma.masked_invalid(shown[ylo:yhi,xlo:xhi]), origin='lower',
                          cmap=_display_cmap(), interpolation='nearest',
                          extent=(xlo-.5,xhi-.5,ylo-.5,yhi-.5), norm=_normalization(cut,product['score']))
                _mark(ax,row,6)
                # Keep exact subpixel predicted coordinate, not the rounded crop center.
                ax.set(xlim=(xlo-.5,xhi-.5),ylim=(ylo-.5,yhi-.5))
            ax.set_xticks([]); ax.set_yticks([])
            if index == 0:
                ax.set_title(product['short_title'],fontsize=10)
            if column == 0:
                ax.set_ylabel(f'{label}\nK={row.k_m:.2f}',fontsize=10)
    mode = 'DISPLAY-ONLY INTERPOLATED' if interpolated else 'MASKED: invalid pixels gray'
    fig.suptitle(f'{sequence} | real catalog positions: {labels[0]} - {labels[-1]}\n{mode}',fontsize=12)
    fig.supxlabel('Fixed catalog sample; not a completeness test. Unequal exposures / preprocessing.\n'
                  'Original-valid-data stretches; cyan/orange: ORIGINAL pixel valid/invalid.\n'
                  + ('Filled holes are cosmetic, NOT detections or measurements; remaining gaps gray.' if interpolated
                     else 'Gray = invalid data; compare locations, not brightness or SNR.'),fontsize=8)
    fig.savefig(path,dpi=dpi)
    plt.close(fig)


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _render_worker(payload):
    """Draw in a clean CPU process; avoid torch/Conda OpenMP collisions."""
    destination = Path(payload['destination'])
    products,positions,mask_statistics = {},{},{}
    for key,product in payload['products'].items():
        image,_ = read_display_image(product['path'])
        products[key] = dict(product,image=image)
        positions[key] = pd.DataFrame(payload['positions'][key])
        filled_count = 0
        if payload['export_interpolated']:
            display_image,fill = interpolate_display_holes(image,max_hole_pixels=payload['max_hole_pixels'])
            products[key]['display_image'] = display_image
            filled_count = int(fill.sum())
        with fits.open(product['path'],memmap=False) as hdul:
            has_dq = 'DQ' in hdul
        invalid_count = int((~np.isfinite(image)).sum())
        mask_statistics[key] = dict(has_dq=has_dq,invalid_pixels=invalid_count,
                                   display_filled_pixels=filled_count,remaining_invalid_pixels=invalid_count-filled_count)
    variants = {}
    for mode in ('masked','display_interpolated') if payload['export_interpolated'] else ('masked',):
        interpolated = mode == 'display_interpolated'
        folder = destination/mode if interpolated else destination
        folder.mkdir(parents=True,exist_ok=True)
        paths_out,cutouts = {},[]
        for key,product in products.items():
            path = folder/f"{payload['sequence']}_{key}_catalog_overlay.png"
            _overview(product['image'],positions[key],f"{payload['sequence']} | {product['short_title']}",path,
                      score=product['score'],dpi=payload['dpi'],
                      display_image=product.get('display_image') if interpolated else None)
            paths_out[key] = str(path)
        labels = payload['labels']
        for start in range(0,len(labels),4):
            path = folder/f"{payload['sequence']}_weak_source_cutouts_{start//4+1:02d}.png"
            _cutout_page(products,positions,labels[start:start+4],payload['sequence'],path,
                         half=payload['half'],dpi=payload['dpi'],interpolated=interpolated)
            cutouts.append(str(path))
        variants[mode] = dict(overlays=paths_out,cutouts=cutouts)
    return dict(variants=variants,mask_statistics=mask_statistics)


def export_catalog_validation(project_root, *, sequences=('90000002','90000003'),
                              output_dir=None, dpi=160, cutout_half_size=24,
                              export_interpolated=True, max_hole_pixels=16):
    """Export frozen-catalog overlays, cutout pages and provenance; offline only.

Repeated calls replace only this exporter’s files under figures. A missing or
stale calibration fails explicitly; it never queries catalogs or trains a model.
Gray-masked figures are always exported. Optional cosmetic copies in the
display_interpolated/ subfolder are labeled and NEVER used for validity flags,
stretch estimates, registration or any science calculation.
"""
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else root/'figures/catalog_validation_output'
    if not output.is_relative_to(root/'figures'):
        raise ValueError('Visualization output must be inside project figures/')
    if dpi < 50 or cutout_half_size < 8:
        raise ValueError('dpi >= 50 and cutout_half_size >= 8 required')
    if not isinstance(max_hole_pixels, (int, np.integer)) or max_hole_pixels < 1:
        raise ValueError('max_hole_pixels must be a positive integer')
    processed = root/'data/processed'
    records = []
    for sequence in sequences:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',str(sequence)):
            raise ValueError('Invalid sequence identifier')
        catalog_root = processed/'evaluation/asteris_paper_catalog'/sequence
        table_path = catalog_root/'weak_sources.csv'
        if not table_path.is_file():
            raise FileNotFoundError(f'No previously calibrated catalog for {sequence}: {table_path}')
        table = pd.read_csv(table_path,dtype={'designation':str,'gaia_source_id':str,'weak_label':str})
        if table.empty or table.weak_label.isna().any() or table.weak_label.duplicated().any():
            raise ValueError('Missing or duplicate frozen weak-source labels')
        if not np.isfinite(table[['x_pix_0based','y_pix_0based','catalog_ra_deg','catalog_dec_deg']].to_numpy(float)).all():
            raise ValueError('Non-finite frozen catalog positions')
        table = table.sort_values('weak_label').reset_index(drop=True)
        old_reference = processed/'asteris_paper_400/coadds'/sequence/f'input_coadd_{sequence}.fits'
        new_reference = processed/'blind_joint'/sequence/f'weighted_coadd_{sequence}.fits'
        offset, diagnostic = estimate_display_translation(new_reference,old_reference)
        solution_path = catalog_root/'astrometric_solution.json'
        solution = json.loads(solution_path.read_text(encoding='utf-8'))
        products = {}
        paths = {'weighted_coadd':new_reference,
                 'joint_score':processed/'blind_joint'/sequence/f'joint_score_{sequence}.fits'}
        for key,path in paths.items():
            image,header = read_display_image(path)
            if not header.get('REFIMAGE') or int(header.get('NCOMBINE',0)) < 2:
                raise ValueError('Missing current coadd reference/exposure metadata')
            products[key] = dict(image=image,path=path,offset=offset,score=key=='joint_score',
                                 refimage=header['REFIMAGE'],exposures=int(header['NCOMBINE']),
                                 short_title=f"Current {'coadd' if key=='weighted_coadd' else 'score'}\n{header['NCOMBINE']} exposures")
        if products['weighted_coadd']['refimage'] != products['joint_score']['refimage']:
            raise ValueError('Current coadd and score have different reference frames')
        for profile in ('160','400'):
            model_root = processed/f'asteris_paper_{profile}'
            path = model_root/'coadds'/sequence/f'asteris8_coadd_{sequence}.fits'
            calibrated = catalog_root/f'asteris_paper_{profile}_asteris8_coadd_{sequence}_catalog.fits'
            # Reject stale cached WCS/catalog overlays after a new model run.
            if not np.array_equal(fits.getdata(path),fits.getdata(calibrated),equal_nan=True):
                raise ValueError(f'Cached catalog calibration no longer matches ASTERIS {profile}')
            # Reuse the already calibrated FITS table, without solving astrometry
            # again or importing WCS/BLAS into a possibly torch-loaded notebook.
            with fits.open(calibrated,memmap=False) as hdul:
                saved = hdul['CATALOG'].data
                saved_labels = np.asarray(saved['weak_label']).astype(str)
                for row in table.itertuples():
                    match = np.flatnonzero(saved_labels == row.weak_label)
                    if len(match) != 1:
                        raise ValueError('Frozen catalog label differs from saved FITS catalog')
                    old = saved[match[0]]
                    names = ('x_pix_0based','y_pix_0based','catalog_ra_deg','catalog_dec_deg')
                    if not np.allclose([old[n] for n in names],[getattr(row,n) for n in names],atol=1e-6,rtol=0):
                        raise ValueError('Frozen coordinates disagree with saved FITS catalog')
            image,_ = read_display_image(path)
            stats = pd.read_csv(model_root/'paper_coadd_statistics.csv',dtype={'sequence':str}).set_index('sequence')
            exposures = int(stats.loc[sequence,'test_exposures'])
            products[f'asteris{profile}'] = dict(image=image,path=path,offset=(0,0),score=False,
                exposures=exposures,short_title=f'ASTERIS train={profile}\n{exposures} exposures, old input')
        destination = output/sequence
        destination.mkdir(parents=True,exist_ok=True)
        positions,tables = {},[]
        for key,product in products.items():
            positions[key] = transfer_positions(table,product['offset'],product['image'])
            tables.append(positions[key].assign(product=key,science_path=str(product['path'])))
        payload = dict(sequence=sequence,destination=str(destination),dpi=dpi,half=cutout_half_size,
            export_interpolated=bool(export_interpolated),max_hole_pixels=int(max_hole_pixels),
            labels=table.weak_label.tolist(),
            products={key:dict(path=str(p['path']),score=p['score'],short_title=p['short_title']) for key,p in products.items()},
            positions={key:t[['weak_label','x_plot','y_plot','in_frame','valid_at_position','k_m']].to_dict('records') for key,t in positions.items()})
        environment = os.environ.copy()
        # Use this installed source tree even for synthetic projects in tests.
        source_root = Path(__file__).resolve().parents[2]
        environment['PYTHONPATH'] = str(source_root)+os.pathsep+environment.get('PYTHONPATH','')
        environment['MPLBACKEND'] = 'Agg'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        worker = subprocess.run([sys.executable,'-m','astr_ir.evaluation.catalog_visualization','--render-worker'],
            input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,check=True,timeout=180,env=environment)
        rendering = json.loads(worker.stdout)
        masked = rendering['variants']['masked']
        paths_out,cutouts = masked['overlays'],masked['cutouts']
        pd.concat(tables,ignore_index=True).to_csv(destination/'plotted_positions.csv',index=False,encoding='utf-8-sig')
        inputs = [table_path,solution_path,old_reference,*[p['path'] for p in products.values()]]
        report = dict(sequence=sequence,weak_sources=len(table),registration=diagnostic,
                      original_wcs_anchor_rms_pix=solution['anchor_rms_pix'],
                      coordinate_rule='x_current=x_catalog+dx; y_current=y_catalog+dy; zero-based; no peak snapping',
                      catalog_role='display only; no processing, detection, training or weak-source reselection',
                      selection_caveat='Frozen historical sample used input peak SNR / K cuts; not a complete or unbiased catalog sample.',
                      comparison_caveat='Current coadd/score and old ASTERIS use different exposure counts and preprocessing; no performance claim.',
                      mask_rule='DQ bit 0 (DO_NOT_USE) or non-finite; keep valid partial coverage. Without DQ, non-finite only; finite unflagged defects cannot be identified here. Never overlay a native detector blindmap on coadds.',
                      interpolation=dict(enabled=bool(export_interpolated),max_hole_pixels=int(max_hole_pixels),
                          rule='Display copy only: enclosed 8-connected holes; Gaussian-weighted original valid neighbors, sigma=1, radius=3; valid pixels unchanged. No catalog input.',
                          normalization='Both variants use identical stretches from ORIGINAL valid pixels; positions/validity flags unchanged.'),
                      mask_statistics=rendering['mask_statistics'],display_variants=rendering['variants'],
                      sources_sha256={str(p):_sha256(p) for p in inputs},overlays=paths_out,cutouts=cutouts)
        (destination/'visualization_metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        records.append(dict(sequence=sequence,weak_sources=len(table),matched_stars=diagnostic['matched_stars'],
                            transfer_rms_pix=diagnostic['registration_rms_2d_pix'],
                            original_wcs_rms_pix=solution['anchor_rms_pix'],**paths_out,cutouts=cutouts,
                            display_interpolated=rendering['variants'].get('display_interpolated'),
                            mask_statistics=rendering['mask_statistics']))
    return records


if __name__ == '__main__':
    if sys.argv[1:] != ['--render-worker']:
        raise SystemExit('Use export_catalog_validation() in the evaluation notebook.')
    print(json.dumps(_render_worker(json.load(sys.stdin))))
