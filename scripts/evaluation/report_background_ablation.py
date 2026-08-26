"""Read completed ablation products; write comparison tables and display-only PNGs."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Circle
    from astr_ir.evaluation.catalog_visualization import (read_display_image, transfer_positions,
            estimate_display_translation, interpolate_display_holes)
    # Do not import background_ablation (or torch) in this rendering process.
    import hashlib
    def sha256(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    def save_json(path, payload):
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    experiment = args.experiment.resolve()
    if not experiment.is_relative_to(ROOT/'data/processed'):
        raise ValueError('Unexpected experiment directory')
    selection = json.loads((experiment/'selection.json').read_text(encoding='utf-8'))
    selected = selection['selected']
    methods = ('old64_single', selected)
    sequences = ('90000002','90000003','90000004','90000005_1','90000005_2')
    tables = []
    for sequence in sequences:
        directory = experiment/'test'/sequence
        if not json.loads((directory/'progress.json').read_text())['complete']:
            raise RuntimeError('Cannot label partial testing as complete')
        tables.append(pd.read_csv(directory/'source_recovery.csv', dtype={'sequence':str}))
    all_rows = pd.concat(tables, ignore_index=True)
    keys = ['sequence','repeat','flux','x','y']
    common = all_rows.loc[~all_rows.groupby(keys).preexisting_match.transform('any')].copy()
    if not common.groupby(keys).size().eq(4).all():
        raise AssertionError('Expected the same four estimators for every paired trial')
    summaries = []
    for (method, flux), group in common.groupby(['method','flux']):
        response = group.paired_flux_response.to_numpy()
        summaries.append(dict(method=method, flux=flux, recovered=int(group.recovered.sum()),
            eligible=len(group), recovery=float(group.recovered.mean()),
            response_median=float(np.nanmedian(response)), response_p16=float(np.nanpercentile(response,16)),
            response_p84=float(np.nanpercentile(response,84)),
            median_localization_error=float(group.loc[group.recovered,'localization_error'].median())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(experiment/'final_comparison.csv', index=False)
    all_rows.to_csv(experiment/'all_test_sources.csv', index=False)
    wide = common.pivot(index=keys, columns='method', values='recovered')
    comparisons = []
    pairs = [(m+'_input',m+'_asteris') for m in methods]
    pairs += [('old64_single_input',selected+'_input'),('old64_single_asteris',selected+'_asteris')]
    for before, after in pairs:
        for flux, group in wide.groupby(level='flux'):
            a, b = group[before].astype(bool), group[after].astype(bool)
            comparisons.append(dict(before=before, after=after, flux=flux, eligible=len(group),
                both=int((a&b).sum()), gained=int((~a&b).sum()), lost=int((a&~b).sum()),
                neither=int((~a&~b).sum()), net_gain=int(b.sum()-a.sum())))
    paired = pd.DataFrame(comparisons)
    paired.to_csv(experiment/'paired_test_comparison.csv', index=False)
    diagnostic = pd.concat([pd.read_csv(experiment/'test'/s/'summary.csv',dtype={'sequence':str}) for s in sequences],ignore_index=True)
    diagnostic.groupby(['method','flux'])[['baseline_candidates','injected_candidates','new_unmatched_candidates']].agg(
        ['mean','max']).to_csv(experiment/'unmatched_diagnostics.csv')
    calibration = json.loads((experiment/'model_thresholds.json').read_text())
    null_rows = []
    for name, arrays in calibration['scores'].items():
        threshold = calibration['thresholds'][name]
        counts = [int(np.count_nonzero(np.asarray(a)>=threshold)) for a in arrays]
        null_rows.append(dict(method=name,threshold=threshold,simulations=len(counts),
                              observed_mean_false_peaks=float(np.mean(counts)),max_false_peaks=max(counts),
                              interpretation='Finite parametric simulations only; NOT real-sky purity'))
    pd.DataFrame(null_rows).to_csv(experiment/'null_calibration.csv',index=False)
    quality = []
    for method in methods:
        root = ROOT/'data/processed/background' if method=='old64_single' else experiment/'background'/method
        statistics = pd.read_csv(root/'background_statistics.csv',dtype={'sequence':str})
        for sequence, group in statistics.groupby('sequence'):
            quality.append(dict(method=method,sequence=sequence,frames=len(group),
                median_large_scale_scatter=float(group.large_scale_scatter_after.median()),
                median_high_frequency_noise=float(group.high_frequency_noise_after.median()),
                median_source_mask_fraction=float(group.source_mask_fraction.median())))
    pd.DataFrame(quality).to_csv(experiment/'background_quality.csv',index=False)
    output = ROOT/'figures/background_ablation_output'
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors = {'old64_single':'#576174', selected:'#007c83'}
    for name, group in summary.groupby('method'):
        group = group.sort_values('flux')
        arm = name.removesuffix('_input').removesuffix('_asteris')
        style = '-' if name.endswith('_asteris') else '--'
        for ax, column in zip(axes, ('recovery','response_median')):
            ax.plot(group.flux, group[column], style, marker='o', color=colors[arm], label=name)
            ax.set_xlabel('Injected integrated flux per exposure [DN]')
            ax.grid(alpha=.2)
    axes[0].set(ylabel='Recovered / common eligible injected sources', ylim=(0,1.05))
    axes[1].set(ylabel='Median paired flux response')
    axes[1].axhline(1.,color='black',lw=.7)
    axes[0].legend(fontsize=8)
    fig.suptitle('Held-out test | 5 sequences, 16 exposures each | catalog-free processing')
    fig.supxlabel('Thresholds frozen on parametric source-free validation; not a measured real-sky false-positive rate.', fontsize=9)
    fig.savefig(output/'comparison.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4),layout='constrained')
    training = []
    for method in methods:
        model_root = experiment/'models'/method
        progress = json.loads((model_root/'checkpoints/training_progress.json').read_text())
        if not progress['complete']:
            raise RuntimeError('Training incomplete')
        history = pd.read_csv(model_root/'checkpoints/training_history.csv')
        ax.plot(history.epoch,history.validation_loss,marker='.',label=method)
        training.append(dict(method=method,epochs=len(history),best_epoch=int(history.loc[history.validation_loss.idxmin(),'epoch']),
            best_validation_loss=float(history.validation_loss.min()),seconds=progress['elapsed_seconds']))
    ax.set(xlabel='Epoch',ylabel='Validation loss (different normalized input domains)',title='Training diagnostics, not a source-recovery metric')
    ax.legend()
    fig.savefig(output/'training.png',dpi=150)
    plt.close(fig)
    # Only now, after all science selection/training/testing, read the frozen
    # star catalog for human display. Never reselect targets from model results.
    display_reports = []
    for sequence in ('90000002','90000003'):
        catalog_root = ROOT/'data/processed/evaluation/asteris_paper_catalog'/sequence
        catalog_path = catalog_root/'weak_sources.csv'
        catalog = pd.read_csv(catalog_path).sort_values('weak_label')
        old_reference = ROOT/'data/processed/asteris_paper_400/coadds'/sequence/f'input_coadd_{sequence}.fits'
        control_reference = experiment/'models/old64_single/coadds'/sequence/f'input_coadd_{sequence}.fits'
        offset, registration = estimate_display_translation(control_reference, old_reference)
        products = {}
        for method in methods:
            for kind, prefix in (('input','input_coadd'),('asteris','asteris8_coadd')):
                path = experiment/'models'/method/'coadds'/sequence/f'{prefix}_{sequence}.fits'
                image, _ = read_display_image(path)
                products[method+'_'+kind] = dict(path=path,image=image,positions=transfer_positions(catalog,offset,image),sha256=sha256(path))
        destination = output/sequence
        destination.mkdir(exist_ok=True)
        cmap = plt.get_cmap('gray').copy()
        cmap.set_bad('#808080')
        # A shared stretch across ALL four products for each cutout; differences
        # in apparent brightness must not be manufactured by separate autoscaling.
        reference_image = products['old64_single_input']['image']
        labels = catalog.weak_label.tolist()
        positions = products['old64_single_input']['positions'].set_index('weak_label')
        for interpolate in (False, True):
            mode = 'display_interpolated' if interpolate else 'masked'
            folder = destination/mode
            folder.mkdir(exist_ok=True)
            shown = {key:interpolate_display_holes(p['image'])[0] if interpolate else p['image'] for key,p in products.items()}
            values = reference_image[np.isfinite(reference_image)]
            lo, hi = np.percentile(values,[1,99.7])
            norm = PowerNorm(.5,vmin=lo,vmax=hi,clip=True)
            for key, product in products.items():
                fig, ax = plt.subplots(figsize=(9,9),layout='constrained')
                ax.imshow(np.ma.masked_invalid(shown[key]), origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
                for row in product['positions'].itertuples():
                    if row.in_frame:
                        color = '#00e5ff' if row.valid_at_position else '#ffad42'
                        ax.add_patch(Circle((row.x_plot,row.y_plot),9,fill=False,color=color,lw=.9))
                        ax.annotate(row.weak_label,(row.x_plot,row.y_plot),xytext=(7,7),textcoords='offset points',color=color,fontsize=8)
                ax.set_title(f'{sequence} | {key} | 16 test exposures\n{mode}: catalog positions, NOT detection claims')
                fig.savefig(folder/f'{key}_catalog.png',dpi=145)
                plt.close(fig)
            for start in range(0,len(labels),4):
                page = labels[start:start+4]
                fig, axes = plt.subplots(len(page),4,figsize=(12,3*len(page)),squeeze=False,layout='constrained')
                for i,label in enumerate(page):
                    point = positions.loc[label]
                    x,y = int(round(point.x_plot)),int(round(point.y_plot))
                    ys,xs = slice(max(0,y-24),min(reference_image.shape[0],y+25)),slice(max(0,x-24),min(reference_image.shape[1],x+25))
                    values = reference_image[ys,xs]
                    values = values[np.isfinite(values)]
                    if not len(values):
                        continue
                    lo,hi = np.percentile(values,[1,99.7])
                    norm = PowerNorm(.5,vmin=lo,vmax=max(hi,lo+1e-6),clip=True)
                    for j,(key,product) in enumerate(products.items()):
                        ax = axes[i,j]
                        ax.imshow(np.ma.masked_invalid(shown[key][ys,xs]),origin='lower',cmap=cmap,norm=norm,interpolation='nearest',
                                  extent=(xs.start-.5,xs.stop-.5,ys.start-.5,ys.stop-.5))
                        row = product['positions'].set_index('weak_label').loc[label]
                        ax.add_patch(Circle((row.x_plot,row.y_plot),6,fill=False,color='#00e5ff' if row.valid_at_position else '#ffad42'))
                        if i==0:
                            ax.set_title(key,fontsize=9)
                        if j==0:
                            ax.set_ylabel(f'{label}\nK={row.k_m:.2f}')
                        ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(f'{sequence} | same 16 test frames | shared per-source stretch | {mode}')
                fig.supxlabel('Catalog for display only; no peak snapping. Gray=invalid. Cosmetic interpolation never enters FITS or measurements.',fontsize=8)
                fig.savefig(folder/f'weak_sources_{start//4+1:02d}.png',dpi=160)
                plt.close(fig)
        for product in products.values():
            if sha256(product['path']) != product['sha256']:
                raise AssertionError('Visualization altered science FITS')
        pd.concat([p['positions'].assign(product=k) for k,p in products.items()]).to_csv(destination/'plotted_positions.csv',index=False)
        record = dict(sequence=sequence,registration=registration,catalog_sha256=sha256(catalog_path),
            files={k:dict(path=str(p['path']),sha256=p['sha256']) for k,p in products.items()},
            common_pixel_grid=True, shared_stretches=True,catalog_role='display only; frozen historical sample not unbiased completeness catalog')
        save_json(destination/'visualization_metadata.json',record)
        display_reports.append(record)
    audit = json.loads((experiment/'control_audit.json').read_text(encoding='utf-8'))
    for item in audit['files']:
        if sha256(item['path']) != item['sha256']:
            raise AssertionError('Existing control science file changed')
    save_json(experiment/'completion.json',dict(complete=True,selected_background=selected,training=training,
        validation_gate_passed=selection['qualified'],production_defaults_changed=False,
        test_sequences=list(sequences),test_exposures_per_sequence=16,
        unique_test_positions=int(all_rows[['sequence','repeat','x','y']].drop_duplicates().shape[0]),
        old_control_fits_unchanged=True,catalog_display_sequences=['90000002','90000003'],
        figures=str(output),limitation='Model-conditional null budget; not empirical real-sky purity or unseen-field generalization.'))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print('Complete: '+str(experiment),flush=True)


if __name__ == '__main__':
    main()
