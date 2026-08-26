"""Reproducible background selection, 400-frame training and held-out evaluation.

Each subcommand is explicit. Training never starts from a notebook import.
Existing science files and historical checkpoints are not overwritten.
"""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
# Avoid CPU oversubscription, and keep plotting in a separate clean process.
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')


def build_single_sequence(sequence, input_root, raw, output, box, filt, start=0, count=80):
    """Disjoint frame batches; the parent alone writes the combined CSV."""
    import time
    import pandas as pd
    from astr_ir.background.processor import BackgroundConfig, load_detector_mask, process_fits_file
    from astr_ir.evaluation.background_ablation import save_json
    config = BackgroundConfig(final_box_size=box, final_filter_size=filt)
    detector = load_detector_mask(Path(raw)/'盲点表')
    rows = []
    started = time.monotonic()
    files = sorted((Path(input_root)/sequence).glob('flicker_corrected_*.fits'))
    if len(files) != 80:
        raise ValueError('Expected 80 flicker science inputs per sequence')
    for index, path in enumerate(files[start:start+count], start+1):
        _, row = process_fits_file(path, Path(output)/sequence, detector, None, config)
        row.update(sequence=sequence, sequence_frame_index=index)
        for name in ('subtracted_path','model_path'):
            row[name] = Path(row[name]).relative_to(output).as_posix()
        rows.append(row)
        if index % 10 == 0:
            print(f'background {sequence}: {index}/80, {time.monotonic()-started:.0f}s',flush=True)
            pd.DataFrame(rows).to_csv(Path(output)/sequence/f'background_batch_{start:03d}.csv',index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'select', 'audit', 'build', 'prepare', 'train', 'infer', 'test', 'calibrate'])
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--method')
    parser.add_argument('--sources', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--null-repeats', type=int, default=4)
    args = parser.parse_args()
    import json
    import pandas as pd
    from astr_ir.evaluation.background_ablation import run_field, save_json, BACKGROUND_METHODS, sha256
    experiment = args.experiment.resolve()
    if not experiment.is_relative_to(ROOT/'data/processed'):
        raise ValueError('Experiment must be a new directory inside data/processed')
    raw = ROOT/'data/raw/our_dataset'
    historical = ROOT/'data/processed/asteris_paper_400/manifests/split_manifest.csv'
    sequences = ('90000002', '90000003', '90000004', '90000005_1', '90000005_2')
    def background_root(method):
        return ROOT/'data/processed/background' if method == 'old64_single' else experiment/'background'/method
    if args.stage == 'audit':
        from dataclasses import asdict
        from astropy.io import fits
        import numpy as np
        from astr_ir.background.processor import BackgroundConfig, subtract_background, load_detector_mask
        from astr_ir.registration import science_valid
        provenance = json.loads((ROOT/'data/processed/pre_asteris_blind_v2/config.json').read_text())
        current = json.loads(json.dumps(asdict(BackgroundConfig())))
        if provenance['catalog_used'] or provenance['background'] != current:
            raise AssertionError('Existing control preprocessing provenance differs')
        detector = load_detector_mask(raw/'盲点表')
        records, checks = [], []
        for sequence in sequences:
            files = sorted((background_root('old64_single')/sequence).glob('background_subtracted_*.fits'))
            if len(files) != 80:
                raise AssertionError('Control must contain 80 frames per sequence')
            for path in files:
                with fits.open(path, memmap=False) as hdul:
                    if hdul[0].header['HIERARCH BKG BOX'] != 64 or 'DQ' not in hdul:
                        raise AssertionError('Unexpected control mesh or missing DQ')
                records.append(dict(path=str(path), sha256=sha256(path)))
            for path in (files[0], files[46], files[-1]):
                flicker_path = ROOT/'data/processed/flicker'/sequence/path.name.replace('background_subtracted_', 'flicker_corrected_', 1)
                image = np.asarray(fits.getdata(flicker_path), float)
                valid = science_valid(flicker_path, image, detector)
                result = subtract_background(image, detector_mask=~valid, config=BackgroundConfig())
                expected = image.astype(np.float32)-result.background_model.astype(np.float32)
                disk = fits.getdata(path)
                finite = np.isfinite(expected) & np.isfinite(disk)
                error = float(np.max(np.abs(expected[finite]-disk[finite])))
                if error > .005:
                    raise AssertionError(f'Control numerical reproduction failed: {path}, {error}')
                checks.append(dict(path=str(path), max_abs_difference=error))
            print('Audited old background '+sequence, flush=True)
        save_json(experiment/'control_audit.json', dict(files=records, sentinel_checks=checks,
                  frames=400, provenance=provenance, control_reused_read_only=True))
        return
    if args.stage == 'validation':
        run_field(raw, historical, experiment/'validation'/args.sequence, args.sequence,
                  seed=2026082601+sequences.index(args.sequence)*1000,
                  sources=args.sources, repeats=args.repeats, null_repeats=args.null_repeats)
        return
    if args.stage == 'select':
        import numpy as np
        fields = ('90000002', '90000003')
        tables = [pd.read_csv(experiment/'validation'/s/'source_recovery.csv', dtype={'sequence': str}) for s in fields]
        rows = pd.concat(tables, ignore_index=True)
        # Same eligibility across ALL arms; historical real stars cannot inflate
        # an arm's recovery by accidentally overlapping an injection location.
        keys = ['sequence', 'repeat', 'flux', 'x', 'y']
        ineligible = rows.groupby(keys).preexisting_match.transform('any')
        paired = rows.loc[~ineligible].copy()
        control_response = paired.loc[paired.method.eq('old64_single_input'), keys+['paired_flux_response']].rename(
            columns={'paired_flux_response':'control_response'})
        paired = paired.merge(control_response,on=keys,validate='many_to_one')
        paired['relative_response'] = paired.paired_flux_response / paired.control_response.where(paired.control_response>0)
        summaries = []
        for method, group in paired.groupby('method'):
            response = group.paired_flux_response.to_numpy()
            summaries.append(dict(method=method.removesuffix('_input'), eligible=len(group),
                recovered=int(group.recovered.sum()), recovery=float(group.recovered.mean()),
                response_median=float(np.nanmedian(response)),
                relative_response_median=float(group.relative_response.median()),
                median_absolute_response_error=float(np.nanmedian(np.abs(response-1))),
                median_relative_response_error=float((group.relative_response-1).abs().median())))
        scores = pd.DataFrame(summaries)
        # Prespecified scientific rule: tolerate <=5% ADDED median transfer bias
        # relative to old background. Absolute coadd response also contains the
        # shared clipping/PSF calibration bias and remains reported separately.
        # <=2 recovered-source loss vs old; prefer greater recovery, then less
        # transfer error. Two-pass must add >=2 recoveries over single-pass to
        # justify complexity, not merely flatten the background.
        old = scores.set_index('method').loc['old64_single']
        single = scores.set_index('method').loc['new32_single']
        candidates = scores.loc[scores.method.ne('old64_single') &
            scores.relative_response_median.between(.95, 1.05) & (scores.recovered >= old.recovered-2)]
        candidates = candidates.loc[candidates.method.ne('new32_two') |
                                    (candidates.recovered >= single.recovered+2)]
        qualified = not candidates.empty
        # The user requested BOTH training arms, not only a positive result.
        # Preserve gate failure explicitly. The simplest predeclared new arm
        # is still trained as a diagnostic comparator, never called a winner.
        selected = ('new32_single' if candidates.empty else
                    candidates.sort_values(['recovered', 'median_relative_response_error'], ascending=[False, True]).iloc[0].method)
        scores.to_csv(experiment/'background_selection.csv', index=False)
        thresholds = {}
        for name in BACKGROUND_METHODS:
            key = name+'_input'
            thresholds[key] = max(json.loads((experiment/'validation'/s/'thresholds.json').read_text())['thresholds'][key] for s in fields)
        selection = dict(selected=selected, thresholds=thresholds, criteria='median paired response relative to old 0.95..1.05; at most 2 losses vs old; two-pass requires >=2 gains vs single',
                         qualified=qualified, gate_winner=selected if qualified else None,
                         production_recommendation=selected if qualified else 'old64_single',
                         selection_status='qualified validation candidate' if qualified else 'No candidate passed; single-pass new arm retained for explicitly requested diagnostic training, NOT promoted.',
                         phase='validation', catalog_used=False, fields=fields,
                         unique_positions=int(rows[['sequence','repeat','x','y']].drop_duplicates().shape[0]),
                         caveat='Small diagnostic sample; null calibration is conditional on simulated noise.')
        save_json(experiment/'selection.json', selection)
        print(scores.to_string(index=False), flush=True)
        print(json.dumps(selection, indent=2), flush=True)
        return
    if args.stage == 'prepare' and args.method == 'old64_single' and not (experiment/'selection.json').exists():
        selection = {'selected': 'old64_single'}
    else:
        selection = json.loads((experiment/'selection.json').read_text(encoding='utf-8'))
    selected = selection['selected']
    methods = tuple(dict.fromkeys(('old64_single', selected)))
    if selected == 'old64_single' and not (args.stage == 'prepare' and args.method == 'old64_single'):
        raise RuntimeError('No new background passes validation: do not fabricate a winning arm; inspect selection report')
    if args.stage == 'build':
        from astr_ir.background.processor import BackgroundConfig, run_batch
        from astr_ir.background.sequence import run_two_pass_batch
        method = args.method or selected
        if method not in methods:
            raise ValueError('Only frozen selected and control methods may be built')
        box, filt, two = BACKGROUND_METHODS[method]
        output = experiment/'background'/method
        if output.exists():
            raise FileExistsError('Fresh background output required')
        config = BackgroundConfig(final_box_size=box, final_filter_size=filt)
        if two:
            stats = run_two_pass_batch(ROOT/'data/processed/flicker', raw, output,
                sequences=sequences, background_config=config, split_manifest=historical)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            records = []
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(build_single_sequence, sequence, ROOT/'data/processed/flicker', raw, output, box, filt, start, 20)
                           for sequence in sequences for start in range(0,80,20)]
                for future in as_completed(futures):
                    records.extend(future.result())
                    save_json(output/'progress.json',dict(frames=len(records),total=400,complete=len(records)==400))
            stats = pd.DataFrame(records).sort_values(['sequence','sequence_frame_index'])
            stats.to_csv(output/'background_statistics.csv',index=False)
        if len(stats) != 400:
            raise AssertionError('Expected all 400 frames')
        save_json(output/'complete.json', dict(frames=len(stats), method=method, catalog_used=False))
        print(stats.groupby(['sequence', 'status']).size(), flush=True)
        return
    if args.stage == 'prepare':
        from dataclasses import asdict
        import platform
        import torch
        torch.set_num_threads(2)
        from astr_ir.noise2noise.dataset import build_split_manifest
        from astr_ir.asteris.dataset import relabel_manifest_for_patch_t
        from astr_ir.asteris.paper_pipeline import prepare_paper_dataset, PaperAsterisConfig
        if not (experiment/'control_audit.json').exists():
            raise RuntimeError('Audit the existing control FITS before preparing stacks')
        common_path = experiment/'common_manifest.csv'
        if not common_path.exists():
            common = build_split_manifest(background_root('old64_single'), raw, common_path, sequences)
            common = relabel_manifest_for_patch_t(common, 8)
            frozen = pd.read_csv(historical, usecols=['sequence','filename','split'], dtype={'sequence':str})
            check = common.merge(frozen, on=['sequence','filename'], validate='one_to_one', suffixes=('', '_frozen'))
            if len(check) != 400 or not check.split.eq(check.split_frozen).all():
                raise AssertionError('Frozen temporal partition changed')
            common.to_csv(common_path, index=False)
        common = pd.read_csv(common_path, dtype={'sequence':str})
        config = PaperAsterisConfig()
        requested = (args.method,) if args.method else methods
        for method in requested:
            if method not in methods:
                raise ValueError('Unknown frozen preparation arm')
            output = experiment/'models'/method
            if output.exists():
                raise FileExistsError('Fresh model preparation required')
            prepare_paper_dataset(background_root(method), raw, output,
                                  sequences=sequences, config=config, frozen_manifest=common)
            print('Prepared '+method, flush=True)
        save_json(experiment/'training_protocol.json', dict(common_manifest_sha256=sha256(common_path),
            official_initialization_sha256=sha256(config.official_checkpoint), methods=methods,
            seed=config.seed, epochs=config.epochs, architecture='unchanged ASTERIS8',
            frames=400, training_frames=220, validation_frames=80, test_frames=80, guard_frames=20,
            registration='same image-only offsets for both arms', catalog_used=False,
            training_config=asdict(config), python=platform.python_version(), torch=torch.__version__,
            source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in (
                ROOT/'src/astr_ir/asteris/paper_pipeline.py',
                ROOT/'src/astr_ir/background/processor.py',
                ROOT/'src/astr_ir/background/sequence.py',
                ROOT/'src/astr_ir/evaluation/weak_detection.py',
                ROOT/'src/astr_ir/evaluation/crowded_fit.py',
                ROOT/'src/astr_ir/evaluation/background_ablation.py')}))
        return
    if args.stage in ('train', 'infer'):
        import torch
        torch.set_num_threads(2)
        from astr_ir.asteris.paper_pipeline import train_paper_model, run_paper_inference, PaperAsterisConfig
        if not torch.cuda.is_available():
            raise RuntimeError('Expected CUDA GPU; do not silently launch an impractical CPU training')
        requested = (args.method,) if args.method else methods
        for method in requested:
            if method not in methods:
                raise ValueError('Unknown frozen arm')
            output = experiment/'models'/method
            if args.stage == 'train':
                if (output/'checkpoints').exists():
                    raise FileExistsError('Refusing to overwrite trained model')
                train_paper_model(output, config=PaperAsterisConfig(), device='cuda')
                save_json(output/'training_complete.json', dict(complete=True, checkpoint_sha256=sha256(output/'checkpoints/best_checkpoint.pt')))
            else:
                run_paper_inference(background_root(method), raw, output,
                    output/'checkpoints/best_checkpoint.pt', evaluation_sequences=sequences, device='cuda')
            torch.cuda.empty_cache()
        return
    if args.stage in ('calibrate', 'test'):
        os.environ.setdefault('ASTR_IR_FRAME_WORKERS', '3')
        import torch
        torch.set_num_threads(2)
        from astr_ir.asteris.paper_pipeline import load_paper_model
        models = {}
        for method in methods:
            model, _, config = load_paper_model(experiment/'models'/method/'checkpoints/best_checkpoint.pt', 'cuda')
            models[method] = (model, config)
        if args.stage == 'calibrate':
            # Recalibrate all four estimators after training, using validation
            # only. No test recovery results influence thresholds.
            from astr_ir.evaluation.background_ablation import load_field, null_raw, products, detect, accepted
            from astr_ir.evaluation.stage_recovery import select_threshold_from_null
            import numpy as np
            scores = {m+suffix: [] for m in methods for suffix in ('_input','_asteris')}
            destination = experiment/'model_null_validation'
            if destination.exists():
                raise FileExistsError('Calibration already exists')
            destination.mkdir(parents=True)
            for sequence in ('90000002', '90000003'):
                raw_images, valid, offsets, *_ = load_field(raw, historical, sequence, 'validation', 16)
                for index in range(args.null_repeats):
                    simulated = null_raw(raw_images, valid, np.random.default_rng(2026082680+100*sequences.index(sequence)+index))
                    for name, product in products(simulated, valid, offsets, methods, models).items():
                        _, table = detect(product)
                        scores[name].append(accepted(table, 5.).snr_empirical.tolist())
                        table.to_csv(destination/f'{sequence}_{index}_{name}.csv', index=False)
                        print(f'Null {sequence} {index} {name}', flush=True)
            thresholds = {name:select_threshold_from_null(arrays, np.arange(5.,15.5,.5), .5)[0] for name, arrays in scores.items()}
            save_json(experiment/'model_thresholds.json', dict(thresholds=thresholds, scores=scores, phase='validation',
                       max_false_per_image=.5, model_conditional=True, exposures=16))
        else:
            thresholds = json.loads((experiment/'model_thresholds.json').read_text())['thresholds']
            run_field(raw, historical, experiment/'test'/args.sequence, args.sequence, phase='test', limit=16,
                      methods=methods, seed=2026082701+sequences.index(args.sequence)*1000,
                      repeats=args.repeats, sources=args.sources, thresholds=thresholds, models=models)


if __name__ == '__main__':
    main()
