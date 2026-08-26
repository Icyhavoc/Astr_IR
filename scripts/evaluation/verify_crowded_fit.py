"""Replay frozen raw-exposure injections to compare old size rejection and new fits.

The legacy branch emulates ONLY the previous overflow rejection; unchanged
small-group fitting, PSFs, calibration and candidate discovery are shared.
Truth is read solely for injection generation and post-detection scoring.
"""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from dataclasses import asdict
import argparse,hashlib,json,sys,time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from astr_ir.evaluation import weak_detection as detection
from astr_ir.evaluation.stage_recovery import inject_native,_stage_exposures,_match_unique
from astr_ir.evaluation.blind_joint import inspect_array
from astr_ir.flicker.processor import load_fits,load_detector_mask,correct_flicker
from astr_ir.registration import science_valid


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


@contextmanager
def legacy_size_rejection():
    original=detection.fit_native_sources
    def fit(*args,**kwargs):
        detailed=kwargs.pop('return_diagnostics',False)
        with patch('astr_ir.evaluation.crowded_fit.fit_crowded_group',side_effect=ValueError('legacy size limit')):
            positions,flux,errors,flags,details=original(*args,return_diagnostics=True,**kwargs)
        for index,detail in enumerate(details):
            if detail['fit_method']=='sparse_joint_local_background': flags[index]='group_too_large'
        result=(positions,flux,errors,flags)
        return (*result,details) if detailed else result
    with patch.object(detection,'fit_native_sources',fit): yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=ROOT/'data/raw/our_dataset')
    parser.add_argument('--threshold',type=float,default=5.)
    parser.add_argument('--fluxes',type=float,nargs='+',default=[6000,12000])
    parser.add_argument('--historical-candidates',type=Path,help='Optional old 12000-DN/default-5 candidate table to reproduce')
    args=parser.parse_args()
    output=args.output_root.resolve(); reference=args.reference_run.resolve()
    if output.exists(): raise FileExistsError('Use a new verification directory')
    if output.is_relative_to(args.dataset_root.resolve()) or output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError('Output cannot overlap raw data or the frozen reference run')
    report=json.loads((reference/'report.json').read_text(encoding='utf-8'))
    if report['phase']!='validation': raise ValueError('This development replay requires validation frames')
    files=[Path(name) for name in report['input_sha256']]
    hashes={str(p):digest(p) for p in files}
    if hashes!=report['input_sha256']: raise RuntimeError('Frozen raw inputs changed')
    output.mkdir(parents=True)
    started=time.perf_counter()
    def emit(message):
        print(f'[{time.perf_counter()-started:.1f}s] {message}',flush=True)
        (output/'progress.json').write_text(json.dumps(dict(message=message,complete=False),indent=2),encoding='utf-8')
    config=detection.DetectionConfig(threshold=args.threshold)
    detector=load_detector_mask(args.dataset_root/'盲点表')
    images=[]; masks=[]; psfs=[]
    emit('Loading frozen exposures and image-only injection PSFs')
    for path in files:
        image,_=load_fits(path); valid=science_valid(path,image,detector)
        flicker=correct_flicker(image,detector_mask=~valid).corrected
        psfs.append(detection.estimate_psf(flicker,valid,inspect_array(flicker,valid)[4],config)[0])
        images.append(image); masks.append(valid)
    raw=np.asarray(images,np.float32); valid=np.asarray(masks,bool); offsets=np.asarray(report['offsets_yx'])
    prior=pd.read_csv(reference/'source_recovery.csv')
    truth=prior.loc[prior.stage.eq('raw') & prior.flux.eq(12000) & prior.threshold.eq(5),['x','y','environment']].drop_duplicates()
    if truth.empty: raise ValueError('No frozen injection positions')
    summary=[]; per_source=[]; historical_reproduced=None
    baseline={}
    for flux in [0.,*args.fluxes]:
        injected=raw if flux==0 else inject_native(raw,valid,offsets,psfs,truth.assign(flux=flux),np.random.default_rng(1))
        exposures=_stage_exposures((injected,valid,offsets),config)
        for method in ('legacy','crowded_joint'):
            emit(f'{method}: blind detection, injected flux={flux:g}')
            if method=='legacy':
                with legacy_size_rejection(): result=detection.analyze_exposures(exposures,config)
            else: result=detection.analyze_exposures(exposures,config)
            candidates=result['sources']; accepted=candidates.loc[candidates.accepted.astype(bool)]
            stem=f'{method}_{flux:g}'
            candidates.to_csv(output/f'{stem}_candidates.csv',index=False,encoding='utf-8-sig')
            (output/f'{stem}_diagnostic.json').write_text(json.dumps(result['diagnostic'],indent=2),encoding='utf-8')
            if method=='legacy' and flux==12000 and args.historical_candidates:
                old=pd.read_csv(args.historical_candidates)
                columns=['x','y','flux','flux_error_nominal','snr_empirical','fit_flag','accepted','initial_score']
                pd.testing.assert_frame_equal(old[columns].reset_index(drop=True),candidates[columns].reset_index(drop=True),
                    check_dtype=False,rtol=1e-9,atol=1e-8)
                historical_reproduced=True
            if flux==0:
                baseline[method]=accepted[['x','y']].to_numpy()
                matched={}; existing={}; new_unmatched=0
            else:
                points=truth[['x','y']].to_numpy(); found=accepted[['x','y']].to_numpy()
                matched=_match_unique(points,found,2.5)
                existing=_match_unique(points,baseline[method],2.5)
                associated=_match_unique(baseline[method],found,2.5)
                used={j for j,_ in matched.values()}|{j for j,_ in associated.values()}
                new_unmatched=len(found)-len(used)
                for index,row in enumerate(truth.itertuples()):
                    distance=np.hypot(candidates.x-row.x,candidates.y-row.y)
                    nearest=candidates.loc[distance.idxmin()] if len(candidates) else None
                    per_source.append(dict(method=method,flux=flux,x=row.x,y=row.y,environment=row.environment,
                        recovered=index in matched,preexisting=index in existing,
                        localization_error=matched[index][1] if index in matched else np.nan,
                        nearest_distance=float(distance.min()) if len(candidates) else np.nan,
                        nearest_flag=nearest.fit_flag if nearest is not None else 'no_candidate',
                        nearest_group_size=int(nearest.group_size) if nearest is not None else 0))
            summary.append(dict(method=method,flux=flux,candidates=len(candidates),accepted=len(accepted),
                recovered=len(matched),eligible_new=len(truth)-len(existing) if flux else 0,
                recovered_new=len(set(matched)-set(existing)),preexisting=len(existing),new_unmatched_candidates=new_unmatched,
                size_rejected=int(candidates.fit_flag.eq('group_too_large').sum()),
                degenerate=int(candidates.fit_flag.eq('degenerate_group').sum()),
                crowded_candidates=result['diagnostic']['crowded_candidates'],
                largest_group=result['diagnostic']['largest_fitted_group']))
            pd.DataFrame(summary).to_csv(output/'comparison.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(per_source).to_csv(output/'source_recovery.csv',index=False,encoding='utf-8-sig')
            emit(f'{method}: flux={flux:g}, recovered={len(matched)}, accepted={len(accepted)}')
    if any(digest(path)!=hashes[str(path)] for path in files): raise RuntimeError('Raw data changed')
    (output/'report.json').write_text(json.dumps(dict(complete=True,catalog_used=False,training_started=False,
        reference_run=str(reference),frames=len(files),positions=len(truth),config=asdict(config),
        historical_candidates_reproduced=historical_reproduced,input_sha256=hashes,
        source_sha256={name:digest(ROOT/'src/astr_ir/evaluation'/name) for name in ('weak_detection.py','crowded_fit.py')},
        scope='Frozen raw-exposure validation replay, not a complete downstream/model performance test',
        caveat='New unmatched real-sky candidates are not known false positives; no calibrated FAP or independent test claim'),
        indent=2),encoding='utf-8')
    (output/'progress.json').write_text(json.dumps(dict(message='Complete',complete=True)),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__': main()
