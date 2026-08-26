"""Paired, catalog-free weak-source transfer tests BEFORE both preprocessors.

Truth exists only in injection generation and scoring. It is never an argument
to preprocessing, PSF construction, candidate discovery or network inference.
Real-image unmatched peaks are NOT counted as known false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from astr_ir.flicker.processor import correct_flicker, load_fits, load_detector_mask
from astr_ir.background.processor import BackgroundConfig, subtract_background
from astr_ir.background.sequence import subtract_sequence
from astr_ir.registration import science_valid, masked_shift
from .blind_joint import inspect_array, register_features
from .weak_detection import (DetectionConfig, Exposure, estimate_psf, local_noise_map,
    source_patch, analyze_exposures, _fit_group)


@dataclass(frozen=True)
class RecoveryConfig:
    seed: int = 20260826
    sources_per_trial: int = 6
    fluxes: tuple[float,...] = (100.,200.,400.)
    thresholds: tuple[float,...] = (4.,5.,6.)
    repeats: int = 2
    match_radius: float = 2.5
    min_separation: float = 20.
    edge: int = 40
    gain_e_per_dn: float | None = None

    def validate(self):
        if self.sources_per_trial<1 or self.repeats<1 or self.edge<16 or self.match_radius<=0 or self.min_separation<=2*self.match_radius:
            raise ValueError('Invalid recovery geometry/counts')
        if not self.fluxes or min(self.fluxes)<=0 or not self.thresholds or min(self.thresholds)<=0:
            raise ValueError('Positive fluxes and thresholds required')
        if self.gain_e_per_dn is not None and self.gain_e_per_dn<=0: raise ValueError('Gain must be positive')


def process_stage_chain(raw, valid, offsets, *, background_config=None, two_pass_config=None, inference=None):
    """No truth parameter; inference returns (denoised, mask) or (input, denoised, mask).

    The three-item form adds the exact paper input coadd as a matched control:
    both coadds then undergo the same single-image blind detection procedure.
    """
    config=background_config or BackgroundConfig()
    flicker=np.stack([np.where(mask,correct_flicker(image,detector_mask=~mask).corrected,np.nan).astype(np.float32)
                      for image,mask in zip(raw,valid)])
    single=np.stack([np.where(mask,subtract_background(image,detector_mask=~mask,config=config).background_subtracted,np.nan).astype(np.float32)
                     for image,mask in zip(flicker,valid)])
    two,diagnostic=subtract_sequence(flicker,valid,offsets,config,two_pass_config)
    stages={'raw':(raw,valid,offsets),'flicker':(flicker,valid,offsets),
            'background_single':(single,valid,offsets),'background_two_pass':(two,valid,offsets)}
    if inference is not None:
        aligned=[]; masks=[]
        for image,mask,offset in zip(two,valid,offsets):
            data,good,_,_=masked_shift(image,mask,offset)
            aligned.append(data); masks.append(good)
        inferred=inference(np.asarray(aligned,np.float32),np.asarray(masks,bool))
        if len(inferred)==3:
            input_coadd,coadd,good=inferred
            stages['paper_input_coadd']=(np.asarray(input_coadd,np.float32)[None],np.asarray(good,bool)[None],np.zeros((1,2)))
        else:
            coadd,good=inferred
        stages['asteris']=(np.asarray(coadd,np.float32)[None],np.asarray(good,bool)[None],np.zeros((1,2)))
    return stages,diagnostic


def _stage_exposures(product, detection_config):
    images,masks,offsets=product; exposures=[]
    for image,mask,offset in zip(images,masks,offsets):
        _,good,_,_,stars,noise=inspect_array(image,mask)
        psf,_=estimate_psf(image,good,stars,detection_config)
        rms,_=local_noise_map(image,good,noise,detection_config) if detection_config.local_noise else (noise,{})
        exposures.append(Exposure(image,good,rms,np.asarray(offset),1.,psf))
    return exposures


def _match_unique(points, detections, radius):
    """Distance-sorted one-to-one matching, only AFTER blind detections exist."""
    points=np.asarray(points,float).reshape(-1,2)
    detections=np.asarray(detections,float).reshape(-1,2)
    matches={}; used=set(); pairs=[]
    for i,p in enumerate(points):
        for j,q in enumerate(detections):
            distance=float(np.linalg.norm(p-q))
            if distance<=radius: pairs.append((distance,i,j))
    for distance,i,j in sorted(pairs):
        if i not in matches and j not in used: matches[i]=(j,distance); used.add(j)
    return matches


def sample_truth(valid, offsets, stars, config, rng):
    """Uniform and near-AUTOMATIC-bright-star positions, no reference catalog."""
    h,w=valid.shape[1:]; positions=[]; modes=[]
    for index in range(config.sources_per_trial):
        near=bool(index%2 and len(stars))
        for _ in range(3000):
            if near:
                source=stars[rng.integers(len(stars)),:2]
                angle=rng.uniform(0,2*np.pi); distance=rng.uniform(6,18)
                point=source+distance*np.array([np.cos(angle),np.sin(angle)])
            else: point=np.array([rng.uniform(config.edge,w-config.edge),rng.uniform(config.edge,h-config.edge)])
            x,y=point
            if not(config.edge<=x<w-config.edge and config.edge<=y<h-config.edge): continue
            if positions and np.min(np.linalg.norm(np.asarray(positions)-point,axis=1))<config.min_separation: continue
            usable=0
            for mask,offset in zip(valid,offsets):
                ix,iy=np.rint(point-np.asarray(offset)[::-1]).astype(int)
                usable+=int(0<=iy<h and 0<=ix<w and mask[iy,ix])
            if usable>=np.ceil(.8*len(valid)): break
        else: raise ValueError('Insufficient space/support for requested injection sample; reduce count explicitly')
        positions.append(point); modes.append('near_bright' if near else 'uniform')
    table=pd.DataFrame(positions,columns=['x','y']); table['environment']=modes
    table['nearest_bright_distance']=[float(np.min(np.linalg.norm(stars[:,:2]-p,axis=1))) if len(stars) else np.nan for p in positions]
    return table


def inject_native(raw, valid, offsets, psfs, truth, rng, gain=None):
    out=np.asarray(raw,np.float32).copy()
    for index,(mask,offset,psf) in enumerate(zip(valid,offsets,psfs)):
        for row in truth.itertuples():
            slices,kernel=source_patch(mask.shape,psf,row.x-offset[1],row.y-offset[0])
            signal=row.flux*kernel
            if gain is not None: signal=rng.poisson(signal*gain)/gain
            # Invalid native samples remain invalid, never filled by the injection.
            out[index][slices]+=np.where(mask[slices],signal,0).astype(np.float32)
    return out


def evaluate_stage_chain(raw, valid, offsets, *, recovery_config=None, detection_config=None,
                         background_config=None, two_pass_config=None, inference=None, progress=None):
    recovery=recovery_config or RecoveryConfig(); recovery.validate()
    detection=detection_config or DetectionConfig(); detection.validate()
    if len(raw)<2 or np.shape(raw)!=np.shape(valid) or np.shape(offsets)!=(len(raw),2):
        raise ValueError('Aligned metadata/masks and at least two native exposures required')
    if min(raw.shape[1:])<=2*recovery.edge: raise ValueError('Images too small for injection edge margin')
    raw=np.asarray(raw,np.float32); valid=np.asarray(valid,bool)&np.isfinite(raw)
    rng=np.random.default_rng(recovery.seed)
    started=time.perf_counter()
    def emit(message, rows=None, summaries=None):
        print(f'[{time.perf_counter()-started:.1f}s] {message}',flush=True)
        if progress is not None: progress(message,rows,summaries)
    emit('Baseline preprocessing started')
    baseline,background_diagnostic=process_stage_chain(raw,valid,offsets,background_config=background_config,
        two_pass_config=two_pass_config,inference=inference)
    emit('Baseline preprocessing complete; fitting image-only PSFs')
    reference=inspect_array(baseline['flicker'][0][0],valid[0])
    native_psfs=[estimate_psf(image,mask,inspect_array(image,mask)[4],detection)[0]
                 for image,mask in zip(baseline['flicker'][0],valid)]
    base_exposures={name:_stage_exposures(product,detection) for name,product in baseline.items()}
    # Thresholds are swept without looking at truth; minimum threshold controls discovery.
    detector=replace(detection,threshold=min(recovery.thresholds))
    base_results={}
    for name,exposures in base_exposures.items():
        emit(f'Baseline blind detection: {name}')
        base_results[name]=analyze_exposures(exposures,detector)
        emit(f'Baseline {name}: {len(base_results[name]["sources"])} candidates')
    rows=[]; summaries=[]
    for repeat in range(recovery.repeats):
        positions=sample_truth(valid,offsets,reference[4],recovery,rng)
        for flux in recovery.fluxes:
            truth=positions.assign(flux=flux)
            emit(f'Injection repeat={repeat} flux={flux:g}: preprocessing')
            injected=inject_native(raw,valid,offsets,native_psfs,truth,rng,recovery.gain_e_per_dn)
            products,_=process_stage_chain(injected,valid,offsets,background_config=background_config,
                two_pass_config=two_pass_config,inference=inference)
            for stage,product in products.items():
                emit(f'Injection repeat={repeat} flux={flux:g}: {stage} detection')
                injected_exposures=_stage_exposures(product,detection)
                result=analyze_exposures(injected_exposures,detector)
                # Paired transfer measurement uses baseline PSF/weights, never feeds them truth.
                delta=[Exposure(b.image-a.image,a.valid & b.valid,a.noise,a.offset,a.throughput,a.psf)
                       for a,b in zip(base_exposures[stage],injected_exposures)]
                response=[]
                for point in positions[['x','y']].to_numpy():
                    try: response.append(float(_fit_group(delta,point[None])[0][0]/flux))
                    except ValueError: response.append(np.nan)
                for threshold in recovery.thresholds:
                    sources=result['sources']; original=base_results[stage]['sources']
                    sources=sources.loc[sources.accepted.astype(bool) & (sources.snr_empirical>=threshold)]
                    original=original.loc[original.accepted.astype(bool) & (original.snr_empirical>=threshold)]
                    found=sources[['x','y']].to_numpy(); existing=original[['x','y']].to_numpy()
                    points=truth[['x','y']].to_numpy()
                    matched=_match_unique(points,found,recovery.match_radius)
                    preexisting=_match_unique(points,existing,recovery.match_radius)
                    associated=_match_unique(existing,found,recovery.match_radius)
                    used={j for j,_ in matched.values()}|{j for j,_ in associated.values()}
                    for i,row in enumerate(truth.itertuples()):
                        match=matched.get(i)
                        rows.append(dict(stage=stage,repeat=repeat,flux=flux,threshold=threshold,
                            x=row.x,y=row.y,environment=row.environment,nearest_bright_distance=row.nearest_bright_distance,
                            recovered=match is not None,preexisting_match=i in preexisting,
                            localization_error=match[1] if match else np.nan,paired_flux_response=response[i],
                            measured_flux=float(sources.iloc[match[0]].flux) if match else np.nan))
                    eligible=set(range(len(truth)))-set(preexisting)
                    summaries.append(dict(stage=stage,repeat=repeat,flux=flux,threshold=threshold,
                        injected=len(truth),eligible_new=len(eligible),recovered_new=len(eligible & set(matched)),
                        preexisting=len(preexisting),new_unmatched_candidates=len(found)-len(used),
                        median_flux_response=float(np.nanmedian(response))))
                emit(f'Completed repeat={repeat} flux={flux:g} stage={stage}',rows,summaries)
    return pd.DataFrame(rows),pd.DataFrame(summaries),dict(recovery=asdict(recovery),detection=asdict(detection),
        background=asdict(background_config or BackgroundConfig()),background_diagnostic=background_diagnostic,
        catalog_used=False,training_started=False,asteris_status='evaluated callback' if inference else 'skipped: no explicitly supplied compatible checkpoint',
        noise_assumption='Injected Poisson noise with supplied gain' if recovery.gain_e_per_dn else 'Background-dominated deterministic injections; no gain assumed',
        caveats=['new_unmatched_candidates is NOT a measured false-positive rate on real sky',
                 'paired_flux_response diagnoses transfer; it is not a blind detection',
                 'Do not select parameters from final test seeds or real catalog positions',
                 'Compare AST against paper_input_coadd to isolate the network; both use identical exposures and single-image detection',
                 'Native multi-exposure detection and coadd detection are different estimators; do not attribute their whole difference to the network'])


def select_threshold_from_null(null_peak_scores, thresholds, max_false_per_image):
    """Freeze a threshold from INDEPENDENT source-free validation simulations.

Inputs are peak score arrays from each null image after the FULL same pipeline,
not pixels, real-sky unmatched peaks, or final-test injection trials.
"""
    if not null_peak_scores or max_false_per_image<0: raise ValueError('Null validation runs and nonnegative false budget required')
    for threshold in sorted(thresholds):
        rate=np.mean([np.count_nonzero(np.asarray(scores)>=threshold) for scores in null_peak_scores])
        if rate<=max_false_per_image: return float(threshold),float(rate)
    raise ValueError('No threshold meets the validation false-positive budget')


def run_recovery_files(files,dataset_root,output_root,*,recovery_config=None,detection_config=None,
                       background_config=None,inference=None):
    files=[Path(p).resolve() for p in files]; output=Path(output_root).resolve()
    if output.exists(): raise FileExistsError('Recovery output must be a new experiment directory')
    if not files or len(set(files))!=len(files): raise ValueError('Unique explicit frozen exposure list required')
    if len({p.parent for p in files})!=1: raise ValueError('Evaluate one field/sequence at a time')
    if output.is_relative_to(Path(dataset_root).resolve()): raise ValueError('Recovery output cannot be inside raw data')
    if any(output.is_relative_to(p.parent) or p.is_relative_to(output) for p in files): raise ValueError('Recovery output overlaps inputs')
    detector=load_detector_mask(Path(dataset_root)/'盲点表'); images=[]; masks=[]; before={}
    for path in files:
        with path.open('rb') as stream: before[str(path)]=hashlib.file_digest(stream,'sha256').hexdigest()
        image,_=load_fits(path); valid=science_valid(path,image,detector)
        images.append(image.astype(np.float32)); masks.append(valid)
    reference=inspect_array(images[0],masks[0]); offsets=[]
    for image,mask in zip(images,masks):
        inspected=inspect_array(image,mask)
        offset,_,_=register_features(reference[3],reference[4],inspected[3],inspected[4]); offsets.append(offset)
    output.mkdir(parents=True)
    def progress(message, rows, summaries):
        (output/'progress.json').write_text(json.dumps(dict(message=message,updated_unix=time.time(),
            complete=False,input_files=[str(p) for p in files]),indent=2),encoding='utf-8')
        if rows is not None:
            pd.DataFrame(rows).to_csv(output/'partial_source_recovery.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(summaries).to_csv(output/'partial_stage_summary.csv',index=False,encoding='utf-8-sig')
    rows,summary,report=evaluate_stage_chain(np.stack(images),np.stack(masks),np.stack(offsets),
        recovery_config=recovery_config,detection_config=detection_config,background_config=background_config,inference=inference,
        progress=progress)
    for path in files:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=before[str(path)]: raise RuntimeError('Input FITS changed during evaluation')
    rows.to_csv(output/'source_recovery.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(output/'stage_summary.csv',index=False,encoding='utf-8-sig')
    report.update(input_sha256=before,offsets_yx=np.asarray(offsets).tolist())
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'progress.json').write_text(json.dumps(dict(message='Complete',complete=True,updated_unix=time.time())),encoding='utf-8')
    return report
