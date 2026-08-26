"""V3 runner: new-directory output, no training, native science remains read-only."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from astropy.io import fits

from astr_ir.dq import build_dq
from astr_ir.flicker.processor import load_detector_mask, robust_std
from astr_ir.registration import masked_shift
from .blind_joint import inspect_frame, register_features, detect_blind
from .weak_detection import (DetectionConfig, Exposure, estimate_psf, local_noise_map,
                             analyze_exposures, blank_correlations)


def run_sequence_v3(input_root,dataset_root,output_root,sequence,*,config=None,limit=None):
    config=config or DetectionConfig(); config.validate()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(sequence)): raise ValueError('Invalid sequence')
    input_root,dataset_root,output_root=map(lambda p:Path(p).resolve(),(input_root,dataset_root,output_root))
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root) or output_root.is_relative_to(dataset_root):
        raise ValueError('Output must not overlap input/raw paths')
    destination=output_root/sequence
    if destination.exists(): raise FileExistsError(f'Use a fresh experiment destination: {destination}')
    files=sorted((input_root/sequence).glob('background_subtracted_*.fits'))
    if limit is not None: files=files[:limit]
    if len(files)<2: raise ValueError('Joint detection requires >=2 frames')
    detector=load_detector_mask(dataset_root/'盲点表')
    reference=inspect_frame(files[0],detector); exposures=[]; diagnostics=[]
    for index,path in enumerate(files):
        image,valid,header,feature,stars,noise=reference if index==0 else inspect_frame(path,detector)
        offset,throughput,registration=register_features(reference[3],reference[4],feature,stars)
        psf,psf_info=estimate_psf(image,valid,stars,config)
        rms,noise_info=local_noise_map(image,valid,noise,config) if config.local_noise else (noise,{})
        scatter=registration['registration_rms']
        penalty=1+(scatter/max(psf_info['psf_sigma'],.7))**2 if np.isfinite(scatter) else 2.
        if psf_info['psf_fallback']: penalty*=1.5
        if registration['transparency_fallback']: penalty*=1.5
        # Conservative variance inflation for nuisance-model uncertainty; not an exact covariance model.
        rms=np.asarray(rms)*np.sqrt(penalty)
        exposures.append(Exposure(image.astype(np.float32),valid,rms,np.asarray(offset),throughput,psf))
        diagnostics.append(dict(filename=path.name,frame_index=index,alignment_dy=float(offset[0]),
            alignment_dx=float(offset[1]),noise=noise,throughput=throughput,variance_penalty=penalty,
            **registration,**psf_info,**noise_info))
    result=analyze_exposures(exposures,config)
    halves=[analyze_exposures(exposures[k::2],config,fit_sources=False) for k in (0,1)]
    null_var=sum(np.divide(1,h['information'],out=np.full_like(h['information'],np.inf),where=h['information']>0) for h in halves)
    null=(halves[0]['flux']-halves[1]['flux'])/np.sqrt(null_var)
    num=np.zeros_like(result['score']); den=np.zeros_like(num); count=np.zeros_like(num,dtype=np.uint16)
    for exposure in exposures:
        centered=exposure.image-np.median(exposure.image[exposure.valid])
        aligned,good,_,variance=masked_shift(centered,exposure.valid,exposure.offset,variance=exposure.noise**2)
        weight=np.divide(exposure.throughput**2,variance,out=np.zeros_like(den),where=good & (variance>0))
        num+=np.where(good,aligned/exposure.throughput,0)*weight; den+=weight; count+=good
    mean=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    sources=result['sources'].copy()
    for k,name in enumerate(('snr_even','snr_odd')):
        sources[name]=[float(halves[k]['score'][int(round(row.y)),int(round(row.x))]) for row in sources.itertuples()]
    sources['both_halves_ge3']=(sources.snr_even>=3)&(sources.snr_odd>=3)
    accepted=sources.loc[sources.accepted.astype(bool)]
    negative=detect_blind(-result['score'],result['valid'],config.threshold,config.min_distance)
    destination.mkdir(parents=True)
    sources.to_csv(destination/'all_candidates.csv',index=False,encoding='utf-8-sig')
    accepted.to_csv(destination/'blind_sources.csv',index=False,encoding='utf-8-sig')
    negative.to_csv(destination/'negative_peaks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(destination/'frame_diagnostics.csv',index=False,encoding='utf-8-sig')
    np.save(destination/'psf_kernels.npy',np.stack([e.psf for e in exposures]))
    report=dict(sequence=sequence,frames=len(files),catalog_used=False,training_started=False,
        method='V3 image-selected PSF + local noise + native joint fit / residual discovery',
        config=asdict(config),detections=len(accepted),candidates=len(sources),negative_peaks=len(negative),
        psf_fallback_frames=sum(d['psf_fallback'] for d in diagnostics),
        transparency_fallback_frames=sum(d['transparency_fallback'] for d in diagnostics),
        null_robust_std=robust_std(null[np.isfinite(null)]),
        input_files=[str(p) for p in files],**result['diagnostic'],
        correlations=blank_correlations(result['nominal'],result['valid']),
        caveat='Negative peaks, half consistency and local scores do not establish completeness or calibrated false-alarm probability. Validate on independent injections.')
    (destination/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    base=reference[2].copy(); base['NCOMBINE']=len(files); base['REFIMAGE']=files[0].name
    base['CATUSED']=False; base['HIERARCH JOINT METHOD']='weak-source-v3'; base['HIERARCH JOINT WHITEN']=False
    arrays=dict(weighted_coadd=mean,joint_score=result['score'],joint_nominal_score=result['nominal'],
        joint_flux=result['flux'],odd_even_null=null,local_score_scale=result['scale'],local_score_center=result['center'])
    for name,array in arrays.items():
        mask=np.isfinite(array) & ((count>0) if name=='weighted_coadd' else result['valid'])
        coverage=count if name=='weighted_coadd' else result['coverage']
        header=base.copy(); header['BUNIT']='DN' if name in {'weighted_coadd','joint_flux'} else 'dimensionless'
        header.add_history('Experimental display/detection products; no catalog used; no interpolation-filled pixels.')
        dq=build_dq(array.shape,no_coverage=~mask,partial_coverage=mask & (coverage<len(files)))
        fits.HDUList([fits.PrimaryHDU(np.where(mask,array,np.nan).astype(np.float32),header),
            fits.ImageHDU(dq,name='DQ'),fits.ImageHDU(coverage.astype(np.uint16),name='COVERAGE'),
            fits.ImageHDU(result['information'].astype(np.float32),name='INFORMATION')]).writeto(destination/f'{name}_{sequence}.fits',overwrite=False)
    return report
