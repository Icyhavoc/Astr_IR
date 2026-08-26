"""Two-pass background estimation from image-only multi-exposure source masks.

The second pass refits ORIGINAL flicker-corrected frames, never subtracts a
second background from pass-one science. No catalog/target table is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label, shift

from .processor import BackgroundConfig, subtract_background, write_fits_products, neighbor_difference_noise, raw_filename_from_corrected
from astr_ir.flicker.processor import load_fits, load_detector_mask, robust_std
from astr_ir.registration import masked_gaussian, masked_shift, science_valid


@dataclass(frozen=True)
class TwoPassConfig:
    threshold: float = 4.5
    smooth_sigma: float = 1.5
    dilation_radius: int = 6
    min_pixels: int = 3
    min_coverage: float = .8
    max_source_fraction: float = .4

    def validate(self):
        if self.threshold<=0 or self.smooth_sigma<=0 or self.dilation_radius<0 or self.min_pixels<1:
            raise ValueError('Invalid image-mask parameters')
        if not 0<self.min_coverage<=1 or not 0<self.max_source_fraction<1:
            raise ValueError('Invalid coverage/mask fraction')


def coadd_source_mask(images, masks, offsets, background_config, config=None):
    """Consume re-iterable frames, make ONE bounded image-derived mask."""
    config=config or TwoPassConfig(); config.validate()
    shape=images[0].shape
    numerator=np.zeros(shape); weight=np.zeros(shape); coverage=np.zeros(shape,int)
    for image,valid,offset in zip(images,masks,offsets):
        result=subtract_background(image,detector_mask=~valid,config=background_config)
        good=valid & ~result.edge_mask
        noise=neighbor_difference_noise(result.background_subtracted,~good)
        if not np.isfinite(noise) or noise<=0: raise ValueError('No usable first-pass noise estimate')
        aligned,support,_,variance=masked_shift(result.background_subtracted,good,offset,variance=noise**2)
        ivar=np.divide(1,variance,out=np.zeros(shape),where=support & (variance>0))
        numerator+=np.where(support,aligned,0)*ivar; weight+=ivar; coverage+=support
    allowed=(coverage>=np.ceil(config.min_coverage*len(images))) & (weight>0)
    coadd=np.divide(numerator,weight,out=np.full(shape,np.nan),where=allowed)
    # Discovery only: suppress broad residuals before searching for compact
    # sources. This high-pass image is NEVER subtracted from science products.
    residual=coadd-masked_gaussian(coadd,allowed,max(16.,8*config.smooth_sigma))
    smooth=masked_gaussian(residual,allowed,config.smooth_sigma)
    values=smooth[allowed & np.isfinite(smooth)]
    if len(values)<64: raise ValueError('Too little common support for two-pass background')
    level=np.median(values); scale=robust_std(values)
    seeds=allowed & np.isfinite(smooth) & (smooth>level+config.threshold*max(scale,1e-12))
    labels,_=label(seeds); sizes=np.bincount(labels.ravel())
    keep=sizes>=config.min_pixels; keep[0]=False
    mask=keep[labels]
    if config.dilation_radius:
        yy,xx=np.ogrid[-config.dilation_radius:config.dilation_radius+1,-config.dilation_radius:config.dilation_radius+1]
        mask=binary_dilation(mask,structure=xx**2+yy**2<=config.dilation_radius**2)
    mask &= allowed
    fraction=float(mask.sum()/max(allowed.sum(),1))
    accepted=fraction<=config.max_source_fraction
    if not accepted: mask[:]=False
    return mask,dict(mask_fraction=fraction,mask_accepted=accepted,passes=2,
        reason='accepted' if accepted else 'mask fraction exceeds cap; second pass uses single-frame masks',
        catalog_used=False,mask_iterations=1,reference='first exposure',config=asdict(config))


def native_source_mask(reference_mask, offset):
    return shift(reference_mask.astype(float),-np.asarray(offset),order=1,mode='constant',cval=0,prefilter=False)>0


def subtract_sequence(images, valid, offsets, background_config=None, config=None):
    """In-memory two-pass API used by paired stage evaluation."""
    background_config=background_config or BackgroundConfig()
    if len(images)<2 or len(images)!=len(valid) or np.shape(offsets)!=(len(images),2):
        raise ValueError('At least two frames with matching masks and offsets are required')
    reference_mask,diagnostic=coadd_source_mask(images,valid,offsets,background_config,config)
    output=[]
    for image,mask,offset in zip(images,valid,offsets):
        result=subtract_background(image,detector_mask=~mask,config=background_config,
            image_source_mask=native_source_mask(reference_mask,offset))
        output.append(np.where(mask,result.background_subtracted,np.nan).astype(np.float32))
    return np.stack(output),diagnostic


class _FitsFrames:
    """Re-iterable, lazy frame/mask access: do not hold 80 BackgroundResults in RAM."""
    def __init__(self, files, detector, masks=False): self.files,self.detector,self.masks=files,detector,masks
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        path=self.files[index]; image,_=load_fits(path)
        return science_valid(path,image,self.detector) if self.masks else image


def run_two_pass_batch(input_root,dataset_root,output_root,*,sequences=None,background_config=None,
                       config=None,limit_per_sequence=None,split_manifest=None):
    """Write a NEW experiment only. Existing output directories are refused."""
    from astr_ir.evaluation.blind_joint import inspect_frame, register_features
    input_root,dataset_root,output_root=map(lambda p:Path(p).resolve(),(input_root,dataset_root,output_root))
    if output_root.exists(): raise FileExistsError(f'Use a fresh two-pass output directory: {output_root}')
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root) or output_root.is_relative_to(dataset_root):
        raise ValueError('Output must not overlap science input/raw paths')
    background_config=background_config or BackgroundConfig(); background_config.validate()
    config=config or TwoPassConfig(); config.validate()
    frozen=None
    if split_manifest is not None:
        frozen=pd.read_csv(split_manifest,usecols=['sequence','filename','split'],dtype={'sequence':str})
        if frozen.duplicated(['sequence','filename']).any(): raise ValueError('Duplicate frozen split entries')
    sequences=sequences or [p.name for p in sorted(input_root.iterdir()) if p.is_dir() and any(p.glob('flicker_corrected_*.fits'))]
    detector=load_detector_mask(dataset_root/'盲点表'); rows=[]
    for sequence in sequences:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',str(sequence)): raise ValueError('Invalid sequence')
        files=sorted((input_root/sequence).glob('flicker_corrected_*.fits'))
        if limit_per_sequence is not None: files=files[:limit_per_sequence]
        if len(files)<2: raise ValueError('Two-pass background requires at least two exposures')
        reference=inspect_frame(files[0],detector); offsets=[]; registration=[]
        for path in files:
            frame=reference if path==files[0] else inspect_frame(path,detector)
            offset,_,diag=register_features(reference[3],reference[4],frame[3],frame[4])
            offsets.append(offset); registration.append(diag)
        labels=['all']*len(files)
        if frozen is not None:
            mapping=frozen.loc[frozen.sequence.eq(sequence)].set_index('filename')['split']
            labels=[str(mapping.loc[raw_filename_from_corrected(p.name)]) for p in files]
            if any(name not in {'train','validation','val','test','guard'} for name in labels):
                raise ValueError('Unknown frozen split label')
        source_masks={}; split_diagnostics={}
        for split_name in dict.fromkeys(labels):
            indices=[i for i,name in enumerate(labels) if name==split_name]
            if len(indices)<2: raise ValueError('Each mask-generation split needs at least two exposures')
            split_files=[files[i] for i in indices]
            images=_FitsFrames(split_files,detector); masks=_FitsFrames(split_files,detector,True)
            source_masks[split_name],split_diagnostics[split_name]=coadd_source_mask(
                images,masks,[offsets[i] for i in indices],background_config,config)
            split_diagnostics[split_name]['input_files']=[str(p) for p in split_files]
            print(f'two-pass {sequence}: mask {split_name}, {len(indices)} exposures',flush=True)
        diagnostic=dict(by_split=split_diagnostics,catalog_used=False,
            split_isolated=frozen is not None,split_manifest=str(split_manifest) if split_manifest else None)
        destination=output_root/sequence; destination.mkdir(parents=True)
        for index,(path,offset) in enumerate(zip(files,offsets)):
            image,header=load_fits(path); valid=science_valid(path,image,detector)
            result=subtract_background(image,detector_mask=~valid,config=background_config,
                image_source_mask=native_source_mask(source_masks[labels[index]],offset))
            header['HIERARCH BKG PASSES']=2; header['CATUSED']=False
            header['HIERARCH BKG MASKSPL']=labels[index]
            science,model,error=write_fits_products(path,destination,header,result,background_config,overwrite=False)
            rows.append(dict(sequence=sequence,sequence_frame_index=index+1,input_filename=path.name,
                subtracted_path=science.relative_to(output_root).as_posix(),model_path=model.relative_to(output_root).as_posix(),
                equation_max_abs_error_float32=error,alignment_dy=float(offset[0]),alignment_dx=float(offset[1]),
                **result.metrics))
            if (index+1)%10==0: print(f'two-pass {sequence}: wrote {index+1}/{len(files)} exposures',flush=True)
        diagnostic.update(background_config=asdict(background_config),registration=registration,
            input_files=[str(p) for p in files])
        (destination/'two_pass_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    output_root.mkdir(parents=True,exist_ok=True)
    stats=pd.DataFrame(rows); stats.to_csv(output_root/'background_statistics.csv',index=False,encoding='utf-8-sig')
    return stats
