from dataclasses import replace
import inspect
import numpy as np
import pandas as pd
import pytest

from astr_ir.evaluation.stage_recovery import (RecoveryConfig, process_stage_chain,
    inject_native, _match_unique, select_threshold_from_null, evaluate_stage_chain, sample_truth)
from astr_ir.evaluation.weak_detection import DetectionConfig, Exposure, analytic_psf, render_sources
from astr_ir.background.processor import BackgroundConfig


def test_injections_are_native_masked_reproducible_and_do_not_mutate_raw():
    raw=np.zeros((2,80,80),np.float32); valid=np.ones_like(raw,bool)
    valid[:,40,40]=False; raw[:,40,40]=np.nan
    offsets=np.array([[0,0],[.5,-.25]])
    truth=pd.DataFrame(dict(x=[40.],y=[40.],flux=[100.]))
    kernels=[analytic_psf(25,1.5)]*2
    out=inject_native(raw,valid,offsets,kernels,truth,np.random.default_rng(1))
    assert np.isnan(out[:,40,40]).all() and np.nansum(raw)==0
    assert 0<np.nansum(out)<200
    a=inject_native(raw,valid,offsets,kernels,truth,np.random.default_rng(7),gain=2)
    b=inject_native(raw,valid,offsets,kernels,truth,np.random.default_rng(7),gain=2)
    assert np.array_equal(a,b,equal_nan=True)


def test_matching_is_one_to_one_and_null_budget_is_independent():
    matches=_match_unique([[1,1],[1.2,1]],[[1.1,1]],2.5)
    assert len(matches)==1
    threshold,rate=select_threshold_from_null([[3,4,6],[3,4.5]],(4,5,6,7),.5)
    assert threshold==5 and rate==.5
    with pytest.raises(ValueError): select_threshold_from_null([[100]],(4,5),0)
    with pytest.raises(ValueError): select_threshold_from_null([], (4,5),1)


def test_truth_is_not_a_processing_argument():
    assert set(inspect.signature(process_stage_chain).parameters)=={
        'raw','valid','offsets','background_config','two_pass_config','inference'}


def test_stage_recovery_end_to_end_without_training_or_catalog(monkeypatch):
    # Tiny deterministic trial exercises both actual preprocessors and detector.
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('network is forbidden'))
    shape=(144,144); yy,xx=np.indices(shape); rng=np.random.default_rng(11)
    images=[]
    for _ in range(2):
        exposure=Exposure(np.zeros(shape),np.ones(shape,bool),2.,np.zeros(2),1.,analytic_psf(25,1.5))
        stars=render_sources(exposure,[[43,43],[100,43],[70,101]],[6000,5000,7000])
        images.append(20000+.3*xx+.2*yy+4*np.sin(yy/5)+rng.normal(0,2,shape)+stars)
    raw=np.asarray(images,np.float32); before=raw.copy(); valid=np.ones_like(raw,bool)
    recovery=RecoveryConfig(sources_per_trial=1,fluxes=(120.,),thresholds=(4.,6.),repeats=1,edge=32)
    detection=DetectionConfig(psf_mode='gaussian',psf_size=25,iterations=1,max_sources=12,noise_box=32)
    background=BackgroundConfig(edge_width=8,rough_box_size=32,ring_inner_radius=20,final_box_size=32,
        final_filter_size=1,validation_block_size=32)
    rows,summary,report=evaluate_stage_chain(raw,valid,np.zeros((2,2)),recovery_config=recovery,
        detection_config=detection,background_config=background)
    assert set(rows.stage)=={'raw','flicker','background_single','background_two_pass'}
    assert len(summary)==8 and len(rows)==8
    assert not report['training_started'] and not report['catalog_used']
    assert report['asteris_status'].startswith('skipped')
    assert np.array_equal(raw,before)
    assert np.isfinite(rows.paired_flux_response).all()
    assert rows.loc[rows.stage.eq('raw'),'paired_flux_response'].between(.9,1.1).all()


def test_inference_callback_is_optional_eval_stage_and_receives_no_truth(monkeypatch):
    import astr_ir.evaluation.stage_recovery as module
    shape=(2,64,64); raw=np.ones(shape,np.float32); valid=np.ones(shape,bool)
    class Flicker:
        corrected=np.ones(shape[1:],np.float32)
    class Background:
        background_subtracted=np.ones(shape[1:],np.float32)
    monkeypatch.setattr(module,'correct_flicker',lambda *a,**k:Flicker())
    monkeypatch.setattr(module,'subtract_background',lambda *a,**k:Background())
    monkeypatch.setattr(module,'subtract_sequence',lambda *a,**k:(raw,{}))
    calls=[]
    def inference(images,masks):
        calls.append((images.shape,masks.shape))
        return images.mean(axis=0),masks.any(axis=0)
    stages,_=process_stage_chain(raw,valid,np.zeros((2,2)),inference=inference)
    assert calls==[(shape,shape)] and stages['asteris'][0].shape==(1,64,64)
    def paired_inference(images,masks):
        return images.mean(axis=0),images.mean(axis=0)*2,masks.any(axis=0)
    paired,_=process_stage_chain(raw,valid,np.zeros((2,2)),inference=paired_inference)
    assert np.allclose(paired['asteris'][0],2*paired['paper_input_coadd'][0],equal_nan=True)
    assert np.array_equal(paired['asteris'][1],paired['paper_input_coadd'][1])
