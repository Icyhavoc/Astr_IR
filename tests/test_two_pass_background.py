import numpy as np
import pytest
from astr_ir.background.processor import BackgroundConfig, subtract_background
from astr_ir.background.sequence import (TwoPassConfig, coadd_source_mask, native_source_mask,
                                         subtract_sequence, run_two_pass_batch)


def test_mask_projection_sign_and_no_wraparound():
    mask=np.zeros((40,40),bool); mask[20,20]=True
    native=native_source_mask(mask,(3,-2))
    assert native[17,22] and native.sum()==1
    assert not native_source_mask(mask,(100,0)).any()


def test_second_pass_refits_original_and_ignores_invalid_pixels():
    rng=np.random.default_rng(9); yy,xx=np.indices((128,128))
    images=np.array([1000+.8*xx+.2*yy+rng.normal(0,3,(128,128))+
        20*np.exp(-((xx-64)**2+(yy-64)**2)/(2*1.6**2)) for _ in range(4)])
    valid=np.ones_like(images,bool); valid[:,45,47]=False
    config=BackgroundConfig(edge_width=8,rough_box_size=32,ring_inner_radius=20,
        final_box_size=32,final_filter_size=1,validation_block_size=32)
    offsets=np.zeros((4,2))
    result,info=subtract_sequence(images,valid,offsets,config)
    mask,_=coadd_source_mask(images,valid,offsets,config)
    assert mask[64,64] and info['passes']==2 and not info['catalog_used']
    expected=subtract_background(images[0],~valid[0],config=config,image_source_mask=mask)
    assert np.allclose(result[0][valid[0]],expected.background_subtracted[valid[0]])
    corrupted=images.copy(); corrupted[~valid]=1e22
    changed,_=subtract_sequence(corrupted,valid,offsets,config)
    assert np.array_equal(changed,result,equal_nan=True)
    assert np.isnan(result[:,45,47]).all()


def test_mask_growth_is_capped_and_output_cannot_overwrite(tmp_path):
    with pytest.raises(FileExistsError): run_two_pass_batch(tmp_path/'in',tmp_path/'raw',tmp_path)
    with pytest.raises(ValueError): TwoPassConfig(max_source_fraction=2).validate()
    with pytest.raises(ValueError): subtract_sequence(np.zeros((1,32,32)),np.ones((1,32,32),bool),np.zeros((1,2)))


def test_two_pass_file_runner_preserves_inputs_and_science_equation(tmp_path):
    from astropy.io import fits
    from PIL import Image
    from astr_ir.evaluation.weak_detection import Exposure,analytic_psf,render_sources
    input_root=tmp_path/'flicker'; sequence=input_root/'s'; sequence.mkdir(parents=True)
    dataset=tmp_path/'raw'; blind=dataset/'盲点表'; blind.mkdir(parents=True)
    shape=(128,128); yy,xx=np.indices(shape); rng=np.random.default_rng(42)
    for name in ('DeadBlindMap.tiff','NoiseBlindMap.tiff'):
        Image.fromarray(np.zeros(shape,np.uint8)).save(blind/name)
    before={}
    for index in range(2):
        exposure=Exposure(np.zeros(shape),np.ones(shape,bool),2.,np.zeros(2),1.,analytic_psf(25,1.5))
        image=1000+.3*xx+.2*yy+rng.normal(0,2,shape)+render_sources(exposure,[[39,40],[86,47],[68,87]],[6000,5000,7000])
        path=sequence/f'flicker_corrected_{index:03d}.fits'
        fits.writeto(path,image.astype(np.float32)); before[path]=path.read_bytes()
    output=tmp_path/'two_pass'
    stats=run_two_pass_batch(input_root,dataset,output,background_config=BackgroundConfig(
        rough_box_size=32,ring_inner_radius=20,final_box_size=32,final_filter_size=1,validation_block_size=32))
    assert len(stats)==2 and (stats.equation_max_abs_error_float32==0).all()
    assert all(path.read_bytes()==contents for path,contents in before.items())
    for index in range(2):
        original=fits.getdata(sequence/f'flicker_corrected_{index:03d}.fits')
        model=fits.getdata(output/f's/background_model_{index:03d}.fits')
        corrected=fits.getdata(output/f's/background_subtracted_{index:03d}.fits')
        assert np.array_equal(original-model,corrected)
    with pytest.raises(FileExistsError): run_two_pass_batch(input_root,dataset,output)


def test_frozen_splits_generate_disjoint_source_masks(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from astropy.io import fits
    from PIL import Image
    import astr_ir.background.sequence as module
    shape=(96,96); incoming=tmp_path/'input/s'; incoming.mkdir(parents=True)
    raw=tmp_path/'raw/盲点表'; raw.mkdir(parents=True)
    for name in ('DeadBlindMap.tiff','NoiseBlindMap.tiff'):
        Image.fromarray(np.zeros(shape,np.uint8)).save(raw/name)
    yy,xx=np.indices(shape); rng=np.random.default_rng(90)
    rows=[]
    for index in range(4):
        image=1000+index*100+.2*xx+rng.normal(0,2,shape)
        fits.writeto(incoming/f'flicker_corrected_{index:03d}.fits',image.astype(np.float32))
        rows.append(dict(sequence='s',filename=f'{index:03d}.fits',split='train' if index<2 else 'test'))
    manifest=tmp_path/'split.csv'; pd.DataFrame(rows).to_csv(manifest,index=False)
    import astr_ir.evaluation.blind_joint as joint
    monkeypatch.setattr(joint,'inspect_frame',lambda p,d:(fits.getdata(p),np.ones(shape,bool),fits.Header(),None,np.empty((0,4)),2.))
    monkeypatch.setattr(joint,'register_features',lambda *a:(np.zeros(2),1.,{}))
    memberships=[]
    def coadd(images,masks,offsets,*args):
        memberships.append([round(float(np.median(images[i]))) for i in range(len(images))])
        return np.zeros(shape,bool),dict(catalog_used=False)
    monkeypatch.setattr(module,'coadd_source_mask',coadd)
    destination=tmp_path/'output'
    run_two_pass_batch(tmp_path/'input',tmp_path/'raw',destination,split_manifest=manifest,
        background_config=BackgroundConfig(edge_width=8,rough_box_size=24,ring_inner_radius=15,final_box_size=24,validation_block_size=24))
    assert len(memberships)==2 and all(len(group)==2 for group in memberships)
    assert max(memberships[0])<min(memberships[1])
    report=json.loads((destination/'s/two_pass_diagnostics.json').read_text())
    assert report['split_isolated'] and set(report['by_split'])=={'train','test'}
    train=set(report['by_split']['train']['input_files']); test=set(report['by_split']['test']['input_files'])
    assert not train & test
