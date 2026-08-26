import numpy as np
from scipy.signal import fftconvolve
from astr_ir.evaluation.blind_joint import gaussian_psf, exposure_statistic, empirical_score, detect_blind, register_features, fit_gaussian_pixels


def test_gaussian_fit_is_accurate_without_threaded_blas():
    yy,xx=np.mgrid[-8:9,-8:9]
    image=100+700*np.exp(-((xx-.3)**2+(yy+.7)**2)/(2*1.65**2))
    good=np.ones_like(image,bool); good[8,8]=False
    image[8,8]=1e20
    parameters,_=fit_gaussian_pixels(image,good)
    assert np.allclose(parameters[1:4],[.3,-.7,1.65],atol=.025)
    assert abs(parameters[0]-700)<5


def test_streaming_reader_waits_for_correct_completed_frame(tmp_path):
    import pandas as pd
    import pytest
    from astr_ir.evaluation.blind_joint import wait_for_upstream
    progress=tmp_path/'progress.csv'
    pd.DataFrame([dict(input_filename='flicker_corrected_a.fits',equation_max_abs_error_float32=0)]).to_csv(progress,index=False)
    wait_for_upstream(progress,'background_subtracted_a.fits',timeout=0)
    with pytest.raises(TimeoutError):
        wait_for_upstream(progress,'background_subtracted_b.fits',timeout=0)


def test_native_fractional_psf_recovers_flux_without_science_interpolation():
    shape=(128,128)
    yy,xx=np.indices(shape)
    offset=np.array([0.3,-0.7])
    sigma=1.5
    flux=100
    # Target is at (64,64) in the output grid and (64,64)-offset natively.
    star=np.exp(-((yy-(64-offset[0]))**2+(xx-(64-offset[1]))**2)/(2*sigma**2))
    star*=flux/star.sum()
    valid=np.ones(shape,bool)
    valid[63,64]=False
    star[63,64]=1e20
    n,d,good=exposure_statistic(star+1000,valid,2,sigma,offset)
    assert good[64,64]
    assert abs(n[64,64]/d[64,64]-flux)<0.02
    assert np.unravel_index(np.argmax(np.where(good,n/np.sqrt(np.maximum(d,1e-30)),0)),shape)==(64,64)


def test_joint_detection_recovers_random_weak_injections_and_noise_scale():
    rng=np.random.default_rng(22)
    shape=(192,192)
    sky=np.zeros(shape)
    points=rng.integers(25,167,size=(8,2))
    for y,x in points:
        sky[y,x]+=32
    star=fftconvolve(sky,gaussian_psf(1.4),mode="same")
    ns,ds=[],[]
    for _ in range(16):
        image=star+rng.normal(0,3,shape)
        n,d,good=exposure_statistic(image,np.ones(shape,bool),3,1.4)
        ns.append(n); ds.append(d)
    n,d=np.sum(ns,axis=0),np.sum(ds,axis=0)
    score,nominal,_,scale=empirical_score(n,d,good)
    detected=detect_blind(score,good)
    found=detected[["y","x"]].to_numpy()
    assert 0.9<scale<1.15
    assert sum(np.any(np.linalg.norm(found-p,axis=1)<3) for p in points)>=6
    assert len(detect_blind(-score,good))<=2


def test_blind_registration_uses_multiple_unique_stars():
    ref=np.zeros((96,96)); moving=np.zeros_like(ref)
    stars=np.array([[25,30,1.5,100],[60,65,1.5,110],[40,60,1.5,150]],float)
    current=stars.copy(); current[:,:2]+=[3,-2]; current[:,3]*=0.8
    for x,y,_,f in stars: ref[int(y),int(x)]=f
    for x,y,_,f in current: moving[int(y),int(x)]=f
    offset,factor,diag=register_features(ref,stars,moving,current)
    assert np.allclose(offset,[2,-3])
    assert np.isclose(factor,0.8)
    assert diag["matched_stars"]==3


def test_manifest_needs_no_measurement_csv_and_preserves_split(tmp_path):
    from astropy.io import fits
    from PIL import Image
    from astr_ir.noise2noise.dataset import build_split_manifest
    input_root=tmp_path/'background'; sequence=input_root/'s'; sequence.mkdir(parents=True)
    dataset=tmp_path/'raw'; blind=dataset/'盲点表'; blind.mkdir(parents=True)
    shape=(128,128)
    for name in ('DeadBlindMap.tiff','NoiseBlindMap.tiff'):
        Image.fromarray(np.zeros(shape,np.uint8)).save(blind/name)
    yy,xx=np.indices(shape)
    rng=np.random.default_rng(48)
    for index in range(24):
        image=rng.normal(0,2,shape)
        for x,y in ((39,40),(86,47),(68,87)):
            image+=250*np.exp(-((xx-x-.02*index)**2+(yy-y+.03*index)**2)/(2*1.5**2))
        fits.writeto(sequence/f'background_subtracted_{index:03d}.fits',image.astype(np.float32))
    manifest=build_split_manifest(input_root,dataset,tmp_path/'split.csv')
    assert len(manifest)==24
    assert not manifest.source_measurement_available.any()
    assert manifest.reference_x.isna().all()
    assert abs(manifest.alignment_dx.iloc[-1]+.46)<.1
    assert abs(manifest.alignment_dy.iloc[-1]-.69)<.1


def test_future_empirical_psf_ignores_catalog_snr_and_coordinates(tmp_path,monkeypatch):
    import pandas as pd
    from astropy.io import fits
    from astr_ir.evaluation.pipeline import _training_psf,_known_sources
    from astr_ir.evaluation.mock_sources import EvaluationConfig
    import astr_ir.evaluation.blind_joint as joint
    yy,xx=np.indices((128,128))
    rows=[]
    for index in range(4):
        image=1000*np.exp(-((xx-64)**2+(yy-64)**2)/(2*1.6**2))
        name=f'{index}.fits'; fits.writeto(tmp_path/name,image.astype(np.float32))
        rows.append(dict(frame_id=f's:{index}',sequence='s',split='train',product_path=name,input_snr=np.nan,track_x=2,track_y=2))
    monkeypatch.setattr(joint,'inspect_frame',lambda *args: (None,None,None,None,np.array([[64,64,1.6,10000]]),1.0))
    psf=_training_psf(pd.DataFrame(rows),tmp_path,lambda row,image: np.ones_like(image,bool),EvaluationConfig(),tmp_path)
    assert np.unravel_index(psf.argmax(),psf.shape)==(15,15)
    assert np.isclose(psf.sum(),1)
    assert _known_sources(pd.Series(dict(track_x=64,track_y=64)))==[]


def test_weak_injection_before_both_preprocessors_preserves_flux_response():
    from astr_ir.flicker.processor import correct_flicker
    from astr_ir.background.processor import subtract_background
    rng=np.random.default_rng(812)
    shape=(192,192); yy,xx=np.indices(shape)
    x,y=rng.integers(70,122,size=2)
    signal=np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.4**2))
    signal*=32/signal.sum()  # Single-exposure matched S/N is only about two.
    detector=np.zeros(shape,bool); detector[53,63]=True
    valid=~detector
    responses=[]
    for _ in range(4):
        raw=30000+.5*xx+.25*yy+8*np.sin(yy*2*np.pi/17)+rng.normal(0,3,shape)
        products=[]
        for image in (raw,raw+signal):
            corrected=correct_flicker(image,detector_mask=detector).corrected
            products.append(subtract_background(corrected,detector_mask=detector).background_subtracted)
        n0,d,_=exposure_statistic(products[0],valid,3,1.4)
        n1,_,_=exposure_statistic(products[1],valid,3,1.4)
        responses.append((n1[y,x]-n0[y,x])/d[y,x])
    assert .85<np.mean(responses)/32<1.1
