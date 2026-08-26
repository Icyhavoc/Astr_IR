from dataclasses import replace
import numpy as np
import pytest

from astr_ir.evaluation.weak_detection import (DetectionConfig, Exposure, analytic_psf,
    shifted_kernel, source_patch, render_sources, estimate_psf, local_calibration,
    local_noise_map, fit_native_sources, analyze_exposures, _small_inverse)
from astr_ir.evaluation.blind_joint import exposure_statistic


def make_exposure(shape=(112,112),seed=0,offset=(0.,0.),psf=None):
    rng=np.random.default_rng(seed)
    return Exposure(rng.normal(0,1,shape),np.ones(shape,bool),1.,np.asarray(offset),1.,
                    analytic_psf(25,1.6) if psf is None else psf)


def test_psf_native_fractional_flux_with_invalid_center():
    exposure=make_exposure(offset=(.3,-.7),psf=analytic_psf(25,1.3,2.,.4))
    model=render_sources(exposure,[[56,56]],[1000.])
    exposure.valid[55,56]=False; model[55,56]=1e20
    n,d,good=exposure_statistic(model+100,exposure.valid,1.,2.5,exposure.offset,psf_kernel=exposure.psf)
    assert good[56,56]
    assert n[56,56]/d[56,56]==pytest.approx(1000,rel=.015)


def test_psf_selection_uses_held_out_automatic_stars():
    exposure=make_exposure((256,256),seed=1)
    positions=np.array([[40,40],[110,40],[180,40],[40,110],[110,110],[180,110],[40,180],[110,180],[180,180]],float)
    true=analytic_psf(25,1.3,2.5,.35)
    exposure.psf=true
    image=exposure.image+render_sources(exposure,positions,np.full(len(positions),12000.))
    stars=np.column_stack((positions,np.full(len(positions),1.9),np.full(len(positions),12000.)))
    chosen,info=estimate_psf(image,exposure.valid,stars)
    assert info['psf_samples']>=8
    assert info['psf_mode'] in {'elliptical','moffat','empirical'}
    assert info['psf_validation_error']<info['psf_circular_error']*.7
    assert chosen.sum()==pytest.approx(1.) and np.all(chosen>=0)
    _,fallback=estimate_psf(image,exposure.valid,stars[:1])
    assert fallback['psf_fallback']


def test_local_noise_and_score_do_not_promote_missing_pixels():
    rng=np.random.default_rng(2); image=rng.normal(size=(256,256))
    image[:,128:]*=4
    valid=np.ones_like(image,bool); valid[:8]=False; image[:8]=1e30
    config=DetectionConfig(noise_box=32,min_noise_pixels=128)
    rms,info=local_noise_map(image,valid,2.,config)
    assert not info['noise_mesh_fallback']
    assert np.median(rms[40:200,20:80])==pytest.approx(1.,rel=.2)
    assert np.median(rms[40:200,180:230])==pytest.approx(4.,rel=.2)
    score,center,scale,_=local_calibration(image,valid,config)
    assert np.isnan(score[:8]).all() and np.all(scale>=1)
    assert .8<np.std(score[40:200,180:230])<1.2


def test_group_fit_uses_original_native_pixels_and_separates_neighbor():
    positions=np.array([[51.2,55.1],[56.4,55.6]])
    flux=np.array([2200.,300.]); exposures=[]
    for i,offset in enumerate(((0.,0.),(.3,-.4),(-.4,.2))):
        exposure=make_exposure(seed=i,offset=offset)
        exposure.image+=100+render_sources(exposure,positions,flux)
        exposure.valid[50,50]=False; exposure.image[50,50]=1e20
        exposures.append(exposure)
    fitted,values,errors,flags=fit_native_sources(exposures,positions,DetectionConfig())
    assert all(flags=='ok') and np.all(errors>0)
    assert np.allclose(values,flux,rtol=.07)
    assert np.max(np.abs(fitted-positions))<=.5


def test_residual_discovery_finds_faint_bright_neighbor_without_truth_input():
    exposure=make_exposure(seed=4)
    truth=np.array([[51.,55.],[56.,55.]])
    exposure.image+=render_sources(exposure,truth,[4000.,260.])
    original=exposure.image.copy()
    result=analyze_exposures([exposure],DetectionConfig(iterations=3,threshold=5.,noise_box=32))
    found=result['sources'].loc[result['sources'].accepted,['x','y']].to_numpy()
    assert all(np.any(np.linalg.norm(found-point,axis=1)<1.5) for point in truth)
    assert len(result['diagnostic']['rounds'])>=2
    assert np.array_equal(original,exposure.image)


def test_invalid_values_cannot_affect_new_detection_and_inputs_unchanged():
    exposure=make_exposure(seed=6)
    exposure.valid[40:42,40]=False
    other=replace(exposure,image=exposure.image.copy())
    other.image[~other.valid]=1e25
    config=DetectionConfig(iterations=1)
    a=analyze_exposures([exposure],config,fit_sources=False)
    b=analyze_exposures([other],config,fit_sources=False)
    assert np.array_equal(a['score'],b['score'],equal_nan=True)


def test_degenerate_covariance_is_still_rejected_without_size_rejection():
    with pytest.raises(ValueError): _small_inverse(np.ones((2,2)))
    exposure=make_exposure()
    _,flux,error,flags=fit_native_sources([exposure],[[50,50],[50,50]],DetectionConfig(max_group=1))
    assert all(flags=='degenerate_group') and np.all(flux==0) and np.all(np.isinf(error))
    with pytest.raises(ValueError): shifted_kernel(np.ones((4,4)))


def test_sparse_system_matches_dense_fit_with_masks_offsets_and_transparency():
    from astr_ir.evaluation.crowded_fit import sparse_group_system,solve_group_system
    from astr_ir.evaluation.weak_detection import _fit_group
    positions=np.array([[40.2,43.1],[45.3,45.7],[61.8,48.4]])
    exposures=[]; bounds=[]
    for index,offset in enumerate(((0.,0.),(.4,-.25))):
        exposure=make_exposure(seed=40+index,offset=offset)
        exposure.throughput=1+index*.2
        exposure.noise=np.full(exposure.image.shape,1+index*.5)
        exposure.image+=1000+render_sources(exposure,positions,[1200,230,570])
        exposure.valid[40:43,44]=False; exposure.image[~exposure.valid]=1e25
        native=positions-np.asarray(offset)[::-1]
        bounds.append((np.floor(native.min(axis=0)).astype(int)-12,
                       np.ceil(native.max(axis=0)).astype(int)+13))
        exposures.append(exposure)
    dense=_fit_group(exposures,positions)
    sparse=solve_group_system(*sparse_group_system(exposures,positions,bounds,None))
    for a,b in zip(dense,sparse): assert np.allclose(a,b,rtol=1e-9,atol=1e-8)


def test_twenty_source_chain_is_jointly_fitted_with_local_background_and_dq():
    yy,xx=np.mgrid[:4,:5]
    positions=np.column_stack((28+18*xx.ravel(),28+18*yy.ravel())).astype(float)
    truth_flux=np.linspace(350,1300,len(positions)); truth_flux[0]=12000
    exposures=[]; originals=[]
    for index,offset in enumerate(((0.,0.),(.3,-.2))):
        exposure=make_exposure((128,144),seed=50+index,offset=offset)
        y,x=np.indices(exposure.image.shape)
        exposure.image+=1000+40*(x//64)-30*(y//64)+render_sources(exposure,positions,truth_flux)
        exposure.valid[30:33,32]=False; exposure.image[~exposure.valid]=1e25
        exposures.append(exposure); originals.append(exposure.image.copy())
    fitted,flux,error,flags,details=fit_native_sources(exposures,positions,DetectionConfig(),return_diagnostics=True)
    assert all(flags=='ok') and np.all(np.isfinite(error)) and np.all(error>0)
    assert np.allclose(flux,truth_flux,rtol=.06)
    assert np.max(np.abs(fitted-positions))<=.75
    assert all(d['group_size']==20 and d['fit_method']=='sparse_joint_local_background' for d in details)
    assert all(np.array_equal(e.image,before) for e,before in zip(exposures,originals))
    other=[replace(e,image=np.where(e.valid,e.image,np.nan)) for e in exposures]
    repeated=fit_native_sources(other,positions,DetectionConfig())
    assert np.allclose(repeated[1],flux) and np.allclose(repeated[2],error)


def test_crowded_errors_include_neighbor_covariance_and_negative_flux_is_not_a_detection():
    from astr_ir.evaluation.crowded_fit import group_bounds,sparse_group_system
    exposure=make_exposure(seed=61)
    positions=np.array([[49.,55.],[52.,55.],[70.,55.]])
    exposure.image=100+render_sources(exposure,positions,[2000.,350.,-250.])
    fitted,flux,error,flags=fit_native_sources([exposure],positions,DetectionConfig(max_group=1))
    normal,_=sparse_group_system([exposure],fitted,group_bounds([exposure],positions),64)
    conditional=1/np.sqrt(np.diag(normal))
    assert np.all(error[:2]>1.05*conditional[:2])
    assert np.allclose(flux,[2000,350,-250],rtol=.005)
    assert list(flags)==['ok','ok','nonpositive_flux']


def test_blind_detection_no_longer_drops_large_connected_source_groups():
    exposure=make_exposure((176,176),seed=70)
    y,x=np.mgrid[:4,:4]
    truth=np.column_stack((40+20*x.ravel(),40+20*y.ravel())).astype(float)
    exposure.image+=render_sources(exposure,truth,np.linspace(350,1500,len(truth)))
    result=analyze_exposures([exposure],DetectionConfig(iterations=2,noise_box=32))
    candidates=result['sources']; found=candidates.loc[candidates.accepted,['x','y']].to_numpy()
    assert all(np.any(np.linalg.norm(found-point,axis=1)<1.5) for point in truth)
    assert not candidates.fit_flag.eq('group_too_large').any()
    assert result['diagnostic']['crowded_candidates']>=16
    assert result['diagnostic']['largest_fitted_group']>=16


def test_crowded_position_proposals_improve_full_objective_and_not_conditional_errors():
    from astr_ir.evaluation.crowded_fit import group_bounds,sparse_group_system,solve_group_system
    exposure=make_exposure((144,144),seed=76)
    y,x=np.mgrid[:4,:4]
    initial=np.column_stack((35+18*x.ravel(),35+18*y.ravel())).astype(float)
    truth=initial+np.array([.35,-.3])
    exposure.image=1000+render_sources(exposure,truth,np.linspace(600,2000,len(truth)))
    bounds=group_bounds([exposure],initial)
    before=solve_group_system(*sparse_group_system([exposure],initial,bounds,64))
    fitted,flux,error,flags=fit_native_sources([exposure],initial,DetectionConfig())
    after=solve_group_system(*sparse_group_system([exposure],fitted,bounds,64))
    assert all(flags=='ok') and after[2]>=before[2]
    assert np.mean(np.linalg.norm(fitted-truth,axis=1))<.25
    assert np.allclose(flux,after[0]) and np.allclose(error,after[1])


def test_v3_runner_writes_new_products_preserves_fits_and_refuses_overwrite(tmp_path):
    from astropy.io import fits
    from PIL import Image
    from astr_ir.evaluation.weak_joint_pipeline import run_sequence_v3
    input_root=tmp_path/'background'; sequence=input_root/'s'; sequence.mkdir(parents=True)
    dataset=tmp_path/'raw'; blind=dataset/'盲点表'; blind.mkdir(parents=True)
    shape=(128,128)
    for name in ('DeadBlindMap.tiff','NoiseBlindMap.tiff'):
        Image.fromarray(np.zeros(shape,np.uint8)).save(blind/name)
    before={}
    for index in range(2):
        exposure=make_exposure(shape,seed=30+index,offset=(index*.1,-index*.2))
        image=exposure.image+render_sources(exposure,[[39,40],[86,47],[68,87]],[6000,5000,7000])
        path=sequence/f'background_subtracted_{index:03d}.fits'
        fits.writeto(path,image.astype(np.float32)); before[path]=path.read_bytes()
    output=tmp_path/'v3'
    result=run_sequence_v3(input_root,dataset,output,'s',config=DetectionConfig(iterations=1,max_sources=12))
    assert not result['training_started'] and not result['catalog_used']
    assert len(list((output/'s').glob('*.fits')))==7
    assert all(path.read_bytes()==contents for path,contents in before.items())
    with fits.open(output/'s/joint_score_s.fits') as hdul:
        assert 'DQ' in hdul and 'COVERAGE' in hdul and 'INFORMATION' in hdul
        assert hdul[0].header['NCOMBINE']==2
    with pytest.raises(FileExistsError): run_sequence_v3(input_root,dataset,output,'s')
