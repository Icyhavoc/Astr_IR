import json

from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from astr_ir.evaluation import catalog_visualization as viz


def test_transfer_keeps_fractional_catalog_positions_and_sign():
    table = pd.DataFrame(dict(x_pix_0based=[10.2,29.,np.nan],y_pix_0based=[12.7,25.,2]))
    image = np.ones((30,30)); image[14,10] = np.nan
    out = viz.transfer_positions(table,(1.3,-0.5),image)
    assert np.allclose(out.x_plot[:2],[9.7,28.5])
    assert np.allclose(out.y_plot[:2],[14.,26.3])
    assert out.in_frame.tolist() == [True,True,False]
    assert out.valid_at_position.tolist() == [False,True,False]
    assert table.x_pix_0based[0] == 10.2  # no mutation or peak snapping
    with pytest.raises(ValueError):
        viz.transfer_positions(table,(np.nan,0),image)


def test_display_excludes_dq_without_modifying_fits(tmp_path):
    path = tmp_path/'image.fits'
    image = np.ones((30,30),np.float32); image[8,9] = 1e10
    dq = np.zeros_like(image,dtype=np.uint16); dq[8,9] = 1
    dq[8,8] = 8  # PARTIAL_COVERAGE alone is not a reason to hide a valid coadd pixel.
    fits.HDUList([fits.PrimaryHDU(image),fits.ImageHDU(dq,name='DQ')]).writeto(path)
    before = path.read_bytes()
    display,_ = viz.read_display_image(path)
    assert np.isnan(display[8,9]) and display[8,8] == 1
    assert path.read_bytes() == before


def test_display_cmap_is_opaque_neutral_gray_without_global_mutation():
    original_bad = matplotlib.colormaps['gray'].get_bad().copy()
    assert np.allclose(viz._display_cmap().get_bad(), [128/255,128/255,128/255,1])
    assert np.array_equal(matplotlib.colormaps['gray'].get_bad(), original_bad)


def test_cosmetic_fill_only_small_enclosed_holes_and_never_changes_validity():
    image = np.full((40,40),7.)
    image[8,8] = np.nan
    image[15:17,15:17] = np.nan
    image[:2,5] = np.nan  # edge-connected; do not extrapolate
    image[25:30,25:30] = np.nan  # larger than 16 pixels
    before = image.copy()
    filled,mask = viz.interpolate_display_holes(image)
    valid = np.isfinite(image)
    assert mask.sum() == 5 and np.allclose(filled[mask],7.)
    assert np.array_equal(filled[valid],image[valid])
    assert np.isnan(filled[:2,5]).all() and np.isnan(filled[25:30,25:30]).all()
    assert np.array_equal(image,before,equal_nan=True)
    points = pd.DataFrame(dict(x_pix_0based=[8.],y_pix_0based=[8.]))
    assert not viz.transfer_positions(points,(0,0),image).valid_at_position.iloc[0]
    assert np.isfinite(filled[8,8])  # does not make the ORIGINAL position valid
    empty,mask = viz.interpolate_display_holes(np.full((8,8),np.nan))
    assert np.isnan(empty).all() and not mask.any()
    for bad_size in (0,-1,1.5):
        with pytest.raises(ValueError,match='positive integer'):
            viz.interpolate_display_holes(image,max_hole_pixels=bad_size)


def test_cosmetic_fill_uses_original_neighbors_not_newly_filled_pixels():
    image = np.arange(400,dtype=float).reshape(20,20)
    image[8:10,8:10] = np.nan
    shown,fill = viz.interpolate_display_holes(image)
    for y,x in np.argwhere(fill):
        neighbors = image[y-3:y+4,x-3:x+4]
        yy,xx = np.mgrid[-3:4,-3:4]
        weights = np.exp(-(xx**2+yy**2)/2)
        valid = np.isfinite(neighbors)
        expected = np.sum(neighbors[valid]*weights[valid])/weights[valid].sum()
        assert shown[y,x] == pytest.approx(expected)


def test_registration_rejects_poor_transfer(monkeypatch):
    monkeypatch.setattr(viz.fits,'getdata',lambda p:np.ones((30,30)))
    monkeypatch.setattr(viz,'inspect_frame',lambda *a:(None,None,None,None,None))
    monkeypatch.setattr(viz,'register_features',lambda *a:(np.zeros(2),1,dict(matched_stars=5,registration_rms=.1)))
    with pytest.raises(ValueError,match='Insufficient astrometric'):
        viz.estimate_display_translation('new','old')


def test_export_is_offline_preserves_inputs_and_rejects_stale_catalog(tmp_path,monkeypatch):
    import urllib.request
    def no_network(*args,**kwargs):
        raise AssertionError('Must remain offline')
    monkeypatch.setattr(urllib.request,'urlopen',no_network)
    monkeypatch.setattr(viz,'estimate_display_translation',lambda *args:(np.array([.25,-.5]),dict(matched_stars=8,registration_rms_2d_pix=.1)))
    processed = tmp_path/'data/processed'
    cat = processed/'evaluation/asteris_paper_catalog/s'
    cat.mkdir(parents=True)
    ra,dec = [63.],[22.]
    pd.DataFrame(dict(weak_label=['W01'],designation=['source'],gaia_source_id=['123'],k_m=[12.],
        x_pix_0based=[15.2],y_pix_0based=[16.8],catalog_ra_deg=ra,catalog_dec_deg=dec)).to_csv(cat/'weak_sources.csv',index=False)
    (cat/'astrometric_solution.json').write_text(json.dumps(dict(anchor_rms_pix=.2)))
    image = np.random.default_rng(3).normal(size=(32,32)).astype(np.float32)
    image[17,15] = np.nan  # catalog position is invalid even in the filled view
    image[0,0] = np.nan
    for profile in ('160','400'):
        model = processed/f'asteris_paper_{profile}'
        coadds = model/'coadds/s'; coadds.mkdir(parents=True)
        fits.writeto(coadds/'asteris8_coadd_s.fits',image)
        fits.writeto(coadds/'input_coadd_s.fits',image)
        columns = [fits.Column(name='weak_label',format='3A',array=['W01'])]
        for key,value in [('x_pix_0based',15.2),('y_pix_0based',16.8),('catalog_ra_deg',63.),('catalog_dec_deg',22.)]:
            columns.append(fits.Column(name=key,format='D',array=[value]))
        fits.HDUList([fits.PrimaryHDU(image),fits.BinTableHDU.from_columns(columns,name='CATALOG')]).writeto(
            cat/f'asteris_paper_{profile}_asteris8_coadd_s_catalog.fits')
        pd.DataFrame(dict(sequence=['s'],test_exposures=[16])).to_csv(model/'paper_coadd_statistics.csv',index=False)
    joint = processed/'blind_joint/s'; joint.mkdir(parents=True)
    for name in ('weighted_coadd','joint_score'):
        fits.writeto(joint/f'{name}_s.fits',image,fits.Header({'REFIMAGE':'first.fits','NCOMBINE':80}))
    before = {p:p.read_bytes() for p in processed.rglob('*') if p.is_file()}
    result = viz.export_catalog_validation(tmp_path,sequences=('s',),dpi=50,cutout_half_size=8)
    assert len(result) == 1 and result[0]['weak_sources'] == 1
    assert len(list((tmp_path/'figures').rglob('*.png'))) == 10
    assert 'display_interpolated' in result[0]['display_interpolated']['overlays']['weighted_coadd']
    for stats in result[0]['mask_statistics'].values():
        assert stats == dict(has_dq=False,invalid_pixels=2,display_filled_pixels=1,remaining_invalid_pixels=1)
    assert all(p.read_bytes() == content for p,content in before.items())
    points = pd.read_csv(tmp_path/'figures/catalog_validation_output/s/plotted_positions.csv')
    assert np.isclose(points.loc[points['product']=='weighted_coadd','x_plot'].iloc[0],14.7)
    assert np.isclose(points.loc[points['product']=='asteris160','x_plot'].iloc[0],15.2)
    assert not points.valid_at_position.any()
    metadata = json.loads((tmp_path/'figures/catalog_validation_output/s/visualization_metadata.json').read_text())
    assert metadata['interpolation']['enabled'] and metadata['interpolation']['max_hole_pixels'] == 16
    mask_only = viz.export_catalog_validation(tmp_path,sequences=('s',),dpi=50,cutout_half_size=8,
        output_dir=tmp_path/'figures/mask_only',export_interpolated=False)
    assert mask_only[0]['display_interpolated'] is None
    assert len(list((tmp_path/'figures/mask_only').rglob('*.png'))) == 5
    with pytest.raises(ValueError,match='inside project figures'):
        viz.export_catalog_validation(tmp_path,output_dir=processed)
    with pytest.raises(FileNotFoundError,match='No previously calibrated'):
        viz.export_catalog_validation(tmp_path,sequences=('unknown',))
    fits.writeto(processed/'asteris_paper_160/coadds/s/asteris8_coadd_s.fits',image+1,overwrite=True)
    with pytest.raises(ValueError,match='no longer matches'):
        viz.export_catalog_validation(tmp_path,sequences=('s',),dpi=50)
