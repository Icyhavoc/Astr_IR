"""Validate full blind preprocessor products and save catalog-free old/new diagnostics."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from astropy.io import fits
from astr_ir.flicker.processor import load_fits,load_detector_mask,make_edge_mask
from astr_ir.background.processor import block_location_scatter,neighbor_difference_noise
from scripts.validation.validate_products import validate_equations


def main():
    processed=ROOT/'data/processed'
    raw=ROOT/'data/raw/our_dataset'
    audit=processed/'pre_asteris_blind_v2'
    backup=processed/'comparison_before_blind_v2'
    detector=load_detector_mask(raw/'盲点表')
    mask=detector|make_edge_mask(detector.shape,24)
    stages={s:pd.read_csv(processed/s/f'{s}_statistics.csv',dtype={'sequence':str}) for s in ('flicker','background')}
    counts={p.name:len(list(p.glob('*.fits'))) for p in sorted(raw.iterdir()) if p.is_dir() and any(p.glob('*.fits'))}
    assert all((audit/f'{s}_complete.json').exists() for s in counts)
    assert len(stages['flicker'])==len(stages['background'])==sum(counts.values())
    for stage,table in stages.items():
        assert table.groupby('sequence').size().to_dict()==counts
        assert not table.photometry_gate_active.any()
    error_f,error_b,verified=validate_equations(raw,processed/'flicker',processed/'background',stages['flicker'],stages['background'])
    assert error_f==error_b==0
    dq_verified=0
    for stage in stages:
        for path in (processed/stage).glob('*/*.fits'):
            with fits.open(path,memmap=False) as hdul:
                hdul.verify('exception')
                assert hdul['DQ'].data.shape==detector.shape
                assert np.all((hdul['DQ'].data[detector]&3)==3)
                assert np.all(np.isfinite(hdul[0].data[(hdul['DQ'].data & 1)==0]))
            dq_verified+=1
    comparison=[]
    for stage in stages:
        for sequence in counts:
            old_paths=sorted((backup/stage/sequence).glob('old_*.fits'))
            assert len(old_paths)==2
            for old_path in old_paths:
                new_path=processed/stage/sequence/old_path.name.removeprefix('old_')
                old,_=load_fits(old_path); new,_=load_fits(new_path)
                use=mask|~np.isfinite(old)|~np.isfinite(new)
                comparison.append(dict(stage=stage,sequence=sequence,old_path=str(old_path),new_path=str(new_path),old_median=float(np.median(old[~use])),new_median=float(np.median(new[~use])),old_block64_scatter=block_location_scatter(old,use,64),new_block64_scatter=block_location_scatter(new,use,64),old_neighbor_noise=neighbor_difference_noise(old,use),new_neighbor_noise=neighbor_difference_noise(new,use)))
    pd.DataFrame(comparison).to_csv(audit/'old_new_comparison.csv',index=False,encoding='utf-8-sig')
    joint_summaries=[]
    for sequence in counts:
        root=processed/'blind_joint'/sequence
        record=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        assert record['frames']==counts[sequence] and not record['catalog_used']
        assert 'local background' in record['method']
        products=list(root.glob('*.fits'))
        assert len(products)==5
        for path in products:
            with fits.open(path,memmap=False) as hdul:
                hdul.verify('exception')
                good=(hdul['DQ'].data&1)==0
                assert np.isfinite(hdul[0].data[good]).all()
                assert np.isnan(hdul[0].data[~good]).all()
                count=hdul['COVERAGE'].data
                assert np.all((count>=0)&(count<=counts[sequence]))
                assert np.array_equal((hdul['DQ'].data&8)!=0,good&(count<counts[sequence]))
        sources=pd.read_csv(root/'blind_sources.csv')
        assert len(sources)==record['detections_ge5']
        assert (sources.snr_empirical>=5).all()
        assert (sources.coverage>=.8*counts[sequence]).all()
        joint_summaries.append(record)
    summary=dict(raw_frames=sum(counts.values()),sequences=counts,verified_fits=verified,dq_verified=dq_verified,flicker_equation_max_error=error_f,background_equation_max_error=error_b,old_science_fits_retained=len(comparison),old_retention='first and last science exposure per stage per sequence',catalog_used=False,training_started=False,stage_status={s:t.groupby(['sequence','status']).size().rename('frames').reset_index().to_dict('records') for s,t in stages.items()})
    summary['joint_fits_verified']=5*len(joint_summaries)
    summary['joint_detection']=joint_summaries
    (audit/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
