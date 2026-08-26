"""Paired raw-to-detection injection trials, never training or overwriting inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=ROOT/'data/raw/our_dataset')
    parser.add_argument('--split-manifest',type=Path,required=True,help='Use only filename, sequence, frame_index, split; never coordinates')
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--phase',choices=['validation','test'],default='validation')
    parser.add_argument('--output-root',type=Path,required=True,help='Fresh directory only')
    parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--seed',type=int,help='Defaults differ: validation=20260826, test=20260827')
    parser.add_argument('--fluxes',type=float,nargs='+',default=[100,200,400])
    parser.add_argument('--thresholds',type=float,nargs='+',default=[4,5,6])
    parser.add_argument('--sources',type=int,default=6)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--gain',type=float,help='Electrons per DN; omitted means background-dominated deterministic injections')
    parser.add_argument('--box-size',type=int,default=64)
    parser.add_argument('--filter-size',type=int,default=5)
    parser.add_argument('--ablation',action='store_true',help='Run 32/64 x 1/3/5 meshes with identical random seeds')
    parser.add_argument('--checkpoint',type=Path,help='Explicit eval-only ASTERIS checkpoint; skipped by default')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    import pandas as pd
    from astr_ir.background.processor import BackgroundConfig
    from astr_ir.evaluation.stage_recovery import RecoveryConfig, run_recovery_files
    # Reading only these columns prevents use of historical target tracks/SNR.
    table=pd.read_csv(args.split_manifest,usecols=['filename','sequence','frame_index','split'],dtype={'sequence':str})
    allowed=['validation','val'] if args.phase=='validation' else ['test']
    selected=table.loc[table.sequence.eq(args.sequence)&table.split.isin(allowed)].sort_values('frame_index')
    if args.limit<2 or len(selected)<args.limit: raise ValueError('Requested split has too few exposures (at least 2 required)')
    selected=selected.iloc[:args.limit]
    if Path(args.sequence).name!=args.sequence or any(Path(name).name!=name for name in selected.filename):
        raise ValueError('Unsafe sequence/filename in frozen split')
    files=[args.dataset_root/args.sequence/name for name in selected.filename]
    inference=None
    if args.checkpoint is not None:
        if args.limit<8: raise ValueError('ASTERIS8 evaluation requires at least eight exposures')
        import torch
        from astr_ir.asteris.paper_pipeline import load_paper_model,denoise_registered_exposures
        model,_,model_config=load_paper_model(args.checkpoint,args.device)
        model.eval()
        def inference(images,valid):
            with torch.inference_mode():
                input_coadd,denoised,good,_=denoise_registered_exposures(images,valid,model,model_config,device=args.device)
            return input_coadd,denoised,good
    seed=args.seed if args.seed is not None else (20260826 if args.phase=='validation' else 20260827)
    recovery=RecoveryConfig(seed=seed,sources_per_trial=args.sources,fluxes=tuple(args.fluxes),
        thresholds=tuple(args.thresholds),repeats=args.repeats,gain_e_per_dn=args.gain)
    combinations=[(box,filt) for box in (32,64) for filt in (1,3,5)] if args.ablation else [(args.box_size,args.filter_size)]
    if args.output_root.exists(): raise FileExistsError('Use a new recovery output directory')
    for box,filt in combinations:
        destination=args.output_root/f'box{box}_filter{filt}' if args.ablation else args.output_root
        report=run_recovery_files(files,args.dataset_root,destination,recovery_config=recovery,
            background_config=BackgroundConfig(final_box_size=box,final_filter_size=filt),inference=inference)
        report.update(phase=args.phase,split_manifest=str(args.split_manifest.resolve()),
            checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
            checkpoint_preprocessing_compatibility='not automatically established; old checkpoints remain historical controls' if args.checkpoint else 'not applicable')
        if args.checkpoint:
            with args.checkpoint.open('rb') as stream: report['checkpoint_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        (destination/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(destination,flush=True)


if __name__=='__main__': main()
