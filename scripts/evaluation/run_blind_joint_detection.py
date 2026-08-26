"""Run blind multi-exposure detection; never reads catalogs or starts training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from astr_ir.evaluation.blind_joint import run_sequence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"data/processed/background")
    parser.add_argument("--output-root",type=Path,default=ROOT/"data/processed/blind_joint")
    parser.add_argument("--sequences",nargs="+")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--method",choices=['legacy','v3'],default='legacy')
    parser.add_argument("--psf-mode",choices=['auto','gaussian','elliptical','moffat','empirical'],default='auto')
    parser.add_argument("--noise-box",type=int,default=64)
    parser.add_argument("--iterations",type=int,default=3)
    parser.add_argument("--threshold",type=float,default=5.)
    parser.add_argument("--follow-preprocess",action="store_true",help="Wait for each frame's completed background audit row before reading it")
    args=parser.parse_args()
    if args.method=='v3':
        from astr_ir.evaluation.weak_detection import DetectionConfig
        from astr_ir.evaluation.weak_joint_pipeline import run_sequence_v3
        if args.follow_preprocess: raise ValueError('V3 requires a completed, frozen preprocessing run')
        if args.output_root.resolve()==(ROOT/'data/processed/blind_joint').resolve():
            raise ValueError('V3 requires an explicit NEW --output-root; legacy products are preserved')
        config=DetectionConfig(psf_mode=args.psf_mode,noise_box=args.noise_box,iterations=args.iterations,threshold=args.threshold)
    sequences=args.sequences or [p.name for p in sorted(args.input_root.iterdir()) if p.is_dir() and any(p.glob("background_subtracted_*.fits"))]
    for sequence in sequences:
        progress=ROOT/"data/processed/pre_asteris_blind_v2"/f"{sequence}_background.csv" if args.follow_preprocess else None
        if args.method=='v3':
            result=run_sequence_v3(args.input_root,ROOT/'data/raw/our_dataset',args.output_root,sequence,config=config,limit=args.limit)
        else:
            result=run_sequence(args.input_root,ROOT/"data/raw/our_dataset",args.output_root,sequence,limit=args.limit,progress_path=progress)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
