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
    parser.add_argument("--follow-preprocess",action="store_true",help="Wait for each frame's completed background audit row before reading it")
    args=parser.parse_args()
    sequences=args.sequences or [p.name for p in sorted(args.input_root.iterdir()) if p.is_dir() and any(p.glob("background_subtracted_*.fits"))]
    for sequence in sequences:
        progress=ROOT/"data/processed/pre_asteris_blind_v2"/f"{sequence}_background.csv" if args.follow_preprocess else None
        result=run_sequence(args.input_root,ROOT/"data/raw/our_dataset",args.output_root,sequence,limit=args.limit,progress_path=progress)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
