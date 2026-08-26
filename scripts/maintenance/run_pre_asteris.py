"""Rerun blind 1/f + background only; preserve two old science frames per stage/sequence."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from astr_ir.flicker import processor as flk
from astr_ir.background import processor as bkg


def process_sequence(sequence, resume=False):
    raw = ROOT / "data/raw/our_dataset"
    output = ROOT / "data/processed"
    audit = output / "pre_asteris_blind_v2"
    marker = audit / f"{sequence}_complete.json"
    if resume and marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    detector = flk.load_detector_mask(raw / "盲点表")
    files = sorted((raw / sequence).glob("*.fits"))
    rows_f, rows_b = [], []
    start = time.monotonic()
    for index, path in enumerate(files, 1):
        _, rf = flk.process_fits_file(path, output / "flicker" / sequence, detector, None, flk.FlickerConfig(), overwrite=True)
        _, rb = bkg.process_fits_file(rf["corrected_path"], output / "background" / sequence, detector, None, bkg.BackgroundConfig(), overwrite=True)
        for row, stage, keys in ((rf, "flicker", ("corrected_path", "model_path")), (rb, "background", ("subtracted_path", "model_path"))):
            row.update(sequence=sequence, sequence_frame_index=index)
            for key in keys:
                row[key] = Path(row[key]).relative_to(output / stage).as_posix()
        rows_f.append(rf)
        rows_b.append(rb)
        if index % 10 == 0 or index == len(files):
            print(f"{sequence}: {index}/{len(files)}  {time.monotonic()-start:.0f}s", flush=True)
            pd.DataFrame(rows_f).to_csv(audit / f"{sequence}_flicker.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(rows_b).to_csv(audit / f"{sequence}_background.csv", index=False, encoding="utf-8-sig")
    record = dict(sequence=sequence, frames=len(files), seconds=time.monotonic()-start)
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "data/raw/our_dataset"
    output = ROOT / "data/processed"
    audit = output / "pre_asteris_blind_v2"
    backup = output / "comparison_before_blind_v2"
    sequences = [p.name for p in sorted(raw.iterdir()) if p.is_dir() and any(p.glob("*.fits"))]
    if not args.resume:
        if backup.exists() or audit.exists():
            raise FileExistsError("Existing run/backup: use --resume to continue without replacing old comparisons")
        backup.mkdir(parents=True)
        for stage, prefix in (("flicker", "flicker_corrected_"), ("background", "background_subtracted_")):
            for sequence in sequences:
                old = sorted((output / stage / sequence).glob(prefix + "*.fits"))
                if old:
                    destination = backup / stage / sequence
                    destination.mkdir(parents=True)
                    for path in dict.fromkeys((old[0], old[-1])):
                        shutil.copy2(path, destination / ("old_" + path.name))
            for path in (output / stage).glob("*.csv"):
                shutil.copy2(path, backup / ("old_" + path.name))
        audit.mkdir(parents=True)
        (audit / "config.json").write_text(json.dumps(dict(flicker=asdict(flk.FlickerConfig()), background=asdict(bkg.BackgroundConfig()), catalog_used=False, training_run=False), indent=2), encoding="utf-8")
    elif not backup.exists() or not audit.exists():
        raise FileNotFoundError("Cannot resume without the original comparison backup and audit directory")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_sequence, s, args.resume) for s in sequences]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    for stage in ("flicker", "background"):
        tables = [pd.read_csv(audit / f"{s}_{stage}.csv", dtype={"sequence": str}) for s in sequences]
        pd.concat(tables, ignore_index=True).to_csv(output / stage / f"{stage}_statistics.csv", index=False, encoding="utf-8-sig")
    print("All pre-ASTERIS products complete. No training was launched.", flush=True)


if __name__ == "__main__":
    main()
