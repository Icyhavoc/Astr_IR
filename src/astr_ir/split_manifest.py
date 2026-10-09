"""Frozen image-only frame splits shared by the N2N and ASTERIS8 branches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from astr_ir.dq import load_detector_mask

def discover_sequences(input_root: str | Path) -> tuple[str, ...]:
    """Discover sequence directories that contain background-subtracted science FITS."""

    input_root = Path(input_root)
    sequences = tuple(
        path.name
        for path in sorted(input_root.iterdir())
        if path.is_dir() and any(path.glob("background_subtracted_*.fits"))
    )
    if not sequences:
        raise FileNotFoundError(
            f"No sequence directories containing background_subtracted_*.fits under {input_root}"
        )
    return sequences


def raw_filename(product_name: str) -> str:
    return product_name.removeprefix("background_subtracted_")


def _split_labels(count: int) -> list[str]:
    if count < 24:
        raise ValueError("Noise2Noise splitting requires at least 24 frames per sequence")
    train_count = int(round(0.60 * count))
    validation_count = int(round(0.15 * count))
    guard = max(1, int(round(0.025 * count)))
    test_count = count - train_count - validation_count - 2 * guard
    if test_count < 4:
        raise ValueError("Not enough frames remain for a test block")
    return (
        ["train"] * train_count
        + ["guard"] * guard
        + ["validation"] * validation_count
        + ["guard"] * guard
        + ["test"] * test_count
    )


def build_split_manifest(
    input_root: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    sequences: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a chronological frame-level split before any temporal pairs or patches."""

    input_root, dataset_root, output_path = Path(input_root), Path(dataset_root), Path(output_path)
    sequences = tuple(sequences) if sequences is not None else discover_sequences(input_root)
    from astr_ir.blind_registration import inspect_frame, register_features
    detector = load_detector_mask(dataset_root)
    rows: list[dict] = []
    for sequence in sequences:
        files = sorted((input_root / sequence).glob("background_subtracted_*.fits"))
        labels = _split_labels(len(files))
        reference = inspect_frame(files[0], detector)
        for index, (path, split) in enumerate(zip(files, labels)):
            current = reference if index == 0 else inspect_frame(path, detector)
            offset, _, registration = register_features(reference[3], reference[4], current[3], current[4])
            header = current[2]
            shape = current[0].shape
            flicker_applied = bool(header.get("HIERARCH FLK APPL", False))
            background_applied = bool(header.get("HIERARCH BKG APPL", False))
            if len(shape) != 2:
                raise ValueError(f"Expected 2-D FITS image, got {shape}: {path}")
            rows.append(
                {
                    "frame_id": f"{sequence}:{index:03d}",
                    "sequence": sequence,
                    "frame_index": index,
                    "split": split,
                    "product_path": path.relative_to(input_root).as_posix(),
                    "product_filename": path.name,
                    "filename": raw_filename(path.name),
                    **{key: np.nan for key in ("star_id", "input_status", "input_snr", "xc", "yc", "r_ap", "r_in", "r_out", "fwhm", "track_x", "track_y", "reference_x", "reference_y")},
                    "source_measurement_available": False,
                    "registration_method": "blind_image_stars_first_frame",
                    "alignment_dx": float(offset[1]),
                    "alignment_dy": float(offset[0]),
                    **registration,
                    "flicker_applied": flicker_applied,
                    "background_applied": background_applied,
                    "upstream_applied": flicker_applied and background_applied,
                    "height": shape[0],
                    "width": shape[1],
                }
            )
    manifest = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    return manifest
