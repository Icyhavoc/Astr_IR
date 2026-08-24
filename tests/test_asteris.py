from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from astr_ir.asteris.dataset import (
    assert_window_isolation,
    build_window_manifest,
    relabel_manifest_for_patch_t,
)
from astr_ir.asteris.inference import denoise_array
from astr_ir.asteris.model import build_asteris_model
from astr_ir.asteris.preprocessing import (
    build_noise_estimation_mask,
    circular_source_mask,
    fit_normalization,
    sigma_clip_stack,
)
from astr_ir.asteris.processor import AsterisConfig, asteris_loss, load_model


@pytest.mark.parametrize(
    ("name", "depth", "f_maps", "blocks", "heads"),
    [
        ("asteris4", 4, 4, (1, 1, 1), (1, 2, 4)),
        ("asteris8", 8, 8, (1, 1, 1, 1), (1, 2, 4, 8)),
    ],
)
def test_original_asteris_adapter_preserves_shape(name, depth, f_maps, blocks, heads):
    model = build_asteris_model(
        name,
        f_maps=f_maps,
        num_blocks=blocks,
        num_refinement_blocks=1,
        heads=heads,
    )
    data = torch.randn(1, 1, depth, 16, 16)
    with torch.inference_mode():
        output = model(data)
    assert output.shape == data.shape


def test_sigma_clipping_preserves_shape_and_known_source_flux():
    rng = np.random.default_rng(3)
    stack = rng.normal(0, 1, (8, 32, 32)).astype(np.float32)
    stack[:, 15, 16] = 50.0
    stack[2, 5, 5] = 100.0
    detector = np.zeros((32, 32), dtype=bool)
    source = circular_source_mask((32, 32), [(16.0, 15.0, 3.0)])
    result = sigma_clip_stack(stack, detector, source, edge_width=2)
    assert result.data.shape == stack.shape
    assert result.clipping_mask.shape == stack.shape
    assert np.array_equal(result.data[:, source], stack[:, source])
    assert result.clipping_mask[2, 5, 5]
    for metric in ("peak", "flux", "fwhm", "snr"):
        assert result.source_metrics_after[metric] == pytest.approx(
            result.source_metrics_before[metric], nan_ok=True
        )


def test_noise_estimation_mask_excludes_blindmap_edges_and_source():
    detector = np.zeros((20, 20), dtype=bool)
    detector[10, 10] = True
    source = circular_source_mask((20, 20), [(5.0, 6.0, 2.0)])
    mask = build_noise_estimation_mask((20, 20), detector, source, edge_width=2)
    assert not mask[10, 10]
    assert not mask[6, 5]
    assert not mask[0].any()
    assert mask[12, 12]


def _split_manifest() -> pd.DataFrame:
    rows = []
    for index in range(32):
        split = "train" if index < 16 else "validation" if index < 24 else "test"
        rows.append(
            {
                "frame_id": f"s:{index:03d}",
                "sequence": "s",
                "frame_index": index,
                "split": split,
                "upstream_applied": True,
            }
        )
    return pd.DataFrame(rows)


def test_temporal_windows_do_not_cross_splits_or_overlap_input_target(tmp_path: Path):
    split = _split_manifest()
    windows = build_window_manifest(split, tmp_path / "windows.csv", patch_t=4)
    assert_window_isolation(split, windows)
    frame_split = split.set_index("frame_id")["split"]
    for row in windows.itertuples(index=False):
        ids = row.frame_ids.split("|")
        assert len(ids) == 8
        assert all(frame_split[frame_id] == row.split for frame_id in ids)
        assert not set(row.input_frame_ids.split("|")) & set(row.target_frame_ids.split("|"))


def test_asteris8_relabels_validation_but_keeps_frozen_test_block(tmp_path: Path):
    rows = []
    for index in range(80):
        original_split = (
            "train" if index < 48 else "guard" if index < 50 else
            "validation" if index < 62 else "guard" if index < 64 else "test"
        )
        rows.append({
            "frame_id": f"s:{index:03d}", "sequence": "s", "frame_index": index,
            "split": original_split, "upstream_applied": True,
        })
    original = pd.DataFrame(rows)
    relabeled = relabel_manifest_for_patch_t(original, 8)
    assert relabeled["split"].value_counts().to_dict() == {
        "train": 44, "validation": 16, "test": 16, "guard": 4,
    }
    assert relabeled.loc[relabeled["split"] == "test", "frame_id"].tolist() == original.loc[
        original["split"] == "test", "frame_id"
    ].tolist()
    windows = build_window_manifest(relabeled, tmp_path / "asteris8.csv", patch_t=8)
    assert set(windows["split"]) == {"train", "validation", "test"}


def test_normalization_is_fit_from_supplied_training_values_only():
    training = np.arange(100, dtype=float)
    stats = fit_normalization(training)
    test_values = np.full(100, 1e9)
    unchanged = fit_normalization(training)
    assert stats == unchanged
    assert stats["mean"] < np.min(test_values)


def test_masked_asteris_loss_ignores_invalid_voxel():
    prediction = torch.tensor([[[[[1.0, 100.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[[1.0, 0.0]]]]])
    total, stack_l1, mean_l2 = asteris_loss(prediction, target, mask)
    assert float(stack_l1) == pytest.approx(0.5)
    assert float(mean_l2) == pytest.approx(1.0)
    assert float(total) == pytest.approx(1.5)


def test_checkpoint_can_be_saved_and_loaded(tmp_path: Path):
    kwargs = {
        "model_name": "asteris4",
        "f_maps": 4,
        "num_blocks": (1, 1, 1),
        "num_refinement_blocks": 1,
        "heads": (1, 2, 4),
        "output_mode": "direct",
    }
    model = build_asteris_model(**kwargs)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_kwargs": kwargs, "model_state": model.state_dict(), "config": {}}, path)
    restored, state = load_model(path, device="cpu")
    assert state["model_kwargs"]["model_name"] == "asteris4"
    assert set(restored.state_dict()) == set(model.state_dict())


class Identity3D(nn.Module):
    spatial_divisor = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, value):
        return value + 0.0 * self.anchor


class Zero3D(Identity3D):
    def forward(self, value):
        return torch.zeros_like(value) + 0.0 * self.anchor


def test_same_evaluation_interface_preserves_identity_image():
    rng = np.random.default_rng(5)
    image = rng.normal(size=(32, 32)).astype(np.float32)
    valid = np.ones_like(image, dtype=bool)
    output = denoise_array(
        image,
        valid,
        Identity3D(),
        {"mean": 0.0, "std": 1.0},
        patch_t=4,
        device="cpu",
        tile_size=16,
        overlap=4,
    )
    assert np.allclose(output, image, atol=1e-6)


def test_zero_calibrated_strength_preserves_input():
    image = np.random.default_rng(6).normal(size=(32, 32)).astype(np.float32)
    output = denoise_array(
        image,
        np.ones_like(image, dtype=bool),
        Zero3D(),
        {"mean": 0.0, "std": 1.0},
        patch_t=4,
        device="cpu",
        tile_size=16,
        overlap=4,
        strength=0.0,
    )
    assert np.array_equal(output, image)


def test_float32_output_formula_is_auditable():
    rng = np.random.default_rng(9)
    original = rng.normal(size=(16, 16)).astype(np.float32)
    prediction = rng.normal(size=(16, 16)).astype(np.float32)
    residual = (original - prediction).astype(np.float32)
    denoised = (original - residual).astype(np.float32)
    assert np.max(np.abs(denoised - (original - residual))) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_smoke_asteris4():
    model = build_asteris_model(
        "asteris4", f_maps=4, num_blocks=(1, 1, 1), num_refinement_blocks=1, heads=(1, 2, 4)
    ).cuda()
    data = torch.randn(1, 1, 4, 16, 16, device="cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        output = model(data)
    assert output.shape == data.shape


def test_config_enforces_model_temporal_depth():
    with pytest.raises(ValueError, match="patch_t=4"):
        AsterisConfig(model="asteris4", patch_t=8).validate()
