from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from astr_ir.noise2noise.dataset import (
    _robust_linear_track,
    _split_labels,
    build_pair_manifest,
    discover_sequences,
)
from astr_ir.noise2noise.model import ResidualDnCNN
from astr_ir.noise2noise.processor import (
    Noise2NoiseConfig,
    aperture_flux_snr,
    denoise_array,
    masked_mse,
)


def test_split_is_frame_level_with_guard_blocks():
    labels = _split_labels(80)
    assert labels.count("train") == 48
    assert labels.count("validation") == 12
    assert labels.count("test") == 16
    assert labels.count("guard") == 4
    assert labels[48:50] == ["guard", "guard"]
    assert labels[62:64] == ["guard", "guard"]


def test_pair_manifest_never_crosses_split_and_uses_requested_lags(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {
                "frame_id": f"s:{index}",
                "sequence": "s",
                "frame_index": index,
                "split": "train" if index < 6 else "test",
                "upstream_applied": True,
            }
            for index in range(10)
        ]
    )
    pairs = build_pair_manifest(manifest, tmp_path / "pairs.csv", lags=(2, 3))
    frame_split = manifest.set_index("frame_id")["split"].to_dict()
    assert set(pairs["lag"]) <= {2, 3}
    assert all(frame_split[row.frame_a] == frame_split[row.frame_b] == row.split for row in pairs.itertuples())


def test_robust_track_ignores_low_snr_centroid_outlier():
    index = np.arange(20, dtype=float)
    values = 100.0 + 0.2 * index
    values[10] += 30.0
    accepted = np.ones(20, dtype=bool)
    track = _robust_linear_track(index, values, accepted)
    assert np.max(np.abs(track - (100.0 + 0.2 * index))) < 0.1


def test_model_preserves_shape_and_has_single_science_output():
    model = ResidualDnCNN(depth=4, features=8)
    model_input = torch.randn(2, 2, 32, 32)
    output = model(model_input)
    assert output.shape == (2, 1, 32, 32)


def test_masked_mse_ignores_invalid_pixel():
    prediction = torch.tensor([[[[1.0, 100.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0]]]])
    assert float(masked_mse(prediction, target, mask)) == 1.0


class IdentityDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return model_input[:, :1] + 0.0 * self.anchor


def test_tiled_identity_inference_preserves_science_image():
    rng = np.random.default_rng(7)
    image = rng.normal(10.0, 2.0, (96, 96)).astype(np.float32)
    valid = np.ones_like(image, dtype=bool)
    denoised, residual = denoise_array(
        image,
        valid,
        IdentityDenoiser(),
        scale=2.0,
        tile_size=64,
        overlap=8,
    )
    assert np.allclose(denoised, image, atol=2e-6)
    assert np.max(np.abs(residual)) < 2e-6


def test_aperture_snr_is_positive_for_synthetic_star():
    yy, xx = np.mgrid[:64, :64]
    image = np.random.default_rng(8).normal(0.0, 1.0, (64, 64))
    image += 20.0 * np.exp(-((xx - 31) ** 2 + (yy - 30) ** 2) / (2 * 2.0**2))
    flux, snr = aperture_flux_snr(
        image,
        {"xc": 32.0, "yc": 31.0, "r_ap": 6.0, "r_in": 9.0, "r_out": 13.0},
    )
    assert flux > 0
    assert snr > 10


def test_config_rejects_temporally_adjacent_noise_pairs():
    with pytest.raises(ValueError, match=">= 2"):
        Noise2NoiseConfig(lags=(1,)).validate()


def test_sequence_discovery_is_data_driven(tmp_path: Path):
    for sequence in ("90000007", "90000005"):
        root = tmp_path / sequence
        root.mkdir()
        (root / "background_subtracted_example.fits").touch()
    (tmp_path / "empty_directory").mkdir()
    assert discover_sequences(tmp_path) == ("90000005", "90000007")
