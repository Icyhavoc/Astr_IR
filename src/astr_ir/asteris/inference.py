"""Overlap-weighted spatial and temporal inference for ASTERIS volumes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size > length:
        return [0]
    step = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if final not in positions:
        positions.append(final)
    return positions


def _blend_window(size: int) -> np.ndarray:
    if size <= 2:
        return np.ones((size, size), dtype=np.float32)
    axis = np.hanning(size).astype(np.float32)
    axis = np.maximum(axis, 1e-3)
    return np.outer(axis, axis)


def infer_volume(
    volume: np.ndarray,
    model: nn.Module,
    *,
    device: str | torch.device,
    tile_size: int = 64,
    overlap: int = 16,
    amp: bool = True,
) -> np.ndarray:
    """Run one normalized ``(T,H,W)`` volume through spatially tiled ASTERIS."""

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("volume must have shape (time, height, width)")
    factor = int(getattr(model, "spatial_divisor", 4))
    tile_size = min(int(tile_size), volume.shape[1], volume.shape[2])
    tile_size -= tile_size % factor
    if tile_size < factor:
        raise ValueError("Inference tile is smaller than the model spatial divisor")
    overlap = min(int(overlap), tile_size - factor)
    ys = tile_positions(volume.shape[1], tile_size, overlap)
    xs = tile_positions(volume.shape[2], tile_size, overlap)
    weight = _blend_window(tile_size)
    output = np.zeros_like(volume, dtype=np.float64)
    weights = np.zeros(volume.shape[1:], dtype=np.float64)
    device_obj = torch.device(device)
    amp_enabled = bool(amp and device_obj.type == "cuda")
    model.eval()
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                patch = torch.from_numpy(volume[:, y0 : y0 + tile_size, x0 : x0 + tile_size][None, None])
                patch = patch.to(device_obj)
                with torch.autocast(device_type=device_obj.type, enabled=amp_enabled):
                    prediction = model(patch)
                predicted = prediction[0, 0].float().cpu().numpy()
                output[:, y0 : y0 + tile_size, x0 : x0 + tile_size] += predicted * weight[None]
                weights[y0 : y0 + tile_size, x0 : x0 + tile_size] += weight
    return (output / np.maximum(weights[None], 1e-12)).astype(np.float32)


def temporal_window_starts(count: int, patch_t: int, stride: int | None = None) -> list[int]:
    full = 2 * int(patch_t)
    if count < full:
        raise ValueError(f"At least {full} frames are required, got {count}")
    step = int(stride or patch_t)
    starts = list(range(0, count - full + 1, step))
    final = count - full
    if final not in starts:
        starts.append(final)
    return starts


def denoise_registered_stack(
    normalized_stack: np.ndarray,
    model: nn.Module,
    *,
    patch_t: int,
    device: str | torch.device,
    tile_size: int = 64,
    overlap: int = 16,
    amp: bool = True,
) -> np.ndarray:
    """Predict every frame using both even→odd and odd→even self-supervised directions."""

    stack = np.asarray(normalized_stack, dtype=np.float32)
    accumulated = np.zeros_like(stack, dtype=np.float64)
    counts = np.zeros(stack.shape[0], dtype=np.float64)
    for start in temporal_window_starts(len(stack), patch_t):
        stop = start + 2 * patch_t
        even_indices = np.arange(start, stop, 2)
        odd_indices = np.arange(start + 1, stop, 2)
        predicted_odd = infer_volume(
            stack[even_indices], model, device=device, tile_size=tile_size, overlap=overlap, amp=amp
        )
        predicted_even = infer_volume(
            stack[odd_indices], model, device=device, tile_size=tile_size, overlap=overlap, amp=amp
        )
        accumulated[odd_indices] += predicted_odd
        accumulated[even_indices] += predicted_even
        counts[odd_indices] += 1
        counts[even_indices] += 1
    if np.any(counts == 0):
        raise RuntimeError("Temporal overlap left one or more frames without a prediction")
    return (accumulated / counts[:, None, None]).astype(np.float32)


def denoise_array(
    image: np.ndarray,
    valid: np.ndarray,
    model: nn.Module,
    normalization: dict[str, float],
    *,
    patch_t: int,
    device: str | torch.device,
    tile_size: int = 64,
    overlap: int = 16,
    strength: float = 1.0,
) -> np.ndarray:
    """Single-image adapter for the model-agnostic evaluation interface.

    The same injected image is repeated as temporal context, so injection truth
    is present in every context slice.  Full science inference uses real
    neighboring frames through :func:`denoise_registered_stack`.
    """

    normalized = (np.asarray(image, np.float32) - normalization["mean"]) / normalization["std"]
    normalized[~np.asarray(valid, bool) | ~np.isfinite(normalized)] = 0.0
    volume = np.repeat(normalized[None], patch_t, axis=0)
    predicted = infer_volume(volume, model, device=device, tile_size=tile_size, overlap=overlap)
    raw_prediction = predicted[patch_t // 2] * normalization["std"] + normalization["mean"]
    result = (
        np.asarray(image, np.float32)
        + float(strength) * (raw_prediction.astype(np.float32) - np.asarray(image, np.float32))
    ).astype(np.float32)
    result[~np.asarray(valid, bool)] = np.asarray(image, np.float32)[~np.asarray(valid, bool)]
    return result
