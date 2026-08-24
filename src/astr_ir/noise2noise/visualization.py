"""Diagnostic plots for training, denoising, and weak-source recovery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["validation_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="Masked MSE", yscale="log", title="Noise2Noise training")
    ax.legend()
    return fig


def plot_denoising_triplet(original: np.ndarray, denoised: np.ndarray):
    residual = original - denoised
    low, high = np.nanpercentile(original, (1, 99))
    residual_limit = np.nanpercentile(np.abs(residual), 99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].imshow(original, origin="lower", cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("Background-subtracted input")
    axes[1].imshow(denoised, origin="lower", cmap="gray", vmin=low, vmax=high)
    axes[1].set_title("Noise2Noise denoised")
    axes[2].imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("Predicted residual")
    return fig


def save_figure(fig, path: str | Path, dpi: int = 160) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
