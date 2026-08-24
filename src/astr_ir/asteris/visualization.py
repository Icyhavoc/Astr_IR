"""Plotting helpers for the ASTERIS notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["validation_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="Masked loss", yscale="log")
    ax.legend()
    return fig


def plot_clipping(original: np.ndarray, clipped: np.ndarray, clipping_mask: np.ndarray):
    lo, hi = np.nanpercentile(original, (1, 99))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(original, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Before clipping")
    axes[1].imshow(clipped, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("After source-protected 3σ")
    axes[2].imshow(clipping_mask, origin="lower", cmap="magma")
    axes[2].set_title("Clipping mask")
    for ax in axes:
        ax.set_axis_off()
    return fig


def plot_denoising_triplet(original: np.ndarray, denoised: np.ndarray):
    residual = original - denoised
    lo, hi = np.nanpercentile(original, (1, 99.7))
    rlim = np.nanpercentile(np.abs(residual), 99)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(original, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    axes[1].imshow(denoised, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    axes[2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-rlim, vmax=rlim)
    for ax, title in zip(axes, ("Input", "ASTERIS", "Input - ASTERIS")):
        ax.set_title(title)
        ax.set_axis_off()
    return fig


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
