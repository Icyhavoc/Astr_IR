"""Diagnostic plots for the background-subtraction notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .processor import BackgroundResult, equivalent_rms_curve


def percentile_limits(image: np.ndarray, low: float = 1.0, high: float = 99.0):
    values = image[np.isfinite(image)]
    return tuple(np.percentile(values, [low, high]))


def plot_background_stages(result: BackgroundResult):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    panels = [
        (result.original, "Input after 1/f correction"),
        (result.rough_background, "Coarse robust background"),
        (result.ring_background, "Clipped annular background"),
        (result.detection_residual, "Flattened detection image"),
    ]
    for ax, (image, title) in zip(axes, panels):
        lo, hi = percentile_limits(image)
        im = ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.72, label="DN")
    return fig


def plot_source_masks(result: BackgroundResult):
    masks = [result.detector_mask, result.known_source_mask, *result.tier_masks, result.combined_mask]
    titles = ["Blind-map", "Known target"] + [f"Cumulative tier {i}" for i in range(1, len(result.tier_masks) + 1)] + ["Final combined"]
    cols = 4
    rows = int(np.ceil(len(masks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, mask, title in zip(axes, masks, titles):
        ax.imshow(mask, origin="lower", cmap="magma", interpolation="nearest")
        ax.set_title(f"{title}\n{100*np.mean(mask):.2f}%")
    for ax in axes[len(masks) :]:
        ax.set_visible(False)
    return fig


def plot_subtraction_overview(result: BackgroundResult):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    lo, hi = percentile_limits(result.original)
    model_lo, model_hi = percentile_limits(result.background_model)
    panels = [
        (result.original, "Input", "gray", lo, hi),
        (result.background_model, "2-D background model", "viridis", model_lo, model_hi),
        (result.background_subtracted, "Background subtracted", "gray", lo - np.median(result.background_model), hi - np.median(result.background_model)),
        (result.original - result.background_subtracted, "Input - subtracted", "viridis", model_lo, model_hi),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.72, label="DN")
    fig.suptitle(f"status={result.status}")
    return fig


def plot_background_histogram(result: BackgroundResult):
    before = result.original[~result.combined_mask]
    after = result.background_subtracted[~result.combined_mask]
    before = before - np.median(before)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bounds = np.percentile(np.concatenate([before, after]), [0.5, 99.5])
    ax.hist(before, bins=120, range=bounds, histtype="step", density=True, label="before - median")
    ax.hist(after, bins=120, range=bounds, histtype="step", density=True, label="after")
    ax.set(xlabel="Unmasked pixel value [DN]", ylabel="density", title="Background pixel distribution")
    ax.legend()
    return fig


def plot_equivalent_rms(result: BackgroundResult):
    sizes, before = equivalent_rms_curve(result.original, result.combined_mask)
    _, after = equivalent_rms_curve(result.background_subtracted, result.combined_mask)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(sizes, before, "o-", label="before")
    ax.plot(sizes, after, "o-", label="after")
    ax.set(xscale="log", yscale="log", xlabel="block size [pixel]", ylabel="equivalent per-pixel RMS [DN]", title="Scale-dependent background RMS")
    ax.legend()
    return fig


def plot_photometry_changes(stats, snr_min: float = 10.0):
    valid = stats["photometry_change_fraction"].notna()
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.scatter(
        stats.loc[valid, "input_snr"],
        100 * stats.loc[valid, "photometry_change_fraction"],
        c=stats.loc[valid, "applied"].map({True: "#2878B5", False: "#B0B0B0"}),
        s=24,
        alpha=0.8,
    )
    ax.axhline(1, color="#C82423", ls="--", lw=1)
    ax.axhline(-1, color="#C82423", ls="--", lw=1)
    ax.axvline(snr_min, color="black", ls=":", lw=1)
    ax.set(xlabel="input SNR", ylabel="aperture flux change [%]", title="Target-star photometry preservation")
    return fig


def save_figure(fig, path: str | Path, dpi: int = 160) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
