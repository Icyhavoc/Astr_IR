"""Visualization helpers for the flicker-correction notebook and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .processor import CorrectionResult, sigma_clipped_profile


def percentile_limits(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    values = image[np.isfinite(image)]
    return tuple(np.percentile(values, [low, high]))


def plot_direction_diagnostics(result: CorrectionResult):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    vmin, vmax = percentile_limits(result.residual, 1, 99)
    axes[0, 0].imshow(result.residual, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Low-frequency-background-subtracted residual")
    axes[0, 1].bar(
        ["row / horizontal", "column / vertical"],
        [result.row_diagnostic.score, result.column_diagnostic.score],
        color=["#2878B5", "#C82423"],
    )
    axes[0, 1].set_ylabel("profile scatter / median uncertainty")
    axes[0, 1].set_title(f"Selected: {result.selected_direction}")
    axes[1, 0].plot(result.row_diagnostic.profile, lw=1)
    axes[1, 0].set(title="Sigma-clipped row profile", xlabel="row", ylabel="DN")
    axes[1, 1].plot(result.column_diagnostic.profile, lw=1, color="#C82423")
    axes[1, 1].set(title="Sigma-clipped column profile", xlabel="column", ylabel="DN")
    return fig


def plot_masks_and_background(result: CorrectionResult):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    masks = [
        (result.detector_mask, "Blind-map detector mask"),
        (result.source_mask, "Known + auto source mask"),
        (result.edge_mask, "Edge mask"),
        (result.combined_mask, "Combined exclusion mask"),
    ]
    for ax, (mask, title) in zip(axes, masks):
        ax.imshow(mask, origin="lower", cmap="magma", interpolation="nearest")
        ax.set_title(f"{title}\n{100 * np.mean(mask):.2f}%")
    return fig


def plot_background_stage(result: CorrectionResult):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        [result.original, result.low_frequency_background, result.residual],
        ["Original", "Low-frequency 2-D background", "Residual"],
    ):
        vmin, vmax = percentile_limits(image, 1, 99)
        im = ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.75, label="DN")
    return fig


def plot_correction_overview(result: CorrectionResult):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    vmin, vmax = percentile_limits(result.original, 1, 99)
    model_lim = max(abs(x) for x in percentile_limits(result.flicker_model, 1, 99))
    panels = [
        (result.original, "Original", "gray", vmin, vmax),
        (result.flicker_model, "Flicker model", "coolwarm", -model_lim, model_lim),
        (result.corrected, "Corrected", "gray", vmin, vmax),
        (result.original - result.corrected, "Original - corrected", "coolwarm", -model_lim, model_lim),
    ]
    for ax, (image, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.72, label="DN")
    fig.suptitle(f"status={result.status}; direction={result.selected_direction}")
    return fig


def _profiles_before_after(result: CorrectionResult, direction: str):
    before = sigma_clipped_profile(result.residual, result.combined_mask, direction)
    after = sigma_clipped_profile(
        result.corrected - result.low_frequency_background,
        result.combined_mask,
        direction,
    )
    return before.profile, after.profile


def plot_profiles_before_after(result: CorrectionResult):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, direction in zip(axes, ("row", "column")):
        before, after = _profiles_before_after(result, direction)
        ax.plot(before, label="before", lw=1)
        ax.plot(after, label="after", lw=1)
        ax.set(title=f"{direction} median profile", xlabel=direction, ylabel="DN")
        ax.legend()
    return fig


def one_dimensional_power(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(profile, dtype=float)
    x = np.nan_to_num(x - np.nanmedian(x))
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    frequency = np.fft.rfftfreq(x.size)
    power = np.abs(spectrum) ** 2
    return frequency[1:], power[1:]


def plot_power_spectrum(result: CorrectionResult):
    before, after = _profiles_before_after(result, result.selected_direction)
    fb, pb = one_dimensional_power(before)
    fa, pa = one_dimensional_power(after)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.loglog(fb, pb, label="before")
    ax.loglog(fa, pa, label="after")
    ax.set(xlabel="spatial frequency [cycle / pixel]", ylabel="power", title="1-D power spectrum")
    ax.legend()
    return fig


def plot_photometry_changes(stats, snr_min: float = 10.0):
    frame = stats.copy()
    valid = frame["photometry_change_fraction"].notna()
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.scatter(
        frame.loc[valid, "input_snr"],
        100 * frame.loc[valid, "photometry_change_fraction"],
        c=frame.loc[valid, "applied"].map({True: "#2878B5", False: "#B0B0B0"}),
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
