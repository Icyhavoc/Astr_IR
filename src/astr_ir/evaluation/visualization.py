"""Plots for model-agnostic mock-source evaluation products."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_completeness_purity(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"input": "#555555", "output": "#d1495b"}
    for method, group in metrics.groupby("method"):
        group = group.sort_values("target_snr")
        color = colors.get(method)
        axes[0].plot(group["target_snr"], group["completeness"], "o-", label=method, color=color)
        axes[0].fill_between(
            group["target_snr"],
            group["completeness_ci_low"],
            group["completeness_ci_high"],
            alpha=0.16,
            color=color,
        )
        axes[1].plot(group["target_snr"], group["purity"], "o-", label=method, color=color)
        axes[1].fill_between(
            group["target_snr"],
            group["purity_ci_low"],
            group["purity_ci_high"],
            alpha=0.16,
            color=color,
        )
    axes[0].set(title="Blind mock-source completeness", ylabel="Completeness")
    axes[1].set(title="Blind detection purity", ylabel="Purity")
    for ax in axes:
        ax.set(xlabel="Injected matched-filter SNR", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend()
    return fig


def plot_photometric_accuracy(injections: pd.DataFrame):
    recovered = injections.loc[(injections["split"] == "test") & injections["detected"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    positions = sorted(recovered["target_snr"].unique())
    width = 0.28
    for offset, method, color in ((-width / 2, "input", "#555555"), (width / 2, "output", "#d1495b")):
        groups = [
            recovered.loc[
                (recovered["method"] == method) & np.isclose(recovered["target_snr"], snr),
                "relative_flux_error",
            ].dropna().to_numpy()
            for snr in positions
        ]
        bp = ax.boxplot(
            groups,
            positions=np.asarray(positions) + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.45)
        for median in bp["medians"]:
            median.set(color=color)
        ax.plot([], [], color=color, linewidth=7, alpha=0.45, label=method)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions, [f"{snr:g}" for snr in positions])
    ax.set(
        xlabel="Injected matched-filter SNR",
        ylabel="Relative matched-filter flux error",
        title="Recovered-source photometric fidelity",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_empirical_psf(psf: np.ndarray):
    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    image = ax.imshow(psf, origin="lower", cmap="magma")
    ax.set(title="Training-only empirical PSF", xlabel="x (pixel)", ylabel="y (pixel)")
    fig.colorbar(image, ax=ax, label="Unit-flux PSF")
    return fig


def save_figure(fig, path: str | Path, dpi: int = 180) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
