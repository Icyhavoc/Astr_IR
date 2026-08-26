"""Build the review-oriented Jupyter notebook in document-work order."""

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "notebooks" / "flicker" / "01_flicker_noise_correction.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md(
    """
    # 红外图像 1/f 条纹噪声校正

    本 Notebook 按《科创任务安排.docx》中“主要工作”的原始顺序分为九段，并在最后批量生成交付产品。

    设计原则：原始 FITS 是唯一科学输入；星源、blindmap 和边缘像元只用于排除条纹估计；输出满足
    `corrected = original - flicker_model`。合并盲点表与输入 DQ，输出 PRIMARY + DQ。
    源掩膜完全从图像自动产生，星表和已知目标坐标只允许在流程结束后用于人眼验证。
    """
)

md("## 0. 环境、路径与参数")
code(
    """
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    start = Path.cwd().resolve()
    PROJECT_ROOT = next(
        (path for path in (start, *start.parents) if (path / "src" / "astr_ir").is_dir()),
        None,
    )
    if PROJECT_ROOT is None:
        raise RuntimeError("无法定位项目根目录；请在 Astr_IR 项目内运行 Notebook")
    DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "our_dataset"
    OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "flicker"
    FIGURE_ROOT = PROJECT_ROOT / "figures" / "flicker_output"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from astr_ir.flicker.processor import (
        FlickerConfig, correct_flicker, load_detector_mask, load_fits,
        run_batch,
    )
    from astr_ir.flicker.visualization import (
        plot_background_stage, plot_correction_overview,
        plot_direction_diagnostics, plot_masks_and_background,
        plot_photometry_changes, plot_power_spectrum,
        plot_profiles_before_after, save_figure,
    )

    config = FlickerConfig(
        direction="auto",
        profile_smooth_size=5,
    )
    RUN_FULL_BATCH = False  # 默认查看已完成产品；完整重跑用 scripts/maintenance/run_pre_asteris.py。
    OVERWRITE_PRODUCTS = False
    config
    """
)
code(
    """
    detector_mask = load_detector_mask(DATASET_ROOT / "盲点表")
    sample_path = sorted((DATASET_ROOT / "90000003").glob("*.fits"))[0]
    image, header = load_fits(sample_path)
    result = correct_flicker(image, detector_mask=detector_mask, config=config)
    print("示例帧:", sample_path.name)
    print("图像尺寸:", image.shape, "EXPOSURE:", header.get("EXPOSURE"))
    """
)

md(
    """
    ## 主要工作 1/9：检查条纹主要沿行方向还是列方向

    水平条纹对应“每行偏置随 y 变化”，记为 `row`；垂直条纹对应“每列偏置随 x 变化”，记为 `column`。
    自动方向分数为 Sigma 裁剪剖面的稳健标准差除以中位数统计误差，并比较扣除统计误差后的信号幅度。
    """
)
code(
    """
    direction_table = pd.DataFrame([
        {"方向": "row / 水平条纹", "score": result.row_diagnostic.score,
         "signal_std": result.row_diagnostic.signal_std},
        {"方向": "column / 垂直条纹", "score": result.column_diagnostic.score,
         "signal_std": result.column_diagnostic.signal_std},
    ])
    display(direction_table)
    print("自动选择:", result.selected_direction)
    fig = plot_direction_diagnostics(result)
    save_figure(fig, FIGURE_ROOT / "01_direction_diagnostic.png")
    plt.show()
    """
)

md(
    """
    ## 主要工作 2/9：合并星源、DQ 和边缘掩膜

    `detector_mask = DeadBlindMap OR NoiseBlindMap`；再与输入 DQ、自动紧致源掩膜、
    边缘 24 像素和非有限像元合并。不读取 CSV 目标位置；坏像元不进入平滑与条纹估计。
    """
)
code(
    """
    mask_summary = pd.Series({
        "detector blindmap union": result.detector_mask.sum(),
        "automatic sources": result.source_mask.sum(),
        "edge": result.edge_mask.sum(),
        "combined (union)": result.combined_mask.sum(),
    }, name="masked pixels")
    display(mask_summary.to_frame())
    fig = plot_masks_and_background(result)
    save_figure(fig, FIGURE_ROOT / "02_masks.png")
    plt.show()
    """
)

md(
    """
    ## 主要工作 3/9：先估计并减去低频二维背景

    将未掩膜像元按 64×64 分块取中位数，在粗网格上高斯平滑后插值回原尺寸。该背景仅用于隔离条纹；
    最终代数关系仍是原图减条纹模型，因此不会把天文低频背景当作输出扣除项。
    """
)
code(
    """
    print("background median:", np.median(result.low_frequency_background))
    print("residual robust std:", 1.4826 * np.median(np.abs(result.residual - np.median(result.residual))))
    fig = plot_background_stage(result)
    save_figure(fig, FIGURE_ROOT / "03_low_frequency_background.png")
    plt.show()
    """
)

md(
    """
    ## 主要工作 4/9：对未掩膜像元计算每行或每列的 Sigma 裁剪中位数

    两个方向都计算，便于诊断；真正的候选条纹模型只使用自动或手动选择的方向。
    """
)
code(
    """
    selected_diag = result.row_diagnostic if result.selected_direction == "row" else result.column_diagnostic
    profile_stats = pd.Series({
        "profile length": len(selected_diag.profile),
        "median valid pixels per line": np.median(selected_diag.valid_count),
        "profile robust std": selected_diag.robust_std,
        "median uncertainty": selected_diag.noise_floor,
        "direction score": selected_diag.score,
    })
    display(profile_stats.to_frame("value"))
    """
)

md("## 主要工作 5/9：对一维条纹序列进行中值低通平滑")
code(
    """
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(selected_diag.profile, label="Sigma-clipped median", lw=1, alpha=0.65)
    ax.plot(result.smoothed_profile, label=f"selected model (size={result.profile_smooth_size})", lw=1.5)
    ax.set(xlabel=result.selected_direction, ylabel="DN", title="One-dimensional flicker profile")
    ax.legend()
    save_figure(fig, FIGURE_ROOT / "05_smoothed_profile.png")
    plt.show()
    """
)

md("## 主要工作 6/9：将一维序列扩展为二维条纹模型")
code(
    """
    print("model shape:", result.flicker_model.shape)
    print("model median/min/max:", np.median(result.flicker_model), result.flicker_model.min(), result.flicker_model.max())
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    lim = np.percentile(np.abs(result.flicker_model), 99)
    im = ax.imshow(result.flicker_model, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title("2-D flicker model")
    fig.colorbar(im, ax=ax, label="DN")
    save_figure(fig, FIGURE_ROOT / "06_flicker_model.png")
    plt.show()
    """
)

md(
    """
    ## 主要工作 7/9：从图像扣除条纹，再恢复低频背景

    实现等价于 `(original - background - model) + background`，即严格的 `corrected = original - model`。
    首选 5 点模型；若局部恶化行占比或最坏增量越界，则依次尝试 3 点和 1 点模型。
    弱条纹或所有候选均未通过质量门时，模型置零并返回具体原因，绝不强行扣除。
    """
)
code(
    """
    equation_error = np.max(np.abs(result.corrected - (result.original - result.flicker_model)))
    display(pd.Series({**result.metrics, "float64 equation max abs error": equation_error}).to_frame("value"))
    fig = plot_correction_overview(result)
    save_figure(fig, FIGURE_ROOT / "07_correction_overview.png")
    plt.show()
    """
)

md("## 主要工作 8/9：比较校正前后的行列中位数和一维功率谱")
code(
    """
    fig = plot_profiles_before_after(result)
    save_figure(fig, FIGURE_ROOT / "08_profiles_before_after.png")
    plt.show()
    fig = plot_power_spectrum(result)
    save_figure(fig, FIGURE_ROOT / "08_power_spectrum.png")
    plt.show()
    print("selected-profile robust-std reduction:", f"{100 * result.metrics['relative_reduction']:.2f}%")
    """
)

md(
    """
    ## 主要工作 9/9：检查恒星测光是否受到影响

    不启用基于已知目标的测光门限。用随机注入源的输入/输出通量响应做回归测试；
    不把含条纹的原始孔径流量当成真实流量。星表仅供最终输出的人工核验。
    """
)
code(
    """
    display(pd.Series({
        "aperture flux before": result.metrics["photometry_before"],
        "aperture flux after": result.metrics["photometry_after"],
        "relative change [%]": 100 * result.metrics["photometry_change_fraction"],
        "photometry gate active": result.metrics["photometry_gate_active"],
    }).to_frame("value"))
    """
)

md(
    """
    ## 10. 批量生成输出产品并汇总验收指标

    每个原始 FITS 输出 `flicker_corrected_*.fits` 和 `flicker_model_*.fits`；完整保留并规范化原科学头，
    数据统一写为 float32。全数据集统计写入 `data/processed/flicker/flicker_statistics.csv`。
    """
)
code(
    """
    if RUN_FULL_BATCH:
        stats = run_batch(
            DATASET_ROOT,
            OUTPUT_ROOT,
            config=config,
            overwrite=OVERWRITE_PRODUCTS,
        )
    else:
        stats_path = OUTPUT_ROOT / "flicker_statistics.csv"
        stats = pd.read_csv(stats_path, encoding="utf-8-sig")

    display(stats.groupby(["sequence", "status"]).size().rename("frames").to_frame())
    """
)
code(
    """
    applied = stats[stats["applied"].astype(bool)]
    acceptance = pd.Series({
        "total frames": len(stats),
        "corrected frames": len(applied),
        "not corrected frames": len(stats) - len(applied),
        "min reduction among corrected [%]": 100 * applied["relative_reduction"].min(),
        "corrected frames meeting >=30%": int((applied["relative_reduction"] >= 0.30).sum()),
        "max background-noise ratio": applied["background_noise_ratio"].max(),
        "selected size=5 frames": int((applied["selected_profile_smooth_size"] == 5).sum()),
        "selected size=3 frames": int((applied["selected_profile_smooth_size"] == 3).sum()),
        "selected size=1 frames": int((applied["selected_profile_smooth_size"] == 1).sum()),
        "max local line increase [DN]": applied["local_max_increase_dn"].max(),
        "max lines degraded over 10 DN": int(applied["local_worse_over_threshold_lines"].max()),
        "max float32 equation error": stats["equation_max_abs_error_float32"].max(),
    })
    display(acceptance.to_frame("value"))
    print("No catalog-conditioned photometry quality gate is used.")
    """
)

md(
    """
    ## 11. 自动测试

    测试覆盖行/列方向识别、弱条纹不校正、恒等式、恒星测光、手动方向、blindmap 合并和 FITS 头保留。
    """
)
code(
    """
    import subprocess
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    print(completed.stdout)
    if completed.stderr.strip():
        print("STDERR:")
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"pytest failed with exit code {completed.returncode}; "
            "see the detailed output above"
        )
    """
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(nb, OUT)
print(OUT)


if __name__ == "__main__":
    pass
