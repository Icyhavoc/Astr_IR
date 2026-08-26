"""Regenerate 01_background_subtraction.ipynb with the documented workflow."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "background" / "01_background_subtraction.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # 二维红外 FITS 图像背景扣除

        本 Notebook 处理 `data/processed/flicker` 中全部 `flicker_corrected_*.fits`，明确排除
        `flicker_model_*.fits`。流程参考 Bagley et al. 的 CEERS Epoch 1 NIRCam 背景扣除：
        粗背景压平、裁剪环形估计、四层源掩膜、稳健二维网格背景，以及尺度相关 RMS 与源测光验证。
        """
    ),
    md(
        """
        ## 科学约定

        与 1/f 校正不同，本步骤的二维背景是最终需要扣除的产品。输出严格满足：

        `background_subtracted = flicker_corrected_input - background_model`

        输入 FITS 的科学头和已有 `FLK` 元数据均保留，并新增 `BKG` 元数据。
        """
    ),
    code(
        """
        from pathlib import Path
        import subprocess
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from astropy.io import fits

        start = Path.cwd().resolve()
        PROJECT_ROOT = next(
            (path for path in (start, *start.parents) if (path / "src" / "astr_ir").is_dir()),
            None,
        )
        if PROJECT_ROOT is None:
            raise RuntimeError("无法定位项目根目录；请在 Astr_IR 项目内运行 Notebook")
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from astr_ir.background.processor import (
            BackgroundConfig, discover_input_files, load_detector_mask,
            load_fits, run_batch, subtract_background,
        )
        from astr_ir.background.visualization import (
            plot_background_histogram, plot_background_stages,
            plot_equivalent_rms, plot_photometry_changes, plot_source_masks,
            plot_subtraction_overview, save_figure,
        )

        INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "flicker"
        DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "our_dataset"
        OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "background"
        FIGURE_ROOT = PROJECT_ROOT / "figures" / "background_output"

        config = BackgroundConfig(
            rough_box_size=100,
            ring_inner_radius=80,
            ring_width=4,
            final_box_size=64,
            final_filter_size=5,
        )
        config
        """
    ),
    md("## 1. 输入清单：只读取 1/f 校正科学图"),
    code(
        """
        inventory = {p.name: discover_input_files(INPUT_ROOT, p.name) for p in sorted(INPUT_ROOT.iterdir()) if p.is_dir() and any(p.glob("flicker_corrected_*.fits"))}
        model_files = sorted(INPUT_ROOT.glob("*/flicker_model_*.fits"))
        print({key: len(value) for key, value in inventory.items()})
        print("excluded flicker models:", len(model_files))
        expected_frames = sum(map(len, inventory.values()))
        assert expected_frames > 0
        assert all(path.name.startswith("flicker_corrected_") for files in inventory.values() for path in files)
        """
    ),
    md("## 2. 读取盲点，按文件顺序选择示例帧（不使用星表）"),
    code(
        """
        detector_mask = load_detector_mask(DATASET_ROOT / "盲点表")
        sample_path = inventory["90000003"][0]
        sample_image, sample_header = load_fits(sample_path)
        print(sample_path.name, sample_image.shape, sample_image.dtype)
        print("blind-map pixels:", int(detector_mask.sum()))
        """
    ),
    md(
        """
        ## 3. 粗背景与裁剪环形估计

        CEERS 使用 `Background2D(box_size=100, filter_size=3)` 得到粗背景，再以 5×RMS
        阈值屏蔽亮像元并使用内半径 80、宽度 4 的环形中位数。这里保留相同空间尺度；为使
        160 帧 1024×1024 图像可批量运行，环形步骤使用裁剪后的掩膜归一化环形卷积，作用是
        为分层源检测提供平坦图像，最终科学背景仍由重源掩膜后的稳健网格估计给出。
        """
    ),
    code(
        """
        sample_result = subtract_background(
            sample_image, detector_mask=detector_mask, config=config
        )
        fig = plot_background_stages(sample_result)
        save_figure(fig, FIGURE_ROOT / "01_background_stages.png")
        plt.show()
        """
    ),
    md("## 4. 四层源掩膜、blindmap、边缘与无效像元"),
    code(
        """
        fig = plot_source_masks(sample_result)
        save_figure(fig, FIGURE_ROOT / "02_source_masks.png")
        plt.show()
        print("final mask fraction:", f"{100*sample_result.metrics['mask_fraction']:.2f}%")
        """
    ),
    md(
        """
        分层检测沿用 CEERS“从扩展源到紧致源”的思想。论文给出的高斯宽度、连通像元数和
        膨胀半径针对 0.03 arcsec/pixel 的拼接图；本项目对单帧数据提高了紧致层检测阈值并缩小
        膨胀半径，避免噪声峰被扩张为满图掩膜。
        """
    ),
    md("## 5. 最终二维背景与扣除结果"),
    code(
        """
        fig = plot_subtraction_overview(sample_result)
        save_figure(fig, FIGURE_ROOT / "03_subtraction_overview.png")
        plt.show()
        pd.Series(sample_result.metrics).to_frame("value")
        """
    ),
    md("## 6. 未掩膜背景分布"),
    code(
        """
        fig = plot_background_histogram(sample_result)
        save_figure(fig, FIGURE_ROOT / "04_background_histogram.png")
        plt.show()
        """
    ),
    md("## 7. CEERS 式尺度相关背景 RMS"),
    code(
        """
        fig = plot_equivalent_rms(sample_result)
        save_figure(fig, FIGURE_ROOT / "05_scale_dependent_rms.png")
        plt.show()
        """
    ),
    md("## 8. 全量处理（自动发现所有序列，当前 400 帧）"),
    code(
        """
        statistics_path = OUTPUT_ROOT / "background_statistics.csv"
        existing_products = sorted(OUTPUT_ROOT.glob("*/*.fits"))
        if statistics_path.exists() and len(existing_products) == 2 * expected_frames:
            stats = pd.read_csv(statistics_path, encoding="utf-8-sig", dtype={"sequence": str})
            print("Reusing verified products in data/processed/background/.")
        else:
            raise RuntimeError("请先运行 scripts/maintenance/run_pre_asteris.py；它会备份首末旧帧后重跑两个阶段。")
        print(stats.groupby(["sequence", "status"]).size().to_string())
        print("frames:", len(stats))
        """
    ),
    md("## 9. 全量质量指标"),
    code(
        """
        applied = stats[stats["applied"]]
        summary = pd.DataFrame({
            "large-scale reduction": applied["large_scale_reduction"].describe(percentiles=[0.5]),
            "high-frequency ratio": applied["high_frequency_noise_ratio"].describe(percentiles=[0.5]),
            "mask fraction": applied["mask_fraction"].describe(percentiles=[0.5]),
        })
        display(summary)
        print("catalog used: False; known-target photometry gates: disabled")
        """
    ),
    code(
        """
        print("随机注入通量响应测试见 tests/test_blind_joint.py 和 tests/test_flicker.py。")
        """
    ),
    md("## 10. FITS 头、文件选择与输出公式检查"),
    code(
        """
        output_fits = sorted(OUTPUT_ROOT.glob("*/*.fits"))
        print("output FITS:", len(output_fits))
        assert len(output_fits) == 2 * expected_frames
        assert not any("flicker_model" in str(path) for path in stats["input_filename"])
        for path in output_fits:
            with fits.open(path, memmap=False) as hdul:
                hdul.verify("exception")
        print("strict FITS validation passed")
        print("max float32 equation error:", stats["equation_max_abs_error_float32"].max())
        assert stats["equation_max_abs_error_float32"].max() == 0
        """
    ),
    md("## 11. 自动测试"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"pytest failed with exit code {completed.returncode}")
        """
    ),
    md(
        """
        ## 结论

        本流程只读取 1/f 校正科学图，使用 blindmap 与重源掩膜拟合平滑二维背景；质量门同时约束
        64 像素尺度起伏和高频噪声；不再通过已知目标位置保护源或控制处理。
        背景网格默认 64 像素，源掩膜完全由图像自动产生，星表仅用于最终人工核验。
        未通过质量门的帧输出原图 float32
        副本和零背景模型，并在统计表中记录原因。
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK} ({len(cells)} cells)")
