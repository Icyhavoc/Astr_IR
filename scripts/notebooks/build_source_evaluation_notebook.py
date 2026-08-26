"""Build the executable, model-agnostic mock-source evaluation notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "evaluation" / "01_mock_source_evaluation.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # 通用伪源注入-恢复评估

        本 Notebook 只读取已经冻结的评估产品，不重新训练或推理。经验 PSF 仅来自训练集，
        检测阈值仅由验证集选择，最终完备度、纯度、F1、光度和位置误差仅来自测试集。

        完整运行命令：`python scripts/evaluation/run_source_evaluation.py --model noise2noise --device cuda`。
        """
    ),
    code(
        """
        from pathlib import Path
        import json
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
            raise RuntimeError("无法定位项目根目录")
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        EVALUATION_ROOT = PROJECT_ROOT / "data" / "processed" / "evaluation" / "noise2noise"
        SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "noise2noise" / "manifests" / "split_manifest.csv"
        FIGURE_ROOT = PROJECT_ROOT / "figures" / "evaluation_output" / "noise2noise"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        from astr_ir.evaluation.visualization import (
            plot_completeness_purity,
            plot_empirical_psf,
            plot_photometric_accuracy,
            save_figure,
        )
        """
    ),
    md("## 1. 数据职责隔离与经验 PSF"),
    code(
        """
        split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig", dtype={"sequence": str})
        psf_diagnostics = pd.read_csv(
            EVALUATION_ROOT / "psf_training_diagnostics.csv",
            encoding="utf-8-sig",
            dtype={"sequence": str},
        )
        psf = np.asarray(fits.getdata(EVALUATION_ROOT / "empirical_psf.fits"), dtype=float)
        display(split.groupby(["sequence", "split"]).size().rename("frames").to_frame())
        display(psf_diagnostics.groupby(["sequence", "accepted"]).size().rename("cutouts").to_frame())
        assert np.isclose(psf.sum(), 1.0, atol=2e-6)
        assert (psf >= 0).all()
        accepted_frames = set(psf_diagnostics.loc[psf_diagnostics["accepted"], "frame_id"])
        assert all(split.set_index("frame_id").loc[frame_id, "split"] == "train" for frame_id in accepted_frames)
        fig = plot_empirical_psf(psf)
        save_figure(fig, FIGURE_ROOT / "empirical_psf.png")
        plt.show()
        """
    ),
    md("## 2. 仅使用验证集选择统一盲检阈值"),
    code(
        """
        calibration = pd.read_csv(
            EVALUATION_ROOT / "validation_threshold_calibration.csv", encoding="utf-8-sig"
        )
        with (EVALUATION_ROOT / "selected_threshold.json").open(encoding="utf-8") as handle:
            selected = json.load(handle)
        display(pd.Series(selected).to_frame("value"))
        display(calibration.loc[calibration["selected"]])
        assert selected["selection_split"] == "validation"
        """
    ),
    md("## 3. 测试集盲检完备度、纯度和95%置信区间"),
    code(
        """
        metrics = pd.read_csv(EVALUATION_ROOT / "metrics_by_snr.csv", encoding="utf-8-sig")
        comparison = pd.read_csv(
            EVALUATION_ROOT / "paired_comparison_by_snr.csv", encoding="utf-8-sig"
        )
        display(metrics[[
            "method", "target_snr", "injected", "tp", "fn", "fp",
            "completeness", "completeness_ci_low", "completeness_ci_high",
            "purity", "purity_ci_low", "purity_ci_high", "f1",
        ]])
        display(comparison)
        fig = plot_completeness_purity(metrics)
        save_figure(fig, FIGURE_ROOT / "completeness_purity.png")
        plt.show()
        """
    ),
    md("## 4. 已恢复伪源的光度保真度"),
    code(
        """
        injections = pd.read_csv(EVALUATION_ROOT / "injection_recovery.csv", encoding="utf-8-sig")
        recovered = injections.loc[injections["detected"]]
        display(
            recovered.groupby(["method", "target_snr"])[
                ["relative_flux_error", "astrometric_error"]
            ].median()
        )
        fig = plot_photometric_accuracy(injections)
        save_figure(fig, FIGURE_ROOT / "photometric_accuracy.png")
        plt.show()
        """
    ),
    md(
        """
        SNR=2–3只保留越过检测阈值的正涨落，因而会出现明显正流量偏差；这是检测选择效应，
        不能把仅对已检出对象计算的低SNR中位数直接解释为模型测光偏差。
        """
    ),
    md("## 5. 检测极限和未注入图像的新候选"),
    code(
        """
        summary = pd.read_csv(EVALUATION_ROOT / "evaluation_summary.csv", encoding="utf-8-sig")
        unmodified = pd.read_csv(
            EVALUATION_ROOT / "unmodified_test_catalog.csv", encoding="utf-8-sig"
        )
        display(summary)
        display(unmodified)
        """
    ),
    md("## 6. 严格验收"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "scripts/validation/validate_source_evaluation.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("source evaluation strict validation failed")
        """
    ),
    md("## 7. 全项目自动测试"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("pytest failed")
        """
    ),
    md(
        """
        ## 结论

        冻结测试集显示 Noise2Noise 在SNR=4–5提供温和的盲检完备度提升，并保持约98%的纯度；
        未注入测试图中出现3个仅输出候选，必须结合原始多帧一致性判断。当前结果是160帧数据上的基线，
        新数据加入后应按完整80帧序列重新划分并扩大每个SNR的注入数量。
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK} ({len(cells)} cells)")
