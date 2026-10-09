"""Build the executable Noise2Noise training and held-out evaluation notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "noise2noise" / "01_noise2noise_self_supervised.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # 背景扣除后 FITS 的 Noise2Noise 自监督去噪

        本 Notebook 展示固定帧级切分、lag=2–5 时间配对、残差 DnCNN 训练记录、
        固定去噪强度和测试产品检查。独立全盲注入评估位于 `notebooks/evaluation/`。
        测试集不参与 checkpoint 或去噪强度选择；旧执行输出仅供历史追溯。

        重跑须使用新的输出目录：`python scripts/run_noise2noise.py --stage prepare --output-root data/processed/noise2noise_YYYYMMDD`。
        训练、推理另行显式执行，不要将准备阶段当作训练授权。
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
            raise RuntimeError("无法定位项目根目录")
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "background"
        OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "noise2noise"
        FIGURE_ROOT = PROJECT_ROOT / "figures" / "noise2noise_output"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        from astr_ir.noise2noise.visualization import (
            plot_denoising_triplet,
            plot_training_history,
            save_figure,
        )
        """
    ),
    md("## 1. 固定帧级切分和时间帧对"),
    code(
        """
        split = pd.read_csv(
            OUTPUT_ROOT / "manifests" / "split_manifest.csv",
            encoding="utf-8-sig",
            dtype={"sequence": str},
        )
        pairs = pd.read_csv(
            OUTPUT_ROOT / "manifests" / "pair_manifest.csv",
            encoding="utf-8-sig",
            dtype={"sequence": str},
        )
        display(split.groupby(["sequence", "split"]).size().rename("frames").to_frame())
        display(
            pairs.loc[pairs["usable"]]
            .groupby(["sequence", "split", "lag"])
            .size()
            .rename("pairs")
            .to_frame()
        )
        assert (pairs["lag"] >= 2).all()
        frame_split = split.set_index("frame_id")["split"]
        assert all(
            frame_split[row.frame_a] == frame_split[row.frame_b] == row.split
            for row in pairs.itertuples(index=False)
        )
        """
    ),
    md(
        """
        每序列使用 48/2/12/2/16 的 train/guard/validation/guard/test 时间块。
        先切完整帧，再在同一集合内配对，最后在线裁 patch，避免帧或相邻 patch 泄漏。
        """
    ),
    md("## 2. 训练历史与最佳 checkpoint"),
    code(
        """
        history = pd.read_csv(OUTPUT_ROOT / "checkpoints" / "training_history.csv", encoding="utf-8-sig")
        best = history.loc[history["validation_loss"].idxmin()]
        display(best.to_frame("best checkpoint"))
        fig = plot_training_history(history)
        save_figure(fig, FIGURE_ROOT / "01_training_history.png")
        plt.show()
        """
    ),
    md("## 3. 固定去噪强度与产品统计"),
    code(
        """
        statistics = pd.read_csv(
            OUTPUT_ROOT / "noise2noise_statistics.csv",
            encoding="utf-8-sig",
            dtype={"sequence": str},
        )
        strengths = statistics["denoise_strength"].dropna().unique()
        assert len(strengths) == 1
        display(pd.Series({"fixed strength": strengths[0], "frames": len(statistics)}).to_frame("value"))
        print("去噪强度未经当前输入的注入/通量实验验证，不能仅凭此表宣称弱源性能提升。")
        """
    ),
    md("## 4. 弱源测试帧的输入、去噪图和预测残差"),
    code(
        """
        sample = statistics.loc[
            (statistics["split"] == "test") & (statistics["sequence"] == "90000002")
        ].iloc[0]
        original = np.asarray(
            fits.getdata(INPUT_ROOT / sample["sequence"] / f"background_subtracted_{sample['filename']}"),
            dtype=np.float32,
        )
        denoised = np.asarray(fits.getdata(OUTPUT_ROOT / sample["denoised_path"]), dtype=np.float32)
        fig = plot_denoising_triplet(original, denoised)
        save_figure(fig, FIGURE_ROOT / "03_weak_source_triplet.png")
        plt.show()
        """
    ),
    md("## 5. 冻结测试集产品指标"),
    code(
        """
        test = statistics.loc[statistics["split"] == "test"]
        metrics = pd.Series({
            "test frames": len(test),
            "median noise ratio": test["noise_ratio"].median(),
            "max float32 equation error": statistics["equation_max_abs_error_float32"].max(),
        })
        display(metrics.to_frame("value"))
        """
    ),
    md("## 6. 产品、切分和 DQ 验收"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "scripts/validation/validate_noise2noise.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("Noise2Noise structural validation failed")
        """
    ),
    md("## 7. 全项目自动测试"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
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

        当前工作目录的旧结果已归档。重新生成产品后，本 Notebook 只检查切分、产品公式、
        DQ 和描述性噪声指标。弱源通量传递、完整率和误检率必须在参数冻结后通过独立全盲
        注入评估；旧 Notebook 执行输出不能代表当前预处理版本。
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


if __name__ == "__main__":
    pass
