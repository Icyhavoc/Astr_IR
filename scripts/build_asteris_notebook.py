"""Build the executable ASTERIS self-supervised training and evaluation notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "asteris" / "01_asteris_self_supervised.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # 背景扣除后 FITS 的 ASTERIS 时空自监督去噪

        这是 ASTERIS 流程的主代码 Notebook，结构与 `noise2noise/01_noise2noise_self_supervised.ipynb`
        保持一致：固定帧级切分、预处理验收、训练、推理、科学指标与自动测试依次展开。

        ASTERIS4 使用连续 8 帧组成 4 帧输入/4 帧目标；ASTERIS8 使用连续 16 帧组成
        8 帧输入/8 帧目标。网络直接复用只读目录中的原作者 3D Restormer-style U-Net，
        本项目只提供薄适配层。测试集不参与 checkpoint、输出模式或任何阈值选择。
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import subprocess
        import sys

        import torch
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
        DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "our_dataset"
        OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "asteris"
        FIGURE_ROOT = PROJECT_ROOT / "figures" / "asteris_output"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        # 安全默认值：执行预处理和小模型 smoke test，但不意外启动长时间训练/推理。
        RUN_PREPARE = True
        RUN_TRAIN = False
        RUN_INFERENCE = False
        RUN_CALIBRATION = False
        RUN_EVALUATION = False

        from astr_ir.asteris.dataset import AsterisPatchDataset, load_registered_stack
        from astr_ir.asteris.model import build_asteris_model, upstream_model_path, upstream_source_sha256
        from astr_ir.asteris.processor import (
            AsterisConfig,
            calibrate_and_finalize,
            load_calibrated_strength,
            load_manifests,
            prepare_manifests,
            run_inference,
            train_model,
        )
        from astr_ir.noise2noise.dataset import load_detector_mask

        config = AsterisConfig(model="asteris4", patch_t=4, patch_size=64, batch_size=1)
        config.validate()
        print("device:", "cuda" if torch.cuda.is_available() else "cpu")
        print("free disk GB:", round(__import__("shutil").disk_usage(PROJECT_ROOT).free / 2**30, 2))
        """
    ),
    md("## 1. 原始 ASTERIS 实现与输出语义"),
    code(
        """
        source_path = upstream_model_path(config.model)
        print("upstream source:", source_path)
        print("source SHA-256:", upstream_source_sha256(config.model))
        print("output equation: direct_prediction = input + learned_correction")

        smoke_model = build_asteris_model(
            "asteris4",
            f_maps=4,
            num_blocks=(1, 1, 1),
            num_refinement_blocks=1,
            heads=(1, 2, 4),
        ).eval()
        smoke_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smoke_model = smoke_model.to(smoke_device)
        with torch.inference_mode():
            smoke_output = smoke_model(torch.randn(1, 1, 4, 16, 16, device=smoke_device))
        print("ASTERIS4 smoke output:", tuple(smoke_output.shape))
        del smoke_model, smoke_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """
    ),
    md(
        """
        原模型末端已经执行 `output_head(features) + input`，并直接与奇数帧目标计算损失，
        因而本项目默认采用直接目标预测。`residual` 模式仅保留作验证集受控实验，不能用测试集选择。
        """
    ),
    md("## 2. 先切完整帧，再构造 2T 帧窗口"),
    code(
        """
        if RUN_PREPARE:
            split, windows, normalizations = prepare_manifests(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT, config=config
            )
        else:
            split, windows, normalizations = load_manifests(OUTPUT_ROOT)

        display(split.groupby(["sequence", "split"]).size().rename("frames").to_frame())
        display(windows.groupby(["sequence", "split"])["usable"].agg(["count", "sum"]))
        display(pd.DataFrame(normalizations).T)

        frame_split = split.set_index("frame_id")["split"]
        assert all(
            all(frame_split[frame_id] == row.split for frame_id in row.frame_ids.split("|"))
            for row in windows.itertuples(index=False)
        )
        assert all(
            not set(row.input_frame_ids.split("|")) & set(row.target_frame_ids.split("|"))
            for row in windows.itertuples(index=False)
        )
        """
    ),
    md(
        """
        每序列仍使用与 N2N 相同的 48/2/12/2/16 train/guard/validation/guard/test 时间块。
        归一化均值和标准差只从训练帧的非源、非盲点、非边缘像元拟合，并写入审计文件。
        """
    ),
    md("## 3. 源保护的全局 3σ clipping 与 mask 验收"),
    code(
        """
        detector_mask = load_detector_mask(DATASET_ROOT)
        sample_window = windows.loc[(windows["split"] == "train") & windows["usable"]].iloc[0]
        frame_table = split.set_index("frame_id", drop=False)
        sample_rows = frame_table.loc[sample_window.frame_ids.split("|")]
        sample_stack, sample_valid, clipping = load_registered_stack(
            sample_rows,
            INPUT_ROOT,
            detector_mask,
            normalizations[str(sample_window.sequence)],
            sigma=config.sigma,
            edge_width=config.edge_width,
            temporal_clip=config.temporal_clip,
        )
        print({
            "low": clipping.low,
            "high": clipping.high,
            "clipped_fraction": clipping.clipped_fraction,
            "source_pixels": int(clipping.source_mask.sum()),
            "valid_fraction": float(sample_valid.mean()),
        })
        display(pd.Series(
            clipping.clipping_mask.sum(axis=(1, 2)), name="clipped voxels per frame"
        ).to_frame())
        clipping_audit = pd.read_csv(
            OUTPUT_ROOT / "manifests" / "clipping_audit.csv",
            encoding="utf-8-sig",
            dtype={"sequence": str},
        )
        display(clipping_audit)
        assert clipping_audit["source_quality_gate_passed"].all()
        assert not clipping.clipping_mask[:, clipping.source_mask].any()
        """
    ),
    md(
        """
        时间轴 clipping 默认关闭。短时序中 seeing 与亚像元配准变化会产生真实的时间方向变化；
        即使已知目标受 mask 保护，未编目弱源仍可能被误判。只有在验证集峰值、孔径流量、FWHM
        和 SNR 质量门均通过后，才应把 `temporal_clip=True` 纳入正式实验。
        """
    ),
    md("## 4. 时空 patch、同步增强与 masked loss"),
    code(
        """
        dataset = AsterisPatchDataset(
            split, windows, INPUT_ROOT, detector_mask, normalizations,
            split="train", patch_size=config.patch_size, samples_per_epoch=4,
            sigma=config.sigma, edge_width=config.edge_width,
            temporal_clip=config.temporal_clip, augment=True,
        )
        sample = dataset[0]
        display(pd.Series({
            "input shape": tuple(sample["input"].shape),
            "target shape": tuple(sample["target"].shape),
            "loss-mask shape": tuple(sample["loss_mask"].shape),
            "valid loss fraction": float(sample["loss_mask"].mean()),
            "window": sample["window_id"],
        }).to_frame("value"))
        assert sample["input"].shape == sample["target"].shape == sample["loss_mask"].shape
        assert sample["input"].shape[:2] == (1, config.patch_t)
        """
    ),
    md("## 5. 训练与最佳验证 checkpoint"),
    code(
        """
        checkpoint = OUTPUT_ROOT / "checkpoints" / "best_checkpoint.pt"
        if RUN_TRAIN:
            checkpoint, history = train_model(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT, config=config,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        elif (OUTPUT_ROOT / "checkpoints" / "training_history.csv").exists():
            history = pd.read_csv(OUTPUT_ROOT / "checkpoints" / "training_history.csv", encoding="utf-8-sig")
        else:
            history = pd.DataFrame()
            print("尚未训练；把 RUN_TRAIN 改为 True 后执行本单元。")
        if not history.empty:
            display(history.loc[history["validation_loss"].idxmin()].to_frame("best"))
            display(history[["epoch", "train_loss", "validation_loss"]].tail(10))
        """
    ),
    md("## 6. 原始推理、验证集强度标定与 FITS 科学恒等式"),
    code(
        """
        if RUN_INFERENCE:
            raw_statistics = run_inference(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT, checkpoint,
                config=config, device="cuda" if torch.cuda.is_available() else "cpu",
                overwrite=False,
            )
        if RUN_CALIBRATION:
            selected_strength, calibration, statistics = calibrate_and_finalize(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT, checkpoint,
                config=config, overwrite=False,
            )
        elif (OUTPUT_ROOT / "manifests" / "strength_calibration.csv").exists():
            calibration = pd.read_csv(
                OUTPUT_ROOT / "manifests" / "strength_calibration.csv", encoding="utf-8-sig"
            )
            selected_strength = load_calibrated_strength(OUTPUT_ROOT)
        else:
            calibration = pd.DataFrame()
            selected_strength = np.nan
        if not RUN_CALIBRATION and (OUTPUT_ROOT / "asteris_statistics.csv").exists():
            statistics = pd.read_csv(
                OUTPUT_ROOT / "asteris_statistics.csv", encoding="utf-8-sig", dtype={"sequence": str}
            )
        elif not RUN_CALIBRATION:
            statistics = pd.DataFrame()
            print("尚无最终产品；依次启用 RUN_INFERENCE 和 RUN_CALIBRATION。")
        if not calibration.empty:
            display(calibration)
            print("validation-selected alpha:", selected_strength)
        if not statistics.empty:
            test_statistics = statistics.loc[statistics["split"] == "test"]
            display(test_statistics.groupby("sequence")[
                ["noise_ratio", "photometry_change_fraction", "aperture_snr_ratio"]
            ].median())
            display(test_statistics.groupby("sequence")["photometry_change_fraction"].apply(
                lambda values: 100 * values.abs().max()
            ).rename("max |flux change| [%]").to_frame())
            assert statistics["equation_max_abs_error_float32"].max() == 0
        """
    ),
    md("## 7. 与 Noise2Noise 共用的盲伪源评估接口"),
    code(
        """
        if RUN_EVALUATION:
            completed = subprocess.run(
                [
                    sys.executable, "scripts/run_source_evaluation.py",
                    "--model", "asteris", "--device", "cuda" if torch.cuda.is_available() else "cpu",
                    "--model-output-root", str(OUTPUT_ROOT),
                    "--evaluation-root", str(PROJECT_ROOT / "data" / "processed" / "evaluation" / "asteris"),
                ],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            print(completed.stdout)
            if completed.stderr.strip():
                print(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError("ASTERIS source evaluation failed")
        else:
            print("RUN_EVALUATION=False：不会在无 checkpoint 时启动长时间盲评估。")
        evaluation_root = PROJECT_ROOT / "data" / "processed" / "evaluation" / "asteris"
        if (evaluation_root / "metrics_by_snr.csv").exists():
            blind_metrics = pd.read_csv(evaluation_root / "metrics_by_snr.csv", encoding="utf-8-sig")
            evaluation_summary = pd.read_csv(evaluation_root / "evaluation_summary.csv", encoding="utf-8-sig")
            display(blind_metrics[["method", "target_snr", "completeness", "purity", "f1"]])
            display(evaluation_summary)
        """
    ),
    md("## 8. ASTERIS 专项自动测试"),
    code(
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_asteris.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("ASTERIS tests failed")
        """
    ),
    md(
        """
        ## 结论

        当前 Notebook 已把上游背景扣除 FITS、固定数据隔离、源保护 3σ clipping、训练集归一化、
        ASTERIS4/8 原始 3D 网络、masked 自监督损失、双向时序推理、验证集 α 标定和统一科学评估
        连接成一条可复现实验链。所有模型与阈值选择只使用训练/验证集，测试集只报告最终结果。
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
