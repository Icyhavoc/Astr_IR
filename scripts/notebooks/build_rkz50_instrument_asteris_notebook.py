"""Build the RKZ50 instrument-specific ASTERIS experiment notebook."""
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "asteris" / "01_rkz50_instrument_asteris.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
nb["cells"] = [
    markdown(
        """
        # RKZ50 仪器专用 ASTERIS

        目标：在不复用 JWST 权重的前提下，比较仪器专用小网络与完整 ASTERIS8。
        四个独立天区严格隔离；正式 checkpoint 按固定误检预算内的盲注入完整率选择，
        validation MSE 只作训练诊断。

        所有运行开关默认关闭。本 Notebook 不使用星表参与训练、阈值或模型选择。
        任何展示图必须读取 DQ/有效覆盖/坏点表，并在绘图层屏蔽左上集中坏点；不得改写 FITS。
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import pandas as pd
        import torch

        start = Path.cwd().resolve()
        PROJECT_ROOT = next(path for path in (start, *start.parents) if (path / "src" / "astr_ir").is_dir())
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from astr_ir.asteris.paper_pipeline import (
            PaperAsterisConfig, RKZ50_FIELD_ASSIGNMENTS, prepare_paper_dataset,
            train_paper_model, select_checkpoint_by_detection, selected_checkpoint,
            run_paper_inference,
        )

        INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "background"
        DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "our_dataset"
        OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "asteris_rkz50_field_split_run1"
        SELECTION_TABLE = PROJECT_ROOT / "data" / "processed" / "evaluation" / "run1" / "checkpoint_detection_metrics.csv"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CONFIG = PaperAsterisConfig()
        CONFIG
        """
    ),
    markdown("## 1. 冻结四天区切分"),
    code(
        """
        assignment = pd.DataFrame([
            {"sequence": sequence, "field_id": field_id, "split": split}
            for sequence, (field_id, split) in RKZ50_FIELD_ASSIGNMENTS.items()
        ])
        assignment
        """
    ),
    code(
        """
        RUN_PREPARE = False
        if RUN_PREPARE:
            split, stacks = prepare_paper_dataset(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT,
                sequences=tuple(RKZ50_FIELD_ASSIGNMENTS), config=CONFIG,
                field_assignments=RKZ50_FIELD_ASSIGNMENTS,
            )
            display(split.groupby(["field_id", "split", "sequence"]).size().rename("frames"))
            display(stacks[["sequence", "split", "frames"]])
        """
    ),
    markdown(
        """
        ## 2. 训练候选 checkpoint

        正式运行前先把 `CONFIG` 改成选定的小网络或 ASTERIS8 配置，并使用新输出目录。
        训练只生成候选模型；最低 validation MSE 不等于正式最佳模型。
        """
    ),
    code(
        """
        RUN_TRAINING = False
        if RUN_TRAINING:
            last_checkpoint, history = train_paper_model(OUTPUT_ROOT, config=CONFIG, device=DEVICE)
            display(history.tail())
            print("last checkpoint:", last_checkpoint)
        """
    ),
    markdown(
        """
        ## 3. 冻结 matched-filter 注入选择

        `SELECTION_TABLE` 必须来自独立 validation 天区，使用与输入共加相同的曝光、背景、
        DQ、配准和 matched-filter。表内阈值须在注入前冻结，且 `catalog_used=False`。
        """
    ),
    code(
        """
        RUN_SELECTION = False
        if RUN_SELECTION:
            chosen = select_checkpoint_by_detection(
                OUTPUT_ROOT, SELECTION_TABLE,
                max_false_peaks_per_image=CONFIG.selection_max_false_peaks_per_image,
                flux_response_tolerance=CONFIG.selection_flux_response_tolerance,
            )
            print("selected:", chosen)
        """
    ),
    markdown("## 4. 独立测试天区推理"),
    code(
        """
        RUN_INFERENCE = False
        if RUN_INFERENCE:
            checkpoint = selected_checkpoint(OUTPUT_ROOT)
            statistics = run_paper_inference(
                INPUT_ROOT, DATASET_ROOT, OUTPUT_ROOT, checkpoint,
                evaluation_sequences=("90000005_1", "90000005_2"),
                device=DEVICE, overwrite=False,
            )
            display(statistics)
        """
    ),
    markdown(
        """
        ## 5. 判定与下一步

        只在以下条件同时成立时推进模型：固定误检预算下的完整率优于输入共加、通量响应
        通过护栏、定位误差未恶化，并且失败案例没有集中在坏点、亮星残差或图像边界。
        若小网络与输入共加无差异，应停止增加 epoch/裁块数，优先增加独立天区。
        """
    ),
]

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
