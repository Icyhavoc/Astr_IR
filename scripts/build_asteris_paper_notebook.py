"""Generate the notebook-first ASTERIS8 160-versus-400 experiment."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "asteris" / "02_asteris8_paper_160_vs_400.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["cells"] = [
    markdown(
        """
        # ASTERIS8：论文式多曝光训练，160帧与400帧对比

        本 Notebook 是更新后的主实验入口。它使用原作者 ASTERIS8 网络和官方初始化，
        将16张独立曝光交错为8帧输入/8帧目标，执行时间轴与全局3σ裁剪、论文损失、
        八帧时间合成和真正的多曝光盲伪源注入。旧的单图复制评估和 α 混合不再用于本实验。

        训练优化器与发布源码一致：AdamW、weight decay 1e-4、学习率 1.5e-4、
        CosineAnnealingLR T_max=2,000,000。8 GB GPU 使用 AMP，并仅在反向传播时去掉
        损失的公共 1e6 倍数以避免 float16 溢出。
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import pandas as pd
        import torch

        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name == "asteris":
            PROJECT_ROOT = PROJECT_ROOT.parents[1]
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from astr_ir.asteris.paper_pipeline import (
            PaperAsterisConfig, prepare_paper_dataset, run_paper_inference, train_paper_model
        )
        from astr_ir.asteris.paper_evaluation import (
            PaperEvaluationConfig, run_paper_mock_evaluation
        )

        INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "background"
        DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "our_dataset"
        PROFILES = {
            "160": ("90000002", "90000003"),
            "400": ("90000002", "90000003", "90000004", "90000005_1", "90000005_2"),
        }
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CONFIG = PaperAsterisConfig()
        CONFIG
        """
    ),
    markdown("## 1. 数据清单与冻结切分"),
    code(
        """
        inventory = []
        for profile, sequences in PROFILES.items():
            for sequence in sequences:
                inventory.append({
                    "profile": profile,
                    "sequence": sequence,
                    "raw_fits": len(list((DATASET_ROOT / sequence).glob("*.fits"))),
                    "background_fits": len(list((INPUT_ROOT / sequence).glob("background_subtracted_*.fits"))),
                })
        pd.DataFrame(inventory)
        """
    ),
    code(
        """
        RUN_PREPARE = False
        if RUN_PREPARE:
            for profile, sequences in PROFILES.items():
                root = PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{profile}"
                prepare_paper_dataset(INPUT_ROOT, DATASET_ROOT, root, sequences=sequences, config=CONFIG)
        """
    ),
    markdown("## 2. ASTERIS8 正式训练"),
    code(
        """
        RUN_TRAINING = False
        if RUN_TRAINING:
            for profile in ("160", "400"):
                root = PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{profile}"
                train_paper_model(root, config=CONFIG, device=DEVICE)
        """
    ),
    markdown("## 3. 冻结测试曝光的时间合成"),
    code(
        """
        RUN_INFERENCE = False
        if RUN_INFERENCE:
            for profile in ("160", "400"):
                root = PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{profile}"
                run_paper_inference(
                    INPUT_ROOT, DATASET_ROOT, root, root / "checkpoints" / "best_checkpoint.pt",
                    evaluation_sequences=("90000002", "90000003"), device=DEVICE, overwrite=True,
                )
        """
    ),
    markdown("## 4. 真正的多曝光盲伪源评估"),
    code(
        """
        RUN_EVALUATION = False
        if RUN_EVALUATION:
            for profile in ("160", "400"):
                root = PROJECT_ROOT / "data" / "processed" / f"asteris_paper_{profile}"
                evaluation = PROJECT_ROOT / "data" / "processed" / "evaluation" / f"asteris_paper_{profile}"
                run_paper_mock_evaluation(
                    INPUT_ROOT, DATASET_ROOT, root, evaluation,
                    root / "checkpoints" / "best_checkpoint.pt", device=DEVICE,
                    config=PaperEvaluationConfig(),
                )
        """
    ),
    markdown("## 5. 160帧与400帧配对比较"),
    code(
        """
        comparison = PROJECT_ROOT / "data" / "processed" / "evaluation" / "asteris_paper_comparison"
        summary_path = comparison / "profile_summary.csv"
        metrics_path = comparison / "metrics_160_vs_400.csv"
        if summary_path.exists():
            display(pd.read_csv(summary_path))
        if metrics_path.exists():
            display(pd.read_csv(metrics_path))
        """
    ),
]

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
