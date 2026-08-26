# Astr_IR 红外图像处理

主流程：**flicker → background → Noise2Noise 或 ASTERIS8**。
主实验入口保留在 Notebook；可复用算法位于 `src/astr_ir/`。原始数据只读，派生产物写入 `data/processed/`。

## 主流程入口

| 阶段 | 主 Notebook | 命令行入口 |
|---|---|---|
| 1/f 校正 | [flicker](notebooks/flicker/01_flicker_noise_correction.ipynb) | `scripts/run_flicker.py` |
| 背景扣除 | [background](notebooks/background/01_background_subtraction.ipynb) | `scripts/run_background.py` |
| N2N 分支 | [noise2noise](notebooks/noise2noise/01_noise2noise_self_supervised.ipynb) | `scripts/run_noise2noise.py` |
| ASTERIS 分支 | [ASTERIS8 论文版，160/400 帧](notebooks/asteris/01_asteris8_paper_160_vs_400.ipynb) | `scripts/run_asteris.py --profile 160 或 400` |

两个模型分支都只读取 `background_subtracted_*.fits`，不相互串联；`flicker_model_*`、`background_model_*` 不作为后续科学输入。
ASTERIS 主入口现已统一为论文版；旧版 ASTERIS4/α 混合实验入口已移除。

## 目录

```text
notebooks/    flicker/ → background/ → noise2noise/ 或 asteris/
              evaluation/ 为可选的检测与对比，不是主流程必经阶段
src/astr_ir/  各阶段算法，以及共享 DQ、配准和评估模块
scripts/      四个主流程 CLI；辅助工具分目录存放
tests/        自动测试
docs/         阶段说明；evaluation/ 评估说明；references/ 方法参考图
data/         raw/ 原始数据；processed/ 科学产品、权重及对比记录
figures/      N2N/论文版历史结果图；过时预处理图已归档
```

辅助命令和清理记录见 [scripts/README.md](scripts/README.md)；模块职责见 [架构说明](docs/architecture.md)。
论文在项目上一级的 `../paper/`，不在 `code/` 内。

真实弱源人工核验：运行 [全盲评估 Notebook](notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb) 第 5 节，
将已有星表位置标在处理后图像上；输出见 `figures/catalog_validation_output/`，不参与处理或训练。

## 安装与检查

在 `code` 目录使用 Python 3.11 或 3.12：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_flicker.py --help
python scripts/run_background.py --help
python scripts/run_noise2noise.py --help
python scripts/run_asteris.py --help
```

上述检查不会启动训练。完整重跑、模型训练和覆盖已有产品均需另行显式执行；参数见各入口的 `--help`。
重建 Notebook 的脚本在 `scripts/notebooks/`，重建会清空该 Notebook 的历史输出，不应为了查看结果而运行。

## 当前数据状态与限制

- 原始集为 5 个序列、共 400 帧；最新 1/f 和背景产品已完成。旧预处理每序列每阶段保留两张对比图，见 [数据目录](data/README.md)。
- 星表只供最终人工核验；全盲预处理不按已知目标位置保护。DQ 坏像元不参与拟合、训练 loss、共加或测光。
- 现有 N2N/ASTERIS 权重和模型结果仍属于旧预处理；不要与新图像/配准清单混用。下一轮实验应另建输出目录。
- 新背景网格尚未证明更优：保留样本的 64 像素块背景起伏中位数约由 12 DN 增至 29 DN。训练前仍需背景尺度与注入源验证，详见 [全盲流程记录](docs/evaluation/blind_pipeline.md)。
- N2N 的历史推理入口仍调用目标测光强度标定，不适用于新全盲 manifest；本次整理没有修改其科学逻辑。见 [N2N 说明](docs/noise2noise.md)。论文版 ASTERIS 不使用该标定。

文件清理分两轮完成：先整理入口，再移出旧实验、派生图和过时材料；见 [数据清理记录](data/README.md) 与 [保留图像](figures/README.md)。
不重跑预处理、不训练、不改写科学像元或权重。归档位于 code 外，可恢复。
