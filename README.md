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
data/         raw/ 原始数据；processed/ 是新实验的空白输出位置
figures/      新实验图像位置；旧结果图已归档
```

辅助命令和清理记录见 [scripts/README.md](scripts/README.md)；模块职责见 [架构说明](docs/architecture.md)。
论文在项目上一级的 `../paper/`，不在 `code/` 内。

真实弱源人工核验可在结果重建并冻结参数后运行 [全盲评估 Notebook](notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb) 第 5 节；星表不参与处理或训练。

弱源检出改进实验在同一 Notebook 第 6 节：逐阶段随机注入、两轮全盲背景、PSF 选择与残差再检测、局部噪声校准。
全部运行开关默认关闭，不训练、不覆盖旧结果；使用方法及未验证边界见 [V3 说明](docs/evaluation/weak_source_v3.md)。

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

- 原始集为 5 个序列、共 400 帧；派生结果现已移出 `processed/` 和 `figures/`，归档路径见 [数据目录](data/README.md)。当前没有可用于判断模型效果的主线新结果。
- 星表只供最终人工核验；全盲预处理不按已知目标位置保护。DQ 坏像元不参与拟合、训练 loss、共加或测光。
- 旧 N2N/ASTERIS 权重和模型结果位于可恢复归档，属于旧预处理；不要与新图像/配准清单混用。下一轮实验使用新输出目录。
- 新背景网格尚未证明更优：保留样本的 64 像素块背景起伏中位数约由 12 DN 增至 29 DN。训练前仍需背景尺度与注入源验证，详见 [全盲流程记录](docs/evaluation/blind_pipeline.md)。
- N2N 主入口现在使用显式固定去噪强度（默认 1.0），不会从星表或旧测光表选强度。该强度的通量保持和弱源效果尚待全盲注入验证，见 [N2N 说明](docs/noise2noise.md)。

本次只修改入口与数据边界并归档旧产物；没有重跑预处理或训练。归档位于 `../_cleanup_backup/interface_optimization_20261009/`，可恢复。
