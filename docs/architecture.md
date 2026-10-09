# 项目架构与维护约定

## 主流程

`raw → flicker → background → noise2noise / asteris`。
Notebook 是实验入口，`scripts/run_*.py` 是四个对应的批处理入口，`src/astr_ir/` 承载可测试的算法。
辅助工具不再与主流程入口平铺，详见 [工具目录](../scripts/README.md)。

| 模块 | 职责 |
|---|---|
| `flicker/` | 掩膜、1/f 模型、质量门、科学图与模型图输出 |
| `background/` | 自动源掩膜、二维背景估计、质量门与扣除 |
| `noise2noise/` | 帧级切分、配对、2D DnCNN、训练与逐帧推理 |
| `asteris/paper_pipeline.py` | 当前 ASTERIS8：多曝光 8→8 训练、时间合成与科学共加 |
| `asteris/model.py` | 只读加载上一级原作者源码，记录源码校验和 |
| `dq.py`、`registration.py`、`blind_registration.py`、`split_manifest.py` | 坏像元标志、单次 FITS/DQ 读取、图像自配准与两分支共用的冻结帧清单 |
| `evaluation/`、`asteris/paper_evaluation.py` | 可选盲检、伪源注入、性能评估 |
| `asteris/catalog_overlay.py` | 已冻结结果上的星表标注，不参与训练或预处理 |

ASTERIS 的 `dataset.py`、`preprocessing.py` 是当前论文流程与历史实验共享组件。
`processor.py`、`inference.py` 仍由历史模型评估和测试引用，因此保留；不能因旧 Notebook 已删除而一并删掉。
N2N 主入口使用固定强度；历史目标测光标定函数只供旧实验追溯，不参与当前全盲流程。ASTERIS 准备阶段只物化 train/validation stack，测试曝光由推理阶段直接读取科学 FITS。

## 数据边界

- `data/raw/our_dataset/` 原始文件只读。
- 每一阶段只读取上一阶段的科学产品，不读取模型/残差产品，也不回写输入。
- 原图尺寸和 DQ 保持可审计；盲元不是零值观测。插值填充值不参与有效样本统计。
- 星表仅用于最终人工核验，不能作为已知弱源的位置先验。
- 派生文件统一在 `data/processed/` 的新实验目录；旧结果已归档，不能与新预处理或新 manifest 混用。
- 评估使用相同冻结测试曝光；不能把 80 帧联合检测与旧 16 帧 ASTERIS coadd 当作等曝光对比。

## 维护检查

1. 修改参数时同步处理器、主 Notebook 和 `scripts/notebooks/` 中对应构建器。
2. 仅改文件路径时保留 Notebook outputs，不执行训练或重新生成全部 Notebook。
3. 运行 `python -m pytest -q`；目录测试检查主入口、Notebook 语法、脚本路径和文档链接。
4. 修改科学算法时再按风险运行相应 `scripts/validation/` 校验及注入源实验。
5. 生成图、FITS、权重、缓存不新增进普通 Git；旧结果可在 `../_cleanup_backup/interface_optimization_20261009/` 恢复。

当前全盲预处理的背景回归及检测限制见 [运行记录](evaluation/blind_pipeline.md)。
