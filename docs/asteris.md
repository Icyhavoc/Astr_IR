# RKZ50 仪器专用 ASTERIS

## 唯一主实验入口

- Notebook：`notebooks/asteris/01_rkz50_instrument_asteris.ipynb`
- CLI：`scripts/run_asteris.py`（原 `run_asteris_paper.py`）
- 实现：`src/astr_ir/asteris/paper_pipeline.py`
- 输出：新实验默认在 `data/processed/asteris_rkz50_field_split/`，正式实验仍建议添加独立 run 名称

旧 ASTERIS4/α 混合 Notebook 和 CLI 已移除；其权重、FITS 和评估表已移至
`D:/Astr_IR/_cleanup_backup/assets_20260826/data/processed/asteris/` 及同层 `evaluation/asteris/`。
它们仍可恢复，但不再占用主流程数据目录；被引用的底层组件继续保留。
旧 160/400 结果只用于历史追溯。正式 CLI 不再按训练帧数命名 profile，而使用 `prepare → train → select → infer` 四个明确阶段。

## 模型与数据流

当前实验分支默认使用原作者 Restormer/ASTERIS 系列中的紧凑 ASTERIS4：`f_maps=8`、blocks=`1/1/2`、refinement block=`1`。完整 ASTERIS8 仍可通过 `--network full8` 作为容量对照。两者都针对 RKZ50 地基红外数据从随机权重开始训练；输入来自 background 科学图：

1. 按独立天区冻结切分，随后进行配准及 DQ 有效区处理。
2. 紧凑版从 8 张独立曝光随机抽样并交错为 4 帧 input / 4 帧 target；完整 ASTERIS8 仍为 8/8。
3. 时间轴及全 3D 的 3σ clipping、MSE 帧排序、全局 z-score、`/4 + 1`。
4. 两半独立中值居中，官方旋转/翻转及 input-target 交换。
5. `1e6 × (0.125 × SmoothL1(stack) + MSE(temporal mean))`，只在有效体素约简。
6. 测试曝光按模型深度合并成 4 或 8 个时间 bin，网络输出时间平均成为科学共加图，不做 α 混合。

AdamW：学习率 `1.5e-4`、weight decay `1e-4`，余弦 `T_max=2,000,000`。
8 GB GPU 使用 AMP；反向传播去掉公共损失尺度 `1e6`，日志保留官方尺度。

原作者网络源码从上一级 `../Asteris/ASTERIS_THU-main/` 只读加载；checkpoint 记录源码 SHA-256、`weight_initialization=random` 和 `jwst_weights_used=false`。
缺少该源码树时应先恢复依赖，不能换成旧模型继续跑。发布的 `ASTERIS8_nrcshort` 权重来自 JWST NIRCam，与 RKZ50 的像元尺度、PSF、大气背景和探测器噪声域显著不同，正式主流程禁止加载；若研究迁移学习，只能另建明确命名的域迁移消融，不能作为主结论。

## 切分与模型选择

当前数据实际只有四个独立指向；`90000005_1` 与 `90000005_2` 是同一天区的两次访问。正式切分固定为：

- train：`90000002`、`90000004`；
- validation：`90000003`；
- test：`90000005_1`、`90000005_2`，两组不得拆开。

训练保存候选 epoch，但不会把最低 validation MSE 命名为正式最佳模型。`select` 阶段读取冻结的全盲注入评估表，只在每图误检数不超过预算且中位通量响应位于容差内的候选中，最大化注入完整率。评估必须使用与输入共加完全相同的曝光、背景、DQ、配准和 matched-filter 检测器；星表不得参与阈值或 checkpoint 选择。

科学 FITS 保留有效覆盖与 DQ。网络内部有限填充值不算作真实测量；新的 manifest 不使用星表位置保护弱源。
处理、训练和推理不会把 catalog overlay 当作输入。

## 命令

以下 prepare 示例只构建 train/validation stack，不训练，也不物化测试 stack。输出目录必须为空：

```powershell
python scripts/run_asteris.py --stage prepare --output-root data/processed/asteris_rkz50_field_split_run1
python scripts/run_asteris.py --network compact4 --stage train --output-root data/processed/asteris_rkz50_field_split_run1
python scripts/run_asteris.py --stage select --output-root data/processed/asteris_rkz50_field_split_run1 `
  --selection-table data/processed/evaluation/run1/checkpoint_detection_metrics.csv
python scripts/run_asteris.py --stage infer --output-root data/processed/asteris_rkz50_field_split_run1
```

主 CLI 不提供 JWST 权重初始化开关，训练固定从随机权重开始。正式推理必须使用 `selected_checkpoint.json` 冻结的 checkpoint；直接把 last checkpoint 或最低 MSE checkpoint 用作正式结果会被拒绝。checkpoint 在推理时同时核对冻结 split manifest 与选择记录。当前背景参数仍需验证，本次未运行 prepare、正式训练或推理。

紧凑版默认使用 64×64 裁块、每个训练序列每 epoch 256 个随机组合，validation 为 32 个固定组合。增加组合数只改善同一天区内的覆盖，不等价于增加独立天区；当前只有两个训练天区，因此不会把扩大 epoch/裁块数当作主要提升来源。
紧凑版的参数量与 smoke-test 记录见 [实验说明](evaluation/rkz50_compact_asteris4.md)。
辅助的多曝光伪源评估、160/400 汇总、星表标注都在 `scripts/evaluation/`。

## 已有历史结果（旧预处理）

| 指标 | 160 帧 | 400 帧 |
|---|---:|---:|
| 最佳 epoch | 19 | 9 |
| 最佳 validation loss（官方尺度） | 11727.5902 | 11656.1913 |
| 共加噪声比中位数 | 0.347099 | 0.336831 |
| SNR 4 completeness | 0.8750 | 0.6875 |
| SNR 4 purity | 0.4667 | 0.4783 |
| SNR 4 F1 | 0.6087 | 0.5641 |
| SNR 5 completeness | 1.0000 | 1.0000 |
| SNR 5 purity | 0.5000 | 0.5714 |

旧实验增加训练帧数改善了噪声及部分纯度指标，但没有一致提升临界弱源检出率。
原始结果已归档到 `D:/Astr_IR/_cleanup_backup/interface_optimization_20261009/processed/evaluation/asteris_paper_comparison/`，Notebook 历史输出未清空。

以上权重、prepared stacks 和 coadd 也已归档，尚未用当前全盲预处理重做。
新实验必须重新 prepare/train，并使用同一版上游输入及等曝光评估；不能直接复用历史结果宣称提升。
背景起伏回归等限制见 [全盲流程记录](evaluation/blind_pipeline.md)。
