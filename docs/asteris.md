# ASTERIS8 论文版

## 唯一主实验入口

- Notebook：`notebooks/asteris/01_asteris8_paper_160_vs_400.ipynb`
- CLI：`scripts/run_asteris.py`（原 `run_asteris_paper.py`）
- 实现：`src/astr_ir/asteris/paper_pipeline.py`
- 输出：`data/processed/asteris_paper_{160,400}/`

旧 ASTERIS4/α 混合 Notebook 和 CLI 已移除；其权重、FITS 和评估表已移至
`D:/Astr_IR/_cleanup_backup/assets_20260826/data/processed/asteris/` 及同层 `evaluation/asteris/`。
它们仍可恢复，但不再占用主流程数据目录；被引用的底层组件继续保留。
`run_asteris.py` 现在要求 `--profile 160` 或 `--profile 400`，不再接受旧入口的 `--model asteris4`、`--stage calibrate` 等参数。

## 模型与数据流

使用原作者 ASTERIS8 网络及官方初始化，输入来自 background 科学图：

1. 固定帧级切分，配准及 DQ 有效区处理。
2. 16 张独立曝光随机抽样并交错为 8 帧 input / 8 帧 target。
3. 时间轴及全 3D 的 3σ clipping、MSE 帧排序、全局 z-score、`/4 + 1`。
4. 两半独立中值居中，官方旋转/翻转及 input-target 交换。
5. `1e6 × (0.125 × SmoothL1(stack) + MSE(temporal mean))`，只在有效体素约简。
6. 测试曝光合并成 8 个时间 bin，网络输出时间平均成为科学共加图，不做 α 混合。

AdamW：学习率 `1.5e-4`、weight decay `1e-4`，余弦 `T_max=2,000,000`。
8 GB GPU 使用 AMP；反向传播去掉公共损失尺度 `1e6`，日志保留官方尺度。

原作者模型从上一级 `../Asteris/ASTERIS_THU-main/` 只读加载；checkpoint 记录源码 SHA-256。
缺少该源码树或官方初始化权重时，应先恢复依赖，不能换成旧模型继续跑。

## 切分与比较

每序列 80 帧固定为 `44 train / 16 validation / 16 test / 4 guard`。
160 profile 用前两个序列；400 profile 用五个序列。
主比较始终使用共同的 `90000002`、`90000003` 冻结测试曝光。

科学 FITS 保留有效覆盖与 DQ。网络内部有限填充值不算作真实测量；新的 manifest 不使用星表位置保护弱源。
处理、训练和推理不会把 catalog overlay 当作输入。

## 命令

以下 prepare 示例只构建数据，不训练；另建输出目录以免覆盖历史实验：

```powershell
python scripts/run_asteris.py --profile 160 --stage prepare --output-root data/processed/asteris_blind_160
python scripts/run_asteris.py --profile 400 --stage prepare --output-root data/processed/asteris_blind_400
```

训练、推理使用同一输出目录，分别指定 `--stage train`、`--stage infer`。
当前背景参数仍需验证，本次整理未执行上述命令。
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
原始结果在 `data/processed/evaluation/asteris_paper_comparison/`，Notebook 历史输出未清空。

2026-08-26 的 400 帧 1/f 与背景已改为全盲流程，以上权重、prepared stacks 和 coadd 尚未随之重做。
新实验必须重新 prepare/train，并使用同一版上游输入及等曝光评估；不能直接复用历史结果宣称提升。
背景起伏回归等限制见 [全盲流程记录](evaluation/blind_pipeline.md)。
