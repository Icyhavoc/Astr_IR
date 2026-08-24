# ASTERIS 时空自监督去噪

## 实现边界

ASTERIS 与 Noise2Noise 是并列模型，不共享网络结构。Noise2Noise 使用二维残差 DnCNN 逐帧推理；ASTERIS 使用原作者的 3D Restormer-style Transformer U-Net，同时建模时间和空间。主要可执行代码位于 `notebooks/asteris/01_asteris_self_supervised.ipynb`，Notebook 调用 `src/astr_ir/asteris/` 中可测试的底层组件。

本项目不复制或修改原作者模型。`astr_ir.asteris.model` 从相邻的只读源码树动态加载：

```text
D:/Astr_IR/Asteris/ASTERIS_THU-main/asteris/ASTERIS_net_4.py
D:/Astr_IR/Asteris/ASTERIS_THU-main/asteris/ASTERIS_net_8.py
```

checkpoint 保存所用上游源码的 SHA-256，便于确认模型实现未发生漂移。

## 数据流

```text
background_subtracted_*.fits
        ↓
固定帧级 train/guard/validation/guard/test 切分
        ↓
配准 + blindmap/边缘/非有限像元 mask + 已知源 mask
        ↓
源保护的全局 3σ clipping
        ↓
仅用训练集拟合的 sequence mean/std
        ↓
连续 2T 帧窗口：偶数帧 input，奇数帧 target
        ↓
同步空间增强 + 3D patch + masked loss
        ↓
ASTERIS4/ASTERIS8 直接目标预测
        ↓
双向时序、重叠加权推理
        ↓
仅用高 SNR 验证集标定 prediction strength α
        ↓
asteris_denoised_*.fits + asteris_residual_*.fits
```

不会重复执行 flicker 校正或背景扣除，也不会读取 `flicker_model_*`、`background_model_*` 或原始 FITS 作为训练科学输入。

## ASTERIS4 与 ASTERIS8

| 模型 | 网络输入深度 T | 原始连续帧窗口 | 编码层级 | 空间尺寸约束 | 建议用途 |
|---|---:|---:|---:|---:|---|
| ASTERIS4 | 4 | 8 | 3 | 4 的倍数 | RTX 4060 8 GB 初始实验 |
| ASTERIS8 | 8 | 16 | 4 | 8 的倍数 | 显存验证后扩展实验 |

两者均包含 3D overlap patch embedding、MDTA attention、GDFN、3D down/up-sampling、U-Net skip connection、refinement blocks 和末端残差连接。`patch_t` 指网络输入深度；构造自监督 input/target 前必须读取 `2 × patch_t` 个连续帧。

## 切分、配对与泄漏控制

ASTERIS4 的每个 80 帧序列沿时间顺序固定切为：

```text
48 train / 2 guard / 12 validation / 2 guard / 16 test
```

ASTERIS8 的一个 input/target 窗口需要 16 个不同帧，因此 12 帧验证块不足。其切分调整为
`44 train / 2 guard / 16 validation / 2 guard / 16 test`；最后 16 帧冻结测试块与 N2N、
ASTERIS4 完全相同，仍可逐帧公平比较。

窗口只在同一 sequence、同一 split 内构造。先切完整帧，再构造窗口，最后在线裁 patch。input 与 target 的 frame ID 不重叠，guard 帧不训练、不验证、不测试。上游质量门回退产品仍是可审计的科学 FITS 副本：它们不进入归一化拟合和训练，但验证集、测试集完整保留，避免选择性删除困难帧。训练集训练参数；验证集选择 checkpoint、输出模式和后续科学门；测试集只做最终报告。

## 3σ clipping 与 mask

默认执行全局 3σ clipping，但阈值只由下列区域之外的像元估计：

- detector dead/noisy blindmap；
- 非有限像元；
- 图像边缘；
- 测量表中已知源位置的保护圆。

源保护区域不会被 clipping。被 clipping 的异常像元可以作为有界的网络输入，但会从 loss mask 中排除；blindmap 和非有限像元同时从输入有效区及损失中排除。策略写入 `manifests/clipping_policy.json`；`manifests/clipping_audit.csv` 对两个序列的代表性训练窗口逐项比较 clipping 前后的源峰值、流量、moment FWHM 和 SNR，并要求源区零 clipping、四项变化为零；每帧推理统计表另记录 clipping fraction。

时间轴 3σ clipping 已实现为可选项，但默认关闭。8/16 帧短序列中的 seeing、PSF 和配准变化可能被误判为时间异常，且未编目弱源不能被已知源 mask 保护。只有当验证集的峰值、孔径流量、FWHM 和 SNR 质量门全部通过时，才应启用 `--temporal-clip`。

## 归一化

每个 sequence 的 mean/std 只从训练帧、非源、非盲点、非边缘像元拟合。验证集和测试集复用冻结统计量，不重新估计。数值保存在：

```text
data/processed/asteris/manifests/normalization.json
data/processed/asteris/manifests/normalization_audit.csv
```

## 训练目标与输出公式

原作者网络的 forward 已包含：

```text
direct_prediction = input + learned_correction
```

原始训练代码直接把该结果与奇数帧 target 比较，因此默认 `output_mode=direct`。训练损失保留原项目的思想，同时严格应用 mask：

```text
loss = masked SmoothL1(full stack) + masked MSE(temporal mean)
```

正式推理先保留 α=1 的原始直接预测，然后只使用 `input_snr >= 10` 的验证帧扫描 α=0–1，
选择满足最大绝对光度变化不超过 1% 的最大 α。当前正式训练选择 `α=0.15`；测试集没有参与选择。
`residual` 模式只用于验证集受控对比，不能根据测试集选择。科学 FITS 统一记录并验证：

```text
asteris_residual = background_subtracted - asteris_denoised
asteris_denoised = background_subtracted - asteris_residual
```

推理为覆盖全部帧，会同时执行 even→odd 与 odd→even 两个方向，并对空间和时间重叠预测加权平均。最终结果逆配准回每个输入 FITS 的原生坐标。

## 运行

主入口是 Notebook：

```text
notebooks/asteris/01_asteris_self_supervised.ipynb
```

同一逻辑也可通过 CLI 自动化：

```powershell
python scripts/run_asteris.py --stage prepare
python scripts/run_asteris.py --stage train --model asteris4 --device cuda
python scripts/run_asteris.py --stage infer --model asteris4 --device cuda
python scripts/run_asteris.py --stage calibrate --model asteris4 --device cuda
python scripts/run_asteris.py --stage evaluate --model asteris4 --device cuda
python scripts/run_asteris.py --stage all --model asteris4 --device cuda
```

输出只写入 `data/processed/asteris/`：

```text
manifests/   checkpoints/   raw_predictions/   raw_residuals/
denoised/    residuals/     asteris_statistics.csv
```

开始完整训练前先运行：

```powershell
python -m pytest -q tests/test_asteris.py
python scripts/run_asteris.py --stage train --epochs 1 --train-samples-per-epoch 2 --validation-samples 2 --f-maps 4 --device cuda
```

默认 ASTERIS4、64×64 patch、batch size 1 是面向 8 GB 显存的保守起点。实际峰值显存取决于 PyTorch/CUDA 版本、feature maps 与 Transformer blocks，不能仅按输入数组大小估计。完整训练须在 smoke test 后进行。

## 公平科学评估

ASTERIS 与 N2N 都读取同一批 `background_subtracted_*`，使用相同帧级 split 和 `astr_ir.evaluation` 的盲伪源注入/恢复接口。至少报告：

- 90000002：SNR、完备度、纯度、F1、位置误差、测光变化；
- 90000003：相邻像元噪声比、峰值、孔径流量、FWHM、位置稳定性和假源数；
- 所有序列：clipping fraction、FITS 公式误差、checkpoint hash。

通用评估的单图 callable 会把同一注入图重复到所有时间 slice，确保真值存在于每个上下文帧；正式科学 FITS 推理则使用真实相邻帧。两种评估语境必须在论文或报告中区分。

## 已知限制

- 只有 2×80 帧，小样本 Transformer 容易过拟合，必须依赖验证早停和盲测试。
- ASTERIS 比二维 N2N 更耗显存和时间；ASTERIS8 不应在未测显存时直接完整训练。
- 插值配准和逆配准会引入小幅相关噪声，应与 N2N 使用同一坐标处理约定。
- 当前只保护测量表中的已知目标；未编目弱源是关闭 temporal clipping 的主要原因。

## 当前正式 ASTERIS4 结果

- 最佳 checkpoint：epoch 29，validation loss `0.547411`；
- 验证集选择 `α=0.15`：噪声比中位数 `0.8519`，SNR 比 `1.1688`，最大绝对光度变化 `0.8207%`；
- 冻结测试 90000002：噪声比中位数 `0.8519`，孔径 SNR 中位数约从 `5.52` 到 `6.22`；
- 冻结测试 90000003：噪声比中位数 `0.8521`，SNR 比中位数 `1.1667`；最大绝对光度变化 `1.74%`，高于验证门，属于需报告的泛化偏差，未据此回调 α；
- 盲伪源评估：SNR 4 完备度从 `36.7%` 到 `48.0%`、纯度 `94.6%`；50%/90% 完备度检测极限分别改善约 `0.31/0.27` SNR；未注入输出新增 6 个候选，说明增益伴随少量假阳性。
