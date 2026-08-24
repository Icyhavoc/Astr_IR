# 背景扣除后 FITS 的 Noise2Noise 自监督去噪与弱源检测基线

## 科学目标

本阶段只读取 `data/processed/background/9000000{2,3}/background_subtracted_*.fits`，不读取任何背景或 flicker 模型。模型通过同一序列中相同天区、不同随机噪声实现的时间帧对训练，目标是在控制流量偏差和假源的前提下降低随机噪声、提高弱源检出率。

科学输出严格满足：

```text
noise2noise_denoised = background_subtracted_input - noise2noise_residual
```

## 数据切分与防泄漏

两个序列各有 80 帧。每个序列按时间顺序独立切分：

```text
0–47    train       48 帧
48–49   guard        2 帧
50–61   validation  12 帧
62–63   guard        2 帧
64–79   test        16 帧
```

总计训练 96 帧、验证 24 帧、测试 32 帧和隔离 8 帧。切分在建立帧对和 patch 之前完成；任何帧对都不能跨集合。背景高频残差的时间相关审计显示 lag≥2 的相关中位数约 0.006–0.018，而高 SNR 序列 lag=1 约为 0.065，因此训练只使用 lag=2、3、4、5。

`90000002` 为弱源序列，输入 SNR 中位数约 4.10；`90000003` 为高 SNR 流量保真序列，中位数约 68.80。低 SNR 序列的逐帧质心存在异常跳变，配准轨迹仅由 accepted 帧进行稳健线性拟合。训练和验证排除上游质量门拒绝帧，测试块保持预先固定。

固定清单位于：

```text
data/processed/noise2noise/manifests/split_manifest.csv
data/processed/noise2noise/manifests/pair_manifest.csv
```

清单构建器会自动发现 `data/processed/background/` 下包含 `background_subtracted_*.fits` 的序列目录，不再固定为两个序列名。加入新的80帧目录并完成上游处理后，重新运行 `--stage prepare` 即可纳入清单；正式扩容时仍应改用完整序列级训练/验证/测试隔离。

## 模型与训练

- 小型 8 层、32 通道残差 DnCNN，输入为归一化科学图和 blindmap 有效像元通道；
- 输出预测噪声，网络内部执行输入减噪声；
- blindmap、非有限像元和配准边界不进入损失；
- 128×128 patch，50% 目标星附近、50% 随机背景；
- 仅使用翻转增强，不旋转探测器行列方向；
- masked MSE、AdamW、最大 30 epoch、固定验证 patch、早停和最佳 checkpoint；
- 每个序列的归一化尺度只从训练帧拟合，约为 245 DN。

最佳 checkpoint 来自第 29 epoch，验证 masked MSE 为 0.707850。

## 验证集去噪强度标定

网络完整残差在测试前不得直接采用。只使用 `90000003` 的高 SNR 验证帧，在 α=0.05–1.00 上扫描：

```text
final = input - α × predicted_noise
```

选择满足验证集最大绝对孔径流量变化不超过 1% 的最大 α。当前选择为 `α=0.23`：验证集最大流量变化 0.9914%，噪声比中位数 0.7817。测试集不参与 checkpoint、α 或检测阈值选择。

## 当前测试集结果

- 32 帧测试集相邻像元噪声比中位数：0.781632，即下降约 21.84%；
- 弱源序列孔径 SNR 中位数：5.52 → 6.86，SNR 比中位数 1.2584；
- 高 SNR 测试帧最大绝对流量变化：0.7721%；
- 320 个去噪/残差 FITS 全部通过严格校验，float32 公式最大误差为 0。

伪源评估已经从模型目录迁移到独立的 `astr_ir.evaluation`，采用训练集经验 PSF、验证集阈值、测试集整图盲检和真值一一匹配。当前 SNR=4 完备度为 36.72%→42.97%，SNR=5 为 73.05%→76.95%；详细完备度、纯度、置信区间和光度结果见 [source_evaluation.md](source_evaluation.md)。

## 运行

```powershell
cd <project-root>

# 一次完成清单、训练、验证集强度标定和全量推理
python scripts/run_noise2noise.py --stage all --overwrite

# 分阶段运行
python scripts/run_noise2noise.py --stage prepare
python scripts/run_noise2noise.py --stage train
python scripts/run_noise2noise.py --stage infer --overwrite

# 独立的通用伪源评估
python scripts/run_source_evaluation.py --model noise2noise --device cuda
python scripts/plot_source_evaluation.py

# 严格验收
python scripts/validate_noise2noise.py
python scripts/validate_source_evaluation.py
python -m pytest -q

# 重建验收 Notebook
python scripts/build_noise2noise_notebook.py
```

模型 checkpoint 和科学数据受 `.gitignore` 保护，不进入普通 Git。复现时必须保留代码 commit、配置、split/pair manifest、checkpoint SHA-256 和归一化/α 标定文件。
