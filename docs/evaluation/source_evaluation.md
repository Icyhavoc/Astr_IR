# 通用伪源注入-恢复评估

## 目标与边界

`astr_ir.evaluation` 是独立于训练模型的科学评估阶段。它不导入 PyTorch，也不假定 Noise2Noise 架构；核心只接收一个保持二维图像形状的推理函数：

```python
output = inference(input_image, valid_mask, frame_metadata)
```

因此同一套经验 PSF、注入、盲检、匹配和统计逻辑可以用于 Noise2Noise、Noise2Void、邻帧预测网络或其他自监督去噪模型。当前命令行脚本提供 Noise2Noise 适配器，新增模型只需增加适配器，不应复制评估算法。

## 防止评估泄漏

- 经验 PSF 只从 `train` 中 SNR≥10 的源构建；
- 匹配滤波检测阈值只在 `validation` 注入实验中选择；
- `guard` 不参与 PSF、阈值或最终指标；
- 最终恢复率、纯度和光度/位置误差只来自冻结的 `test`；
- 输入和模型输出使用同一个检测阈值、同一个输入派生空白掩膜和同一份注入真值。

当前160帧数据沿用 Noise2Noise 已固定的96/24/32/8帧 train/validation/test/guard 划分。数据增加后应优先采用整组80帧的序列级切分。

## 经验 PSF 与注入

当前点源 PSF 由48帧高信噪训练图像构建：对源中心做亚像素对齐、环带背景扣除、孔径流量归一化和逐像素中位数组合，再截断为非负、单位总流量的31×31核。测试不使用高斯假设，也不使用测试帧估计 PSF。

伪源被注入到背景扣除后的模型输入，位置满足：

- blindmap 有效且远离边缘；
- 远离已知目标和输入图中≥3.5σ的已有峰；
- 伪源间距至少36像素；
- 坐标含随机亚像素偏移；
- 目标亮度由注入前背景的匹配滤波噪声换算为指定 SNR。

默认测试 SNR 为2、3、4、5、7、10。每个SNR在32帧测试图中每帧注入8个源，共256个源；六档总计1536个物理注入。由于输入和输出各记录一次恢复状态，`injection_recovery.csv` 有3072行。

## 盲检与匹配

检测器先在整幅允许区域生成经验 PSF 匹配滤波显著性图，再查找局部极大值。检测过程不知道注入坐标。验证集在3.5–7.0σ扫描统一阈值，在输入和输出两种方法均满足目标纯度时最大化平均 F1；当前阈值为4.0σ。

检测目录与真值在2.5像素内进行匈牙利算法一一匹配，一个检测不能对应多个注入源。指标定义为：

```text
completeness = TP / (TP + FN)
purity       = TP / (TP + FP)
FDR          = 1 - purity
F1           = 2 * completeness * purity / (completeness + purity)
```

同时报告匹配滤波流量相对误差、天体测量误差、每帧假阳性数，以及未注入原图中仅在模型输出达到阈值的新候选峰。后者不能直接判定为假源或真源，仍需多帧一致性和原始数据回查。

置信区间使用按序列分层、以完整测试帧为簇的1000次 bootstrap，保留同一图像内多个注入源之间的相关性。

## 当前结果

| SNR | 输入完备度 | 输出完备度 | 输入纯度 | 输出纯度 |
|---:|---:|---:|---:|---:|
| 2 | 1.56% | 2.34% | 100.00% | 66.67% |
| 3 | 13.28% | 15.23% | 100.00% | 95.12% |
| 4 | 36.72% | 42.97% | 100.00% | 97.35% |
| 5 | 73.05% | 76.95% | 99.47% | 98.50% |
| 7 | 97.27% | 97.27% | 100.00% | 99.20% |
| 10 | 98.05% | 98.05% | 100.00% | 99.21% |

插值得到50%完备度极限由SNR 4.366降至4.207，改善0.159；90%完备度极限由6.400降至6.285，改善0.115。32张未注入测试图的源排除区外，输入有0个4σ峰，模型输出有3个仅输出候选。当前模型表现为温和的弱源完备度增益，同时存在小幅新增峰风险。

输入/输出使用同一份注入真值进行配对比较：SNR=4有16个“仅输出恢复”且没有“仅输入恢复”，完备度增益6.25个百分点，帧簇 bootstrap 95%区间为3.13–9.77个百分点；SNR=5对应10个和0个，增益3.91个百分点，区间1.56–6.64个百分点。McNemar精确检验的未分簇参考 p 值分别为3.1×10⁻⁵和0.00195；正式解释优先采用帧簇置信区间。

SNR=2–3的已恢复源存在明显正流量偏差，主要来自只选择越过检测阈值对象的选择效应（Eddington bias）；低SNR光度不能只对已检出样本解释为模型测光偏差。

## 运行和输出

```powershell
cd <project-root>

# GPU推理和科学统计
python scripts/evaluation/run_source_evaluation.py --model noise2noise --device cuda

# 在不导入GPU模型的干净CPU进程中绘图
python scripts/evaluation/plot_source_evaluation.py

# 严格检查数据隔离、计数、一一匹配、PSF和SNR=5质量门
python scripts/validation/validate_source_evaluation.py
```

数据输出位于：

```text
data/processed/evaluation/noise2noise/
├── evaluation_config.json
├── empirical_psf.fits
├── psf_training_diagnostics.csv
├── validation_threshold_calibration.csv
├── selected_threshold.json
├── injection_recovery.csv
├── blind_detections.csv
├── trial_metrics.csv
├── metrics_by_snr.csv
├── evaluation_summary.csv
├── paired_comparison_by_snr.csv
└── unmodified_test_catalog.csv
```

生成图位于 `figures/evaluation_output/noise2noise/`。这些文件均为可再生派生数据，受 `.gitignore` 保护。

增加到960帧后，建议把 `--test-repeats-per-snr` 提高到3–5，或相应增加每帧注入数，使每个SNR至少有1000个注入；同时把划分改为完整80帧序列级隔离。

Noise2Noise 清单构建会自动发现 background 输出中的新序列；通用评估从清单读取序列，不维护独立的硬编码数据集列表。
