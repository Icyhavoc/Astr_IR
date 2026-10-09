# Noise2Noise 分支

主入口：`notebooks/noise2noise/01_noise2noise_self_supervised.ipynb`；
CLI：`scripts/run_noise2noise.py`；实现：`src/astr_ir/noise2noise/`。

## 输入、输出与当前状态

只读取 `background_subtracted_*.fits`。manifest 构建器会发现全部已完成 background 的序列。
当前 `data/processed/` 已清空；旧结果位于
`D:/Astr_IR/_cleanup_backup/interface_optimization_20261009/processed/`，不能当作新输入版本的结果。

新实验的 `data/processed/noise2noise_<run>/` 将包含：

- `checkpoints/`：best/last 权重和训练历史。
- `manifests/`：帧切分、帧对、归一化。
- `denoised/`、`residual/`：逐帧科学与残差 FITS。
- `noise2noise_statistics.csv`：逐帧结果。

旧评估也一并归档。新实验须在冻结参数后另做全盲注入评估。

## 当前代码的数据与模型约定

- 每个 80 帧序列采用 48 train / 2 guard / 12 validation / 2 guard / 16 test；先切帧再配对。
- 时间配对 lag 为 2–5，帧对不跨集合。
- 新 manifest 使用图像自动星点配准，不读取星表轨迹；坏像元在插值前排除。
- 128×128 patch 均匀随机取样，不按已知源位置偏置。
- 8 层、32 通道残差 DnCNN，科学图与有效像元通道输入，masked MSE 与 AdamW。
- 归一化从训练帧估计，验证集选择 checkpoint；测试集不参与选择。

科学输出：`noise2noise_denoised = background_subtracted_input - noise2noise_residual`。

## 固定强度与科学限制

主 CLI 要求显式 `--stage`。`infer` 使用 `--strength` 固定去噪强度（默认 1.0），不再调用历史高 SNR 星源测光标定；checkpoint 必须与冻结的 split manifest 一致。
新结果写入空白输出目录，上游 DQ 保留在科学输出中。

历史标定曾选择 α=0.23，旧测试相邻像元噪声比中位数约 0.7816；这些仅是归档记录。
固定强度 1.0 的通量传递、完整率和误检率尚未在当前输入上验证，不能宣称弱源检出改善。

## 检查与评估

```powershell
python scripts/run_noise2noise.py --help
python -m pytest -q tests/test_noise2noise.py
python scripts/run_noise2noise.py --stage prepare --output-root data/processed/noise2noise_<run>
```

伪源方法、保存表和绘图说明见 [独立评估](evaluation/source_evaluation.md)；
新旧数据边界见 [全盲记录](evaluation/blind_pipeline.md)。
Notebook 的旧执行输出保留供追溯，本次未重新执行训练或推理。
