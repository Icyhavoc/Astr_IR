# Noise2Noise 分支

主入口：`notebooks/noise2noise/01_noise2noise_self_supervised.ipynb`；
CLI：`scripts/run_noise2noise.py`；实现：`src/astr_ir/noise2noise/`。

## 输入、输出与保留数据

只读取 `background_subtracted_*.fits`。现有历史实验使用前两个序列共 160 帧；
新的 manifest 构建器会发现全部已完成 background 的序列。

`data/processed/noise2noise/` 保留：

- `checkpoints/`：best/last 权重和训练历史。
- `manifests/`：帧切分、帧对、归一化、历史强度标定。
- `denoised/`、`residual/`：各 160 张科学与残差 FITS。
- `noise2noise_statistics.csv`：逐帧结果。

当前采用的伪源评估在 `data/processed/evaluation/noise2noise/`。
旧模型根目录的两张早期评估表及对应旧图已归档，避免两套不同格式被误认为同一结果。

## 当前代码的数据与模型约定

- 每个 80 帧序列采用 48 train / 2 guard / 12 validation / 2 guard / 16 test；先切帧再配对。
- 时间配对 lag 为 2–5，帧对不跨集合。
- 新 manifest 使用图像自动星点配准，不读取星表轨迹；坏像元在插值前排除。
- 128×128 patch 均匀随机取样，不按已知源位置偏置。
- 8 层、32 通道残差 DnCNN，科学图与有效像元通道输入，masked MSE 与 AdamW。
- 归一化从训练帧估计，验证集选择 checkpoint；测试集不参与选择。

科学输出：`noise2noise_denoised = background_subtracted_input - noise2noise_residual`。

## 尚未完成的全盲适配

CLI 的 infer/all 仍调用历史目标测光强度标定，不能直接用于新的全盲 manifest。
本次仅清理文件和文档，没有修改这一算法；不要运行旧 all 命令后把结果当作新全盲基线。

现有权重和 FITS 对应旧预处理。历史标定选择 α=0.23，测试相邻像元噪声比中位数约 0.7816；
这些是历史记录，不代表新 400 帧数据上的性能。
当前 background 已更新，因此历史科学公式校验不能直接与新 background 混用。

## 检查与评估

```powershell
python scripts/run_noise2noise.py --help
python -m pytest -q tests/test_noise2noise.py
```

伪源方法、保存表和绘图说明见 [独立评估](evaluation/source_evaluation.md)；
新旧数据边界见 [全盲记录](evaluation/blind_pipeline.md)。
保留 Notebook 的已执行输出供查看，本次未重新执行训练、标定或推理。
