# 二维背景扣除

主入口：`notebooks/background/01_background_subtraction.ipynb`；
批处理：`scripts/run_background.py`；实现：`src/astr_ir/background/processor.py`。

## 输入与输出

只读取 `data/processed/flicker/{sequence}/flicker_corrected_*.fits`，
不读取 `flicker_model_*` 或原始曝光。

在 `data/processed/background/{sequence}/` 生成：

- `background_subtracted_*.fits`：N2N / ASTERIS 的科学输入。
- `background_model_*.fits`：背景模型，保留以审计扣除。
- 阶段根目录的 `background_statistics.csv`：状态、质量指标和产品路径。

float32 科学公式：`background_subtracted = flicker_corrected_input - background_model`。

## 方法

项目以 CEERS 背景流程为适配参考，示意见 [参考图](references/reference_background.png)，论文位于 code 上一级的 `paper/`。
当前本地实现：

1. 稳健粗网格估计背景，压平大尺度起伏。
2. 排除坏像元后按 5 倍稳健 RMS 检测亮结构。
3. 使用掩膜归一化环形卷积为源检测生成平坦图。
4. 多尺度自动检测与膨胀构造源掩膜，不使用星表位置。
5. 在未掩膜区域拟合 Sigma 裁剪/biweight 二维网格，经中值滤波和插值生成背景。
6. 从输入扣除背景，保留科学头、DQ 和阶段元数据。

这不是 CEERS 参数的直接照搬，也不使用精确环形中位数。
参数以 `BackgroundConfig` 为准：粗网格 100，环内半径 80、宽度 4，最终网格 64、中值滤波宽度 5。

## 质量门与已知限制

- 64 像素块背景位置散布至少下降 10%。
- 相邻像元差分噪声增加不超过 2%，背景模型必须有限。
- 不启用已知目标测光门。必要条件失败时输出输入的 float32 副本和零模型。

最新 400 帧重跑虽完成质量校验，但相较旧版 32 像素网格，十张保留同帧样本的背景起伏中位数约由 **12 DN 增至 29 DN**。
64 网格尚未证明更好；下一轮训练前仍需背景尺度和随机注入源对照。
详见 [全盲运行记录](evaluation/blind_pipeline.md)，不把旧 160 帧的测光门结果混作当前结果。

## 使用

```powershell
python scripts/run_background.py --help
python -m pytest -q tests/test_background.py tests/test_dq.py
```

旧 2026-08-20 诊断图和过时任务书已归档。未来 Notebook 生成的图仍写入 `figures/background_output/`。
本次清理不重跑、不修改背景参数，也不改写科学 FITS。
