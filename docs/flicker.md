# 1/f 条纹噪声校正

主入口：`notebooks/flicker/01_flicker_noise_correction.ipynb`；
批处理：`scripts/run_flicker.py`；实现：`src/astr_ir/flicker/processor.py`。

## 输入与输出

读取 `data/raw/our_dataset/{sequence}/*.fits` 的 400 张原始曝光。
不读取模型图或旧 `Fixed_*` 派生图；后者已归档到 code 外。

输出在 `data/processed/flicker/{sequence}/`：

- `flicker_corrected_*.fits`：供 background 使用的科学图。
- `flicker_model_*.fits`：条纹模型，供公式审计。
- 阶段根目录的 `flicker_statistics.csv`：质量门、回退状态和产品路径。

float32 科学公式：`flicker_corrected = original.astype(float32) - flicker_model`。

## 算法与有效像元

1. 合并 Dead/Noise blindmap、输入 DQ、非有限像元与边缘掩膜。
2. 从图像自动生成源掩膜；不读取星表或已知目标轨迹。
3. 暂时分离低频二维背景，比较行/列剖面以判断条纹方向。
4. 对有效像元计算逐行/列 Sigma 裁剪中位数，构建一维条纹模型。
5. 首选 5 点中值平滑；局部约束不通过时尝试 3 点和 1 点。
6. 扩展为二维模型，扣除并检查质量门。

低频背景只用于分离条纹，不在本阶段永久扣除。
派生 FITS 保持原尺寸并携带 `DQ` 扩展；坏像元不参与估计，后续阶段按 DQ 排除，不能当作零强度观测。

## 默认质量门

参数以 `FlickerConfig` 为准：

- 边缘宽度 24，背景网格 64；方向分数低于 1.6 时不校正。
- 选定方向的剖面散布至少下降 30%，高频噪声增加不超过 2%。
- 恶化行比例不超过 26%；增加超过 10 DN 的行不超过 13%；最坏单行增量不超过 80 DN。
- 已知目标测光门停用。任一必要质量门失败时输出输入的 float32 副本和零模型，并记录原因。

## 使用与当前记录

```powershell
python scripts/run_flicker.py --help
python -m pytest -q tests/test_flicker.py tests/test_dq.py
```

需要重跑时显式选择输入、输出及覆盖参数；本次目录清理未重跑处理。
Notebook 默认复用现有产品；重新生成的诊断图写入 `figures/flicker_output/`。
旧 2026-08-20 图像已归档，不再与当前 400 帧产品混放。

当前运行、背景回归和旧新对比入口见 [全盲流程记录](evaluation/blind_pipeline.md)。
旧 160 帧验收数值保存在归档文档中，不再作为当前结果重复列出。
