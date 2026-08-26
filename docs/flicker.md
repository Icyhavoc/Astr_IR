# RKZ50 红外图像 1/f 条纹噪声预处理

本模块实现《科创任务安排.docx》要求的二维红外科学图像行/列相关条纹校正。代码只读取
`data/raw/our_dataset/90000002` 与 `90000003` 中的 160 帧原始 FITS；现有
`processed_fits/Fixed_*` 不作为输入。

## 先看哪个文件

推荐直接打开并逐单元运行：

```text
notebooks/flicker/01_flicker_noise_correction.ipynb
```

Notebook 已执行并保存输出，分段顺序严格对应任务文档“主要工作”：

1. 自动或手动判断水平（row）/垂直（column）条纹；
2. 合并星源、blindmap 与边缘掩膜；
3. 估计并减去低频二维背景；
4. 对未掩膜像元计算逐行/列 Sigma 裁剪中位数；
5. 对一维条纹序列做中值低通平滑；
6. 扩展为二维条纹模型；
7. 扣除模型，并保证 `corrected = original - flicker_model`；
8. 比较行列中位数和一维功率谱；
9. 用 CSV 中的孔径参数重新测光，检查目标星流量。

## 目录结构

```text
src/astr_ir/flicker/
├─ processor.py                       # 主处理流程与 FITS/CSV 输出
└─ visualization.py                   # 所有诊断和验收图
scripts/
├─ run_flicker.py                     # 命令行批处理入口
└─ build_flicker_notebook.py          # 重新生成 Notebook 结构
tests/test_flicker.py                 # 11 个自动测试
notebooks/flicker/01_flicker_noise_correction.ipynb
figures/flicker_output/               # Notebook 生成的诊断图
data/processed/flicker/
   ├─ flicker_statistics.csv
   ├─ 90000002/
   │  ├─ flicker_corrected_*.fits
   │  └─ flicker_model_*.fits
   └─ 90000003/
      ├─ flicker_corrected_*.fits
      └─ flicker_model_*.fits
```

## DQ 如何处理

老师的原话“DQ 就我发你的 blindmap 生成的，你做的东西不需要这个”理解为：不需要额外生成、保存或读取
一套与 blindmap 重复的 DQ 数据。

因此最终代码中没有 `dq.fits`、DQ 扩展、DQ bit 定义或 `dq` 参数。实际实现为：

```python
detector_mask = (DeadBlindMap != 0) | (NoiseBlindMap != 0)
```

该 detector mask 直接与星源掩膜、边缘掩膜和非有限像元合并，且只用于“哪些像元不能参与条纹估计”。
这样既执行了质量屏蔽，也在每个派生 FITS 的 `DQ` 扩展中保留 `DO_NOT_USE`/`DETECTOR_BAD` 信息，避免文件离开本项目代码后盲元状态丢失。代码入口是
`load_detector_mask()` 与 `combine_masks()`。

## 算法和质量门

- 星源掩膜：CSV 中已知目标星圆形掩膜，加上高通图中的自动紧致源掩膜。
- 边缘掩膜：默认屏蔽四边各 24 像素。
- 二维背景：64×64 分块中位数、粗网格高斯平滑、三次插值。
- 方向分数：逐行/列剖面稳健标准差除以中位数的预期统计误差。
- 一维模型：Sigma 裁剪中位数后首选 5 点中值滤波，并移除直流分量；局部门失败时依次尝试 3 点和不平滑的 1 点剖面。
- 弱条纹：方向分数低于 1.6 时返回 `not_needed_weak_stripe`，模型为零。
- 改善门：选定方向的剖面稳健标准差必须下降至少 30%。
- 噪声门：高频背景噪声不得增加超过 2%。
- 测光门：SNR≥10 的正常恒星孔径流量变化必须可验证且不超过 1%；低 SNR 帧仍记录数值，但不使用不稳定的百分比否决。
- 局部门：恶化行比例不超过 26%，增加超过 10 DN 的行比例不超过 13%，最坏单行增量不超过 80 DN。
- 任一质量门失败：输出原图的 float32 副本和零模型，并在统计表中记录原因，不强行校正。

所有输出 FITS 为 float32（`BITPIX=-32`），保留并规范化原始科学头，同时写入 `FLK PROD`、
`FLK DIR`、`FLK APPL`、`FLK SCORE` 和 `FLK REDUC` 元数据。float32 输出层面严格满足：

```text
flicker_corrected = original.astype(float32) - flicker_model
```

## 当前全量结果

- 160/160 帧完成，全部自动选择 `row`，即图像中的水平条纹。
- 155 帧通过全部质量门并校正；102 帧使用首选 5 点剖面，53 帧因局部门回退到 1 点剖面，3 点剖面未成为最终候选。
- 5 帧安全退回原图/零模型：2 帧的全部候选均违反测光门，3 帧没有同时满足局部与测光约束的候选。
- 已校正帧的行剖面稳健标准差下降 50.81%–100%，中位数为 57.79%，全部超过 30%。
- 已校正帧最大高频背景噪声比为 1.00045，没有明显增加。
- 80 个 SNR≥10 的正常星帧最终最大绝对流量变化为 0.9722%。
- 已校正帧每帧局部恶化行数中位数为 233，增加超过 10 DN 的行数中位数为 109；最大值分别为 253 和 126，最坏单行增量由旧方案的 146.76 DN 限制到 78.87 DN。
- 320 个输出 FITS 均通过 Astropy 严格 FITS 校验。
- `corrected = original - model` 的最大 float32 误差为 0。

逐帧详细指标和 4 个退回帧的候选测光变化见 `data/processed/flicker/flicker_statistics.csv`。
统计表中的产品路径相对于 `data/processed/flicker/` 保存，避免记录本机绝对路径，便于跨平台复现。

## 运行方法

在项目根目录启动 Jupyter：

```powershell
jupyter notebook notebooks/flicker/01_flicker_noise_correction.ipynb
```

命令行全量运行：

```powershell
python scripts/run_flicker.py --overwrite
```

手动指定方向：

```powershell
python scripts/run_flicker.py --direction row --overwrite
python scripts/run_flicker.py --direction column --overwrite
```

只处理每组前 2 帧用于快速检查：

```powershell
python scripts/run_flicker.py --limit-per-sequence 2 --overwrite
```

运行测试：

```powershell
python -m pytest -q
```

## 关键参数

参数集中在 `FlickerConfig`。通常只需修改 Notebook 中的：

```python
config = FlickerConfig(direction="auto")
```

若要调试阈值，可显式设置 `background_block_size`、`profile_smooth_size`、
`fallback_profile_smooth_sizes`、`max_local_worse_fraction`、
`max_local_worse_over_threshold_fraction`、`max_local_increase_dn`、
`min_direction_score`、`min_relative_improvement` 和 `max_photometry_change`。修改后应重新运行 Notebook
及自动测试，并检查 `figures/flicker_output/` 中的诊断图。

## 注意事项

- 当前 FITS 没有完整标准 WCS；本算法完全在探测器像素坐标中工作，不依赖 WCS 或多帧配准。
- CSV 只有已知目标星逐帧测量，不是完整源目录；自动源掩膜用于补充保护其它明显紧致源。
- `90000002` 的目标星本身为低 SNR，百分比测光变化会被小分母放大，应优先查看绝对流量和输入状态。
- 低频二维背景只参与条纹估计，最终输出不会将它作为天文背景扣除。
