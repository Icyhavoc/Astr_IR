# RKZ50 红外图像二维背景扣除

本模块处理 `data/processed/flicker/90000002` 和 `90000003` 中的 160 帧
`flicker_corrected_*.fits`。输入发现使用精确 glob，因此 160 个
`flicker_model_*.fits` 不会进入本流程，也不读取 `processed_fits/Fixed_*`。

## 方法依据

流程参考 `paper/CEERS Epoch 1 NIRCam Imaging Reduction Methods and Simulations Enabling Early JWST(1).pdf`
第 12–14 页及 `docs/reference_background.png`：

1. 用稳健二维网格估计粗背景，压平大尺度起伏；
2. 以背景残差的 5×稳健 RMS 屏蔽亮像元；
3. 在大环尺度上估计初始背景，为源检测生成平坦图；
4. 从扩展源到紧致源进行多层检测和圆形膨胀，构造重源掩膜；
5. 在原图未掩膜区域以 Sigma 裁剪和 biweight location 估计最终二维背景；
6. 对粗网格做中值滤波和三次样条插值；
7. 扣除二维背景，并检查尺度相关 RMS、掩膜边缘残差、高频噪声和孔径流量。

CEERS 的参数针对 0.03 arcsec/pixel 多滤镜拼接图。本项目是单帧 1024×1024 探测器图像，
因此保留 `rough_box_size=100`、环内半径 80 和环宽 4 的大尺度框架，但将最终网格设为
32 像素，并提高紧致源层阈值、减小膨胀半径，以避免随机噪声被扩张成满图掩膜。

CEERS 使用精确环形中位数。本实现先对粗背景残差做稳健裁剪和亮源屏蔽，再用掩膜归一化
环形卷积估计同一空间尺度；这样可以在 160 帧 1024×1024 图像上高效运行。该环形结果只
用于源检测，最终科学背景仍来自重源掩膜后的稳健二维网格。

## 科学公式与质量门

背景在本任务中是需要永久扣除的量。float32 输出严格满足：

```text
background_subtracted = flicker_corrected_input - background_model
```

候选背景必须同时满足：

- 64 像素块背景位置散布至少下降 10%；
- 相邻像元差分估计的高频噪声不得增加超过 2%；
- 输入 CSV 中 SNR≥10 的目标星孔径流量必须可验证，且变化不得超过 1%；
- 背景模型全部为有限数。

若任一质量门失败，科学输出为输入图的 float32 副本，背景模型为零，并在统计表中记录拒绝原因。
低 SNR 帧仍记录测光变化，但不以不稳定的小分母百分比否决。

## 文件结构

```text
src/astr_ir/background/
├─ processor.py                     # 掩膜、背景、质量门、FITS/CSV 输出
└─ visualization.py                 # 背景、掩膜、RMS 与测光图
scripts/
├─ run_background.py               # 命令行批处理入口
├─ build_background_notebook.py    # 重建 Notebook
└─ build_background_task_document.py
tests/test_background.py
notebooks/background/01_background_subtraction.ipynb
docs/科创任务安排_背景扣除.docx
figures/background_output/
data/processed/background/
   ├─ background_statistics.csv
   ├─ 90000002/
   │  ├─ background_subtracted_*.fits
   │  └─ background_model_*.fits
   └─ 90000003/
      ├─ background_subtracted_*.fits
      └─ background_model_*.fits
```

输出 FITS 保留输入科学头和 `flicker` 流程已写入的 `FLK` 卡片，并新增 `BKG PROD`、
`BKG APPL`、`BKG BOX`、`BKG REDUC` 和 `BKG MASKFR`。本流程不保存额外 mask FITS；
blindmap、源掩膜、边缘掩膜仅用于背景估计和诊断图。

## 运行

```powershell
cd <project-root>
python -m pytest -q
python scripts/build_background_notebook.py
python scripts/run_background.py --overwrite
```

快速检查两组各一帧：

```powershell
python scripts/run_background.py --limit-per-sequence 1 --overwrite
```

参数集中在 `BackgroundConfig`。修改默认参数后，应同步 Notebook 参数格和
`scripts/build_background_notebook.py`，重新运行测试与全量批处理，并复核背景 RMS、高频噪声、源边缘残差和测光。

## 当前全量结果

- 160/160 帧完成；159 帧通过全部质量门并扣除背景。
- `90000003` 中 1 帧候选目标星流量变化为 1.188%，超过 1% 门限，因此安全退回原图/零模型。
- 已应用帧的 64 像素尺度背景位置散布下降 92.13%–94.39%，中位数 93.30%。
- 已应用帧最大高频噪声比为 1.00130。
- 最终掩膜占比为 19.01%–22.14%，中位数 20.11%。
- 源掩膜边缘扣除后偏差绝对值不超过 7.87 DN。
- 80 个 SNR≥10 帧最终最大绝对孔径流量变化为 0.8222%。
- 320 个输出 FITS 全部通过 Astropy 严格校验，科学头和 `FLK` 元数据均保留。
- `background_subtracted = input - background_model` 的最大 float32 误差为 0。
- 全项目自动测试为 `22 passed`。

`background_statistics.csv` 中的产品路径相对于 `data/processed/background/` 保存，不包含本机盘符或绝对路径。
