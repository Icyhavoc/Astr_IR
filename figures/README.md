# 保留图像

当前共 40 张 PNG：原先保留的 12 张历史实验图，以及 14 张灰色掩膜星表标注图和 14 张仅显示用插值副本。

## 新增：真实弱源位置标注

入口是 [全盲评估 Notebook](../notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb) 第 5 节。
独立运行该节即可导出，不需要重跑前面的处理，也不训练、不联网。

`catalog_validation_output/{90000002,90000003}/` 每个序列包含：

- 4 张全幅图：`weighted_coadd`、`joint_score`、`asteris160`、`asteris400`，文件名以 `_catalog_overlay.png` 结尾。
- 3 页 `_weak_source_cutouts_01/02/03.png`：每页四个弱源、四种处理结果。
- `plotted_positions.csv`：沿用 W01–W12 标签、星表 ID 和转换后的零基坐标。
- `visualization_metadata.json`：输入 SHA-256、旧 WCS RMS、图像配准匹配数与二维 RMS。
- `display_interpolated/`：上述 7 张图的仅显示用插值副本，图上明确标注 DISPLAY-ONLY INTERPOLATED。

默认图将共加图自身 DQ 的 DO_NOT_USE 与非有限像元显示为不透明中性灰，不直接叠加原始探测器盲元表。
有效的部分覆盖像元保留；旧 FITS 缺少 DQ 时仅遮挡非有限值，不能将未标记的有限亮点认定为盲元。
展示副本只填补最多 16 像素、不接触边界的八连通缺口：用原有效邻域的高斯加权均值（sigma=1，半径 3 像素）。
有效像元不变，大缺口和边界继续显示灰色；每种产品的填补/保留像元数记录在 JSON 中。
两种版本采用相同的原有效像元显示拉伸与坐标有效性标记。插值不是观测，不能用于检测、测光或评价模型。
Notebook 第 5 节的 `SHOW_INTERPOLATED=False` 默认预览掩膜版，改为 `True` 可查看展示副本。

每个序列沿用此前冻结的 12 个弱源位置；圆圈不是检出结果，不移动到附近亮点。
新图采用图像星点求出的整体平移，旧模型图使用原坐标；不重采样或改写科学 FITS。
当前共加/检测为 80 帧，旧模型为 16 帧且输入版本不同，各图显示拉伸独立，不能直接据此宣称性能提升。
绘图在独立 CPU 进程中执行，避免本机 PyTorch/绘图库的 OpenMP 冲突。

## 原有历史图像

| 目录 | 数量 | 对应内容 |
|---|---:|---|
| `noise2noise_output/` | 3 | N2N 训练曲线、历史 α 标定、弱源示例 |
| `evaluation_output/noise2noise/` | 3 | 独立伪源评估的 PSF、完备度/纯度、测光误差 |
| `asteris_paper_output/` | 2 | 已分析的斑块与背景来源诊断 |
| `asteris_paper_output/catalog_overlays/` | 4 | 160/400 论文版真实源标注与弱源局部图 |

这些模型图属于旧预处理结果，不代表新 400 帧全盲流程已得到性能提升。
星表标注仅便于人工核验，不参与模型输入或保护。

19 张过时图已按原相对路径移入 `D:/Astr_IR/_cleanup_backup/assets_20260826/figures/`：
旧 flicker 9 张、background 6 张、旧 ASTERIS4 评估 3 张、早期 N2N 注入图 1 张。
空的 `asteris_output/` 也已移出。

未来重跑 Notebook 可重新创建 `flicker_output/` 和 `background_output/`，但应与生成该图的产品版本一致。
当前全盲预处理与联合检测的已执行展示仍在 [诊断 Notebook](../notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb)。
