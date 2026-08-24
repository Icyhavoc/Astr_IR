# 项目架构与维护约定

## 设计原则

1. **原始数据不可变**：`data/raw/our_dataset/` 只读，代码不得原地覆盖输入。
2. **代码与数据分离**：算法位于 `src/astr_ir/`，入口位于 `scripts/`，派生数据只写入 `data/processed/`。
3. **阶段依赖单向**：background 只读取 flicker 的科学校正产品，不读取条纹模型，也不回写前一阶段。
4. **公式可审计**：FITS 输出保持 float32 恒等式，并保留科学头和阶段元数据。
5. **配置有单一来源**：默认参数定义在 `FlickerConfig`、`BackgroundConfig`；Notebook 只显式覆盖需要展示的参数。
6. **失败安全**：质量门拒绝候选校正时输出输入图的 float32 副本与零模型，并记录状态。
7. **局部约束**：flicker 除全局剖面改善外，还限制恶化行占比、超过 10 DN 的恶化行占比和最坏单行增量；首选 5 点模型不通过时依次尝试 3 点与不平滑的 1 点模型。

## 模块职责

```text
src/astr_ir/flicker/processor.py
  掩膜、低频背景隔离、方向判断、条纹模型、质量门、FITS/CSV 输出

src/astr_ir/background/processor.py
  粗背景、环形平坦化、分层源掩膜、二维背景、质量门、FITS/CSV 输出

src/astr_ir/noise2noise/
  帧级切分、时间配对、配准、残差 DnCNN、训练、瓦片推理与 FITS/CSV 输出

src/astr_ir/asteris/
  帧级切分复用、2T 时间窗口、源保护 3σ clipping、训练集归一化、原始 ASTERIS4/8
  薄适配、3D patch、masked loss、双向时序推理、验证集强度标定与 FITS/CSV 输出

src/astr_ir/evaluation/
  与模型解耦的经验 PSF、伪源注入、盲检、一一匹配、完备度/纯度/光度统计和绘图

src/astr_ir/*/visualization.py
  只负责诊断图，不包含科学处理决策

scripts/run_*.py
  参数解析和批处理入口，不重复算法

notebooks/
  各处理阶段的主代码、展示、验收和可重复实验；由 scripts/build_*_notebook.py 生成
```

当前两个处理器规模尚可，配置和工具函数与流程强耦合，因此暂不为满足目录形式而机械拆成 `config.py`、`utils.py`。当共享工具函数开始被多个模块复用，或处理器继续增长时，再提取公共模块并增加对应测试。

## 数据流与命名

| 阶段 | 输入 | 科学输出 | 模型输出 | 统计表 |
|---|---|---|---|---|
| flicker | 原始 `*.fits` | `flicker_corrected_*.fits` | `flicker_model_*.fits` | `flicker_statistics.csv` |
| background | `flicker_corrected_*.fits` | `background_subtracted_*.fits` | `background_model_*.fits` | `background_statistics.csv` |
| noise2noise | `background_subtracted_*.fits` | `noise2noise_denoised_*.fits` | `noise2noise_residual_*.fits` | `noise2noise_statistics.csv` |
| asteris | `background_subtracted_*.fits` | `asteris_denoised_*.fits` | `asteris_residual_*.fits` | `asteris_statistics.csv` |
| source evaluation | 测试输入 + 冻结模型推理函数 | 不生成科学 FITS | 经验 PSF 与盲检目录 | `metrics_by_snr.csv` |

统计表中的产品路径均相对于各自阶段的输出根目录，避免写入本机盘符。

## 修改检查清单

- 修改核心参数时同步处理器、Notebook 参数格和 Notebook 构建器。
- 运行 `python -m pytest -q`。
- 重新生成并执行四个 Notebook，确认无错误单元。
- flicker：复核方向、剖面/功率谱、背景高频噪声、孔径测光、局部行恶化及回退尺寸分布。
- background：复核大尺度 RMS、高频噪声、源边缘残差和孔径测光。
- noise2noise：先按完整帧固定训练/验证/测试集，再配对与裁 patch；复核验证集强度标定、测试噪声、弱源 SNR和高 SNR 测光。
- asteris：沿用相同帧级 split；先验证源保护 clipping、blindmap 与 train-only normalization，再构造同集合 2T 窗口；先做 GPU smoke test，复核双向时序推理、测试噪声、弱源 SNR、高 SNR 测光与 FITS 恒等式。
- source evaluation：PSF只能来自训练集、阈值只能来自验证集，测试集报告盲检完备度、纯度、F1、光度/位置误差和未注入新候选。
- 对全部输出执行 Astropy 严格 FITS 校验，并检查 float32 公式误差为零。
- Noise2Noise 运行 `python scripts/validate_noise2noise.py`，检查不存在跨集合帧对且测试集从未参与 checkpoint 或强度选择。
- 通用伪源评估运行 `python scripts/validate_source_evaluation.py`，检查训练/验证/测试职责隔离和真值一一匹配。
- 可直接运行 `python scripts/validate_products.py` 完成产品、公式、测光、噪声及局部条纹剖面审计。
- 提交前用 `git status --short` 确认无 FITS、缓存、Office 锁文件和本机绝对路径。

## 版本和大数据

代码发布使用语义化版本。原始数据和大批量 FITS 产品不要进入普通 Git；若论文复现需要固定数据版本，推荐把数据发布到 Zenodo/机构存储，并在 release 中记录 DOI、SHA-256 清单、配置和代码 commit。需要协作同步大文件时可引入 DVC 或 Git LFS，但应避免把可再生的中间产品永久保存在 Git 历史。
