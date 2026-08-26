# 弱源检出 V3：实现与使用边界

入口：[全盲评估 Notebook 第 6 节](../../notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb)。
主流程仍是 flicker → background → N2N 或 ASTERIS；新增功能是独立实验/评估支路。
V3 初始实现只改代码并测试，未启动训练、未改网络或损失函数、未覆盖现有 FITS。
2026-08-26 已启动独立的真实数据验证，进展/结果见 Notebook 第 7 节和 [实测记录](validation_20260826.md)。
validation 诊断与独立 test 集性能验收须区分，不把模拟源恢复等同于真实星表弱源已检出。
第 1–5 节、旧检查点和旧共加图仍是历史结果，不代表 V3 输出。

## 新增内容

1. **逐阶段配对注入**：随机位置、多个通量档，覆盖均匀位置和自动亮星附近；在原始曝光内存副本注入。
   分别经过 raw、flicker、单轮背景、两轮背景和可选 ASTERIS，记录盲恢复、定位偏差、配对通量响应。
   处理函数不接受注入真值或真实星表；只有注入生成与事后评分能看到模拟源位置。
   缺省采用背景主导的确定性注入；只有明确提供电子/DN 增益才加入源泊松噪声。
2. **两轮背景**：第一轮粗处理后按图像自动配准共加，以高通后的共加图寻找紧凑候选源，扩张并反投影其掩膜。
   高通只用于生成掩膜，不作用于科学输出。掩膜只迭代一次，超过有效区 40% 时拒绝额外掩膜并记录。
   第二轮重新拟合原 flicker 输入，保持 `science = input - background_model`，不会二次重复扣背景。
   用于模型训练/测试时必须提供 `--split-manifest`，源掩膜按冻结 train/validation/test/guard 分开生成，避免跨划分信息泄漏。
   32/64 网格 × 1/3/5 滤波可用同种子消融；尚未自动选择“最优”参数，旧默认 64/5 保留为实验起点。
3. **PSF 模型选择**：从自动发现的孤立亮星建立样本，交替分为形状拟合与验证子集。
   比较圆高斯、椭圆高斯、Moffat 和经验 PSF。经验核至少需要 8 个可用样本；不足 4 个时明确回退圆高斯。
   自动模式要求验证误差至少改善 5% 才替代圆高斯；每帧的核和诊断均可追溯。
   目前每帧一个核，未实现视场内空间变化的 PSF 网格。
   本次只读检查了 `90000002` 的前两张现有背景图，均只有 2 个满足条件的孤立 PSF 样本，正确触发圆高斯回退；
   因此不能宣称实际数据已经用上经验 PSF，后续应根据诊断检查样本质量/数量。
4. **局部噪声和质量降权**：自动排除明显正负结构后分块估计噪声；少样本网格明确回退。
   PSF/透过率回退与配准误差引入保守方差膨胀，属于启发式质量惩罚，不是完整系统误差模型。
   检测图使用局部中心和尺度，尺度不低于名义模型的 1；输出局部尺度、中心和相邻像元相关性。
   **未开启噪声白化**：未证实结构来源及传播模型前，不把真实天体结构当作噪声消除。
5. **残差再检测**：全图匹配滤波 → 自动候选分组 → 原始有效像元联合拟合 → 残差再次盲检，最多 3 轮。
   原曝光上共同拟合源通量并消去各曝光局部常数背景；候选位置允许有限亚像素优化，不读取星表。
   后续轮次冻结第一轮的噪声标定，避免扣除亮源后不断缩小 RMS、放大假峰。
   最多 1000 个候选；`max_group=12` 现在是稠密/稀疏拟合的切换点，不再是整组拒绝上限。
   大组使用稀疏 PSF 模板联合求解所有源通量，并在互不重叠的 64 像素背景格内消去常数背景。
   局部窗口仅用于提出位置调整，最终通量和误差来自原始有效像元的整组解（包含邻源协方差）；
   位置调整只有改善同一固定区域的全组目标才接受。退化和非正通量仍明确标记，阈值及覆盖要求不变。
   详见 [拥挤源组修复记录](crowded_fit_20260826.md)，Notebook 第 7 节保留修复前历史结果，第 8 节展示复核。

## 不覆盖旧结果的运行方式

以下为**以后手动启动**的示例，不是本次已执行的任务。路径已存在时新实验会拒绝覆盖。

```powershell
# 先在 validation 帧上消融；只读取 split/filename 等列，不读取历史坐标或 SNR。
python scripts/evaluation/run_stage_recovery.py --split-manifest data/processed/asteris_paper_400/manifests/split_manifest.csv --sequence 90000002 --phase validation --limit 8 --sources 4 --repeats 1 --fluxes 6000 12000 --ablation --output-root data/processed/weak_source_v3_trial01/recovery_validation

# 冻结背景参数后，另行运行两轮背景。这里的 64/5 只是保留的起点，不代表已选优。
python scripts/run_background.py --two-pass --sequences 90000002 --box-size 64 --filter-size 5 --split-manifest data/processed/asteris_paper_400/manifests/split_manifest.csv --output-root data/processed/weak_source_v3_trial01/background

# 在新产品上启动 V3 检测；旧 blind_joint 不覆盖。
python scripts/evaluation/run_blind_joint_detection.py --method v3 --input-root data/processed/weak_source_v3_trial01/background --output-root data/processed/weak_source_v3_trial01/joint --sequences 90000002
```

`run_stage_recovery.py` 默认跳过 ASTERIS，不加载 torch/权重。只有显式 `--checkpoint` 才做 `eval()` / `inference_mode()` 推理，且要求至少 8 帧。
该选项绝不会调用训练函数；旧检查点和新预处理是否兼容不能由此自动证明，会写入报告警告。
显式提供检查点时另外保留 `paper_input_coadd` 对照，它与 ASTERIS 输出使用相同单图检测器，隔离网络处理的影响。
模型比较必须固定同一测试曝光列表、输入版本和曝光数，不能把 80 帧与 16 帧直接比较。
正式测试使用新的输出目录和独立随机种子；禁止根据 test 结果或真实星表位置反复挑参数。

## 如何读评估输出

- `source_recovery.csv`：每个模拟源的恢复、已有源重合标记、定位误差、测得通量、配对通量响应。
- `stage_summary.csv`：按阶段/通量/阈值/重复统计。已有源重合者从“新增恢复率”分母剔除，但保留明细。
- `report.json`：输入哈希、冻结曝光、配准、参数、训练/星表未使用声明及限制。
- V3 `all_candidates.csv` 保留失败候选、`group_size`、`fit_method`；`blind_sources.csv` 只保存通过拟合和经验阈值的候选。
  `position_sweeps_accepted` 计数仅对稀疏大组路径有意义，稠密路径不使用该计数。
- V3 FITS 包含 DQ/coverage/information；局部尺度和中心单独输出。插值展示图绝不进入检测。

`paired_flux_response` 是处理前后差值对注入通量的响应，用于定位信号在哪一步削弱；**不是检出证据**。
真实星场上的 `new_unmatched_candidates` 可能包含真实但原先漏检的天体，**不能直接当作假源数/纯度**。
负峰、奇偶一致性和中心 MAD 也不是已校准的误报率。

固定误报预算比较需要独立的无源验证模拟先走完同一条处理链，然后使用
`select_threshold_from_null(null_peak_scores, thresholds, max_false_per_image)` 冻结峰值检测阈值。
这不是按全部像元数计算的“5σ”；样本不足时不能给出可靠尾部概率。
真实数据完整消融、独立无源验证和测试集恢复率尚待后续运行，当前不宣称真实弱源检出已提升。

真实 2MASS/Gaia 星表保持只读事后可视化用途，不参与源保护、背景、检测、训练和参数选择。
