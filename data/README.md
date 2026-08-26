# 数据目录

主流程为 `raw → flicker → background → noise2noise / asteris_paper`。
以下路径相对 `code/data/`；当前科学数据保留原位，不与归档实验混放。

## 当前保留

| 目录 | 内容 |
|---|---|
| `raw/our_dataset/` | 5×80=400 张原始曝光、盲元图及原始测量/说明材料 |
| `processed/flicker/` | 400 张科学图 + 400 张条纹模型，及统计表 |
| `processed/background/` | 400 张科学图 + 400 张背景模型，及统计表 |
| `processed/noise2noise/` | N2N 历史 best/last 权重、manifest、320 张科学/残差图 |
| `processed/asteris_paper_{160,400}/` | 论文版 best/last 权重、manifest、42 个 stack 数组、12 张共加/残差 FITS |
| `processed/comparison_before_blind_v2/` | 每序列每阶段两张旧图，共 20 张，以及旧统计 |
| `processed/pre_asteris_blind_v2/` | 全盲批次配置、续跑标志、进度表、校验及同帧对比 |
| `processed/blind_joint/` | 25 张联合检测 FITS，以及检测目录与质量诊断 |
| `processed/evaluation/` | N2N、论文版160/400评估、配对比较和星表人工核验材料 |

模型结果仍对应旧预处理。不要与新 background/配准清单混用。
`paper_stacks` 被论文版 manifest 和训练代码引用，不能因为它是缓存就删除；
旧输入已更新，直接重新 prepare 会改变实验数据。进度文件用于续跑/审计，也予以保留。

## 2026-08-26 第二轮清理

从 code 移出 831 个文件，共 3,990,681,520 字节（约 3.72 GiB）：

- 旧版 ASTERIS4 完整实验（含 smoke test、权重、逐帧产品）及其独立评估。
- raw 中两组共 160 张 `processed_fits/Fixed_*` 旧派生图；400 张原始曝光不动。
- MATLAB 图形状态缓存。
- N2N 模型根目录两份已被独立评估取代的早期表格。
- 19 张过时诊断图、两份过时任务书及其索引；另移出一个空的旧图像目录。

全部按原相对路径归档到 `D:/Astr_IR/_cleanup_backup/assets_20260826/`。
`archive_manifest.json` 逐文件记录 SHA-256、大小、原因；`protected_manifest.json` 记录原始曝光、盲元图及当前权重的校验和。
归档可恢复，**仅减少 code 的文件和占用，没有释放同一磁盘的总空间**。
旧版三份阶段文档另存于归档的 `documentation_before/`。

不改写科学像元，不重跑预处理，不启动训练。科学数据默认不进入普通 Git。

验收：103 项测试通过；400 张原始曝光、1,600 张预处理 FITS、320 张 N2N FITS、12 张论文版 FITS、42 个 stack 数组、20 张旧图和 25 张联合检测 FITS 的保留清单与引用均完整。
原始曝光、盲元图及当前权重/训练记录共 412 个文件的 SHA-256 未变；保留的数据和结果图未改写。
