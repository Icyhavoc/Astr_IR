# 数据目录

正式流程：`raw → flicker → background → Noise2Noise / ASTERIS8`。两个模型分支都只读取背景扣除后的科学图。

- `raw/our_dataset/`：原始 FITS、盲元图及随附资料，只读。
- `processed/`：当前为空。下次实验按阶段重新生成，不能把旧权重与新背景或新 manifest 混用。

2026-10-09 将原 `processed/` 的 2,712 个文件（19,774,967,169 字节）整体移至
`D:/Astr_IR/_cleanup_backup/interface_optimization_20261009/processed/`。
同目录的 `archive_manifest.json` 记录数量和体量。归档位于同一磁盘，清理的是工作目录视野，未释放磁盘空间；可按原相对路径恢复。未改动原始数据或归档内容。

重跑时使用新的输出目录，先验证 1/f 与背景阶段的全盲注入和通量保持，再决定是否训练模型；尤其不要直接启动耗时的 ASTERIS 全量训练。历史图表和 Notebook 输出仅作追溯，不能作为当前输入版本的科学结论。
