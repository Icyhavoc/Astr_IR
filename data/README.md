# Data layout

- `raw/our_dataset/`：不可修改的原始输入，包括两组原始 FITS、盲元图、逐帧测量表与原始任务资料。
- `processed/flicker/`：由 `scripts/run_flicker.py` 生成。
- `processed/background/`：由 `scripts/run_background.py` 生成，只读取前一阶段的 `flicker_corrected_*.fits`。

这些目录中的科学数据默认不进入普通 Git 历史。共享数据时应使用 Git LFS、DVC 或独立数据仓库，并记录版本、校验和与生成命令。
