# 脚本入口与整理记录

所有命令在 `code/` 下运行。主实验优先使用 [README 中的四个 Notebook](../README.md)。

## 主流程

`scripts/` 顶层只保留：

```text
run_flicker.py → run_background.py → run_noise2noise.py
                                  或 run_asteris.py（论文版 ASTERIS8）
```

原论文版 `run_asteris_paper.py` 已统一命名为 `run_asteris.py`；必须指定 `--profile 160` 或 `400`。
旧 CLI 的 `--model asteris4`、`--stage calibrate/evaluate` 不再适用。
阶段产品和模型目录没有因此改名，旧结果仍可追溯。

## 可选工具

| 目录 | 内容 | 使用边界 |
|---|---|---|
| `evaluation/` | 盲联合检测、伪源评估、160/400 比较、绘图、星表标注 | 非主流程必经阶段；星表仅事后核验 |
| `validation/` | 产品公式/DQ、N2N、伪源和全盲流程校验 | 按数据版本选择，不把历史指标套到新结果 |
| `notebooks/` | 六个现存 Notebook 的构建器 | 重建会清空对应 Notebook 历史输出 |
| `maintenance/` | `run_pre_asteris.py` | 上次带旧图备份的预处理批次；不是模型训练入口 |

常用辅助命令（需要相应数据/模型已准备好）：

```powershell
python scripts/evaluation/run_asteris_paper_evaluation.py --profile 160 --device cuda
python scripts/evaluation/run_asteris_paper_evaluation.py --profile 400 --device cuda
python scripts/evaluation/compare_asteris_paper_profiles.py
python scripts/evaluation/annotate_asteris_paper_catalog.py
python scripts/evaluation/run_blind_joint_detection.py
python scripts/evaluation/run_source_evaluation.py --model noise2noise --device cuda
python scripts/evaluation/plot_source_evaluation.py
python scripts/validation/validate_products.py
python scripts/validation/validate_blind_pipeline.py
python scripts/validation/validate_noise2noise.py
python scripts/validation/validate_source_evaluation.py
```

这些辅助命令可能写入评估、图像或校验报告，不是本次整理实际执行的清单。
预处理续跑和旧图保留规则见 [全盲流程记录](../docs/evaluation/blind_pipeline.md)。
`maintenance/run_pre_asteris.py --resume` 仅恢复该已记录批次；已有完成标志时不会重新计算。

## 2026-08-26 第一轮清理（入口代码）

移除六份不再需要的旧文件内容：

- 旧 `run_asteris.py`：已用论文版入口替换。
- `01_asteris_self_supervised.ipynb` 及 `build_asteris_notebook.py`：旧 ASTERIS4/α 混合入口，避免与论文版混淆。
- `resummarize_asteris_paper_evaluation.py`：一次性 trial ID 修补；160/400 已保存表均已修正，正式评估代码已直接写入正确 ID。
- `build_background_task_document.py`：旧 160 帧、已知目标保护任务书生成器；历史文档保留，生成器不再对应现流程。
- `src/astr_ir/asteris/visualization.py`：无调用者的早期绘图辅助模块。

相应移除仅由任务书生成器直接使用的开发依赖 `python-docx`、`lxml`；没有卸载环境中的软件包。
清除 Python/pytest 缓存；统一忽略 `figures/*_output/` 的新增产物，已跟踪的历史诊断图保持原样。
历史任务 DOCX 和参考 PNG 移至 `docs/references/`；不是删除科学材料。

被删旧文件及旧版 ASTERIS 文档已按相对路径备份至项目上一级
`../_cleanup_backup/code_20260826/`（不是系统回收站）。不要整目录覆盖恢复：旧 CLI 名称现已由论文版使用。
原始数据、权重、FITS、评估表、20 张旧图对比样本及保留 Notebook 的历史输出均不删除。

共享的 ASTERIS dataset/preprocessing，以及历史评估仍引用的 processor/inference 保留。
目录整理没有改变预处理或模型算法，也没有启动训练。

验收：103 项测试通过，12 个 CLI 的 `--help` 检查通过，`git diff --check` 通过。
6 个保留 Notebook 的 outputs、execution_count 和 cell ID 校验一致；3,385 个数据/图像文件的路径、大小和修改时间未变。
仅更新 `data/README.md` 的目录说明。共清除 9 个缓存目录、69 个可自动再生的缓存文件（870,248 字节）。

## 第二轮：数据、图像和文档

在第一轮之后，用户进一步要求清理 data/figures/docs；旧实验和过时材料已移出 code。
第一轮“数据不动”的验收描述仅对应第一轮，当前保留项、移出项与恢复位置见 [数据清理记录](../data/README.md)。
当前图像版本见 [figures/README.md](../figures/README.md)。辅助脚本不读取已归档的旧图或早期 N2N 评估表。
若追溯旧 ASTERIS4 的通用评估，必须显式提供归档的 `--model-output-root` 及单独 `--evaluation-root`；
不能使用当前 background 冒充其旧输入。论文版评估继续使用 `evaluation/run_asteris_paper_evaluation.py`。
