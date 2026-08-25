# Astr_IR 红外图像处理项目

本项目包含一条可复现的红外 FITS 处理链：先校正逐行/列相关的 1/f 条纹噪声，再扣除二维背景，最后可并列使用二维 Noise2Noise 基线或三维 ASTERIS 时空 Transformer 进行自监督去噪和弱源检测评估。原始数据保持只读，所有派生产品统一写入 `data/processed/`。

## 处理链

```text
data/raw/our_dataset/{90000002,90000003,90000004,90000005_1,90000005_2}/*.fits
                │
                ▼
      astr_ir.flicker.processor
                │
                ▼
data/processed/flicker/flicker_corrected_*.fits
                │
                ▼
     astr_ir.background.processor
                │
                ▼
data/processed/background/background_subtracted_*.fits
                │
          ┌─────┴─────┐
          ▼           ▼
 Noise2Noise 2D    ASTERIS 3D
          │           │
          ▼           ▼
noise2noise_denoised  asteris_denoised
```

`flicker_model_*.fits` 只用于记录 1/f 模型，不会进入背景扣除；`processed_fits/Fixed_*` 也不作为任何流程输入。

## 目录

```text
data/raw/our_dataset/       原始 FITS、blindmap、测量表和原始说明文档
data/processed/flicker/     1/f 校正产品与统计表（自动生成）
data/processed/background/  背景扣除产品与统计表（自动生成）
data/processed/noise2noise/  自监督模型、清单和去噪产品（自动生成）
data/processed/asteris/      ASTERIS 清单、checkpoint、时空去噪产品（自动生成）
data/processed/asteris_paper_{160,400}/  论文发布版 ASTERIS8 对比实验
data/processed/evaluation/   与模型解耦的伪源注入、盲检和科学评估（自动生成）
src/astr_ir/                可安装的核心 Python 包
scripts/                    批处理、Notebook 和任务文档构建入口
notebooks/                  已执行的分析与验收 Notebook
tests/                      各处理阶段、模型和通用科学评估的自动测试
docs/                       算法、架构与任务说明
figures/                    Notebook 生成的诊断图
paper/、ppt/                现有论文与汇报材料（保留原位）
```

更详细的职责边界见 [docs/architecture.md](docs/architecture.md)，算法说明见 [docs/flicker.md](docs/flicker.md)、[docs/background.md](docs/background.md)、[docs/noise2noise.md](docs/noise2noise.md)、[docs/asteris.md](docs/asteris.md) 与 [docs/source_evaluation.md](docs/source_evaluation.md)。

## 安装

建议使用 Python 3.11 或 3.12，并在项目根目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

也可按传统方式安装：

```powershell
python -m pip install -r requirements.txt
```

## 快速运行

```powershell
# 先运行 1/f 条纹校正
python scripts/run_flicker.py --overwrite

# 再运行二维背景扣除
python scripts/run_background.py --overwrite

# 构建清单、训练并完成全量推理
python scripts/run_noise2noise.py --stage all --overwrite

# ASTERIS 主代码在 notebook；CLI 可执行同一流程（先做 GPU smoke test）
python scripts/run_asteris.py --stage prepare
python scripts/run_asteris.py --stage train --epochs 1 --train-samples-per-epoch 2 --validation-samples 2 --f-maps 4 --device cuda
python scripts/run_asteris.py --stage calibrate --model asteris4 --device cuda
python scripts/run_asteris.py --stage all --model asteris4 --device cuda

# 论文发布版 ASTERIS8：160/400 帧训练、共同测试推理和配对评估
python scripts/run_asteris_paper.py --profile 160 --stage all --device cuda --overwrite
python scripts/run_asteris_paper.py --profile 400 --stage all --device cuda --overwrite
python scripts/run_asteris_paper_evaluation.py --profile 160 --device cuda
python scripts/run_asteris_paper_evaluation.py --profile 400 --device cuda
python scripts/compare_asteris_paper_profiles.py

# 用原始 RA/DE 指向完成 WCS，查询 2MASS/Gaia，并标注真实弱源
python scripts/annotate_asteris_paper_catalog.py

# 独立运行通用伪源盲检评估和绘图
python scripts/run_source_evaluation.py --model noise2noise --device cuda
python scripts/run_source_evaluation.py --model asteris --device cuda --model-output-root data/processed/asteris --evaluation-root data/processed/evaluation/asteris
python scripts/plot_source_evaluation.py

# 运行全部自动测试
python -m pytest -q

# 按当前原始帧数动态校验全部产品、目录清单、科学公式和局部质量门
python scripts/validate_products.py

# 严格校验 Noise2Noise 的 320 个 FITS、数据隔离和测试集科学门
python scripts/validate_noise2noise.py

# 严格校验伪源评估的数据隔离、一一匹配和指标
python scripts/validate_source_evaluation.py
```

前两阶段快速冒烟检查可加 `--limit-per-sequence 1`。Noise2Noise 与 ASTERIS 都应先做小批量 GPU 冒烟训练。各流程的主 Notebook 位于 `notebooks/flicker/`、`notebooks/background/`、`notebooks/noise2noise/`、`notebooks/asteris/` 和 `notebooks/evaluation/`；重建脚本位于 `scripts/build_*_notebook.py`。

## 数据与 GitHub

`.gitignore` 默认忽略原始/派生数据、缓存和生成图，避免把数百个 FITS 二进制文件提交到普通 Git 历史。代码、测试、Notebook、统计方法和文档应提交；大数据建议使用 Git LFS、DVC 或 Zenodo/机构存储，并在发布版本中记录数据校验和与下载地址。

推荐首次提交前执行：

```powershell
git init
git status --short
git add README.md pyproject.toml requirements.txt .gitignore src scripts tests notebooks docs
git status --short
```

确认 `git status` 中没有原始 FITS、派生 FITS、临时 Office 文件或缓存后再提交。
