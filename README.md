# Astr_IR 红外图像处理项目

本项目包含一条可复现的二维红外 FITS 处理链：先校正逐行/列相关的 1/f 条纹噪声，再对条纹校正后的科学图估计并扣除二维背景。原始数据保持只读，所有派生产品统一写入 `data/processed/`。

## 处理链

```text
data/raw/our_dataset/9000000{2,3}/*.fits
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
```

`flicker_model_*.fits` 只用于记录 1/f 模型，不会进入背景扣除；`processed_fits/Fixed_*` 也不作为任何流程输入。

## 目录

```text
data/raw/our_dataset/       原始 FITS、blindmap、测量表和原始说明文档
data/processed/flicker/     1/f 校正产品与统计表（自动生成）
data/processed/background/  背景扣除产品与统计表（自动生成）
src/astr_ir/                可安装的核心 Python 包
scripts/                    批处理、Notebook 和任务文档构建入口
notebooks/                  已执行的分析与验收 Notebook
tests/                      两个流程的自动测试
docs/                       算法、架构与任务说明
figures/                    Notebook 生成的诊断图
paper/、ppt/                现有论文与汇报材料（保留原位）
```

更详细的职责边界见 [docs/architecture.md](docs/architecture.md)，算法说明见 [docs/flicker.md](docs/flicker.md) 与 [docs/background.md](docs/background.md)。

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

# 运行全部自动测试
python -m pytest -q

# 严格校验 640 个产品、目录清单、科学公式和局部质量门
python scripts/validate_products.py
```

快速冒烟检查可加 `--limit-per-sequence 1`。Notebook 位于 `notebooks/flicker/` 和 `notebooks/background/`；重建脚本位于 `scripts/build_*_notebook.py`。

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
