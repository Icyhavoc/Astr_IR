"""Shared, display-only catalog cells for the four main pipeline notebooks."""

from __future__ import annotations

import textwrap

import nbformat as nbf


def build_cells(stage: str, archived_fits: str, note: str):
    png = f"../../figures/starmap_reference/{stage}_90000003.png"
    archive_path = "_cleanup_backup/interface_optimization_20261009/processed/" + archived_fits
    markdown = f"""## 2MASS 真实星表位置参照（90000003，一张示例）

下图是归档科学 FITS 的**处理后图像**，叠加 [2MASS 点源星表](https://irsa.ipac.caltech.edu/data/2MASS/docs/releases/allsky/doc/sec1_6b.html)预测位置。青色方框是三颗配准星；橙色圆圈是未参与拟合的 K<12 星表位置；绿色圆圈是固定的较暗星表位置。圆圈**不代表检出**，单帧中多数非配准星没有显著响应。FITS 没有完整 WCS，所以本图对每张科学图独立求三锚星旋转和平移；锚星残差不能当作全视场精度。显示灰度按本图百分位拉伸，不用于跨阶段测光比较。

{note} 星表只在输出后显示，不进入掩膜、背景估计、训练或候选筛选。下方单元只读取一张归档 FITS；要看新运行结果，请将 `STARMAP_FITS` 显式改成对应科学产品路径。

![{stage} 的 90000003 星表叠加图]({png})"""
    code = f"""from astr_ir.evaluation.stage_starmap import plot_stage_starmap
import matplotlib.pyplot as plt

STARMAP_FITS = PROJECT_ROOT.parent / {archive_path!r}
STARMAP_CATALOG = PROJECT_ROOT / "data/reference/2mass_90000003.csv"
if STARMAP_FITS.is_file():
    fig, starmap_solution = plot_stage_starmap(
        STARMAP_FITS, STARMAP_CATALOG, stage={stage!r}
    )
    display(fig)
    plt.close(fig)
    print("3-anchor RMS (px):", round(starmap_solution["anchor_rms_pix"], 3))
else:
    print("归档 FITS 不在本机；上方仓库图片仍可查看。若有新结果，请设置 STARMAP_FITS。")"""
    return [
        nbf.v4.new_markdown_cell(markdown),
        nbf.v4.new_code_cell(textwrap.dedent(code)),
    ]
