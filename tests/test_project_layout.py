"""Guard the notebook-first layout without running data processing or training."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))


def test_only_four_main_script_entry_points():
    assert {p.name for p in (ROOT / "scripts").glob("*.py")} == {
        "run_flicker.py", "run_background.py", "run_noise2noise.py", "run_asteris.py"
    }
    assert [p.name for p in (ROOT / "notebooks/asteris").glob("*.ipynb")] == [
        "01_asteris8_paper_160_vs_400.ipynb"
    ]
    source = (ROOT / "scripts/run_asteris.py").read_text(encoding="utf-8")
    assert "from astr_ir.asteris.paper_pipeline import" in source
    assert "from astr_ir.asteris.processor import" not in source


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_script_project_roots_and_references(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    # Only evaluate the literal Path(__file__) root expression, never the script.
    assignments = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in {"ROOT", "PROJECT_ROOT"} for t in node.targets)
    ]
    assert len(assignments) == 1
    expression = ast.Expression(assignments[0].value)
    actual = eval(compile(expression, str(path), "eval"), {"Path": Path, "__file__": str(path)})
    assert actual == ROOT
    for target in re.findall(r"scripts/[\w/]+\.py", source):
        assert (ROOT / target).is_file(), (path, target)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_syntax_and_script_references(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path.name}:cell{index}")
        for target in re.findall(r"scripts/[\w/]+\.py", cell.source):
            assert (ROOT / target).is_file(), (path, target)


def test_local_markdown_links():
    paths = [ROOT / "README.md", ROOT / "data/README.md", ROOT / "scripts/README.md"]
    paths += list((ROOT / "docs").rglob("*.md"))
    for path in paths:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            assert (path.parent / target).exists(), (path, target)


def test_asteris_builder_and_notebook_root_discovery_match():
    notebook = json.loads((ROOT / "notebooks/asteris/01_asteris8_paper_160_vs_400.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in notebook["cells"] if "PROJECT_ROOT = next" in "".join(c["source"]))
    line = next(line for line in cell.splitlines() if line.startswith("PROJECT_ROOT = next"))
    builder = (ROOT / "scripts/notebooks/build_asteris_paper_notebook.py").read_text(encoding="utf-8")
    assert line in builder
    expression = ast.parse(line).body[0].value
    for start in (ROOT, ROOT / "notebooks", ROOT / "notebooks/asteris"):
        assert eval(compile(ast.Expression(expression), "root", "eval"), {"start": start}) == ROOT


def test_catalog_visualization_cells_match_builder():
    builder = ast.parse((ROOT/'scripts/notebooks/build_blind_pipeline_notebook.py').read_text(encoding='utf-8'))
    templates = [ast.literal_eval(node.value.args[0]).strip() for node in builder.body
                 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                 and isinstance(node.value.func,ast.Name) and node.value.func.id in {'md','code'}]
    notebook = nbformat.read(ROOT/'notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb',as_version=4)
    cells = [c for c in notebook.cells if c.id.startswith('catalog-validation-')]
    assert len(cells) == 3
    assert all(c.source.strip() in templates for c in cells)


def test_weak_source_cells_match_builder_and_never_run_by_default():
    builder=ast.parse((ROOT/'scripts/notebooks/build_blind_pipeline_notebook.py').read_text(encoding='utf-8'))
    templates=[ast.literal_eval(n.value.args[0]).strip() for n in builder.body
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
        and n.value.func.id in {'md','code'}]
    notebook=nbformat.read(ROOT/'notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb',as_version=4)
    cells=[c for c in notebook.cells if c.id.startswith('weak-source-v3-')]
    assert len(cells)==2 and all(c.source.strip() in templates for c in cells)
    source=next(c.source for c in cells if c.cell_type=='code')
    switches=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id.startswith('RUN_') for t in n.targets)]
    assert len(switches)==3 and all(ast.literal_eval(n.value) is False for n in switches)
