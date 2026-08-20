from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.validate_products import equation_max_error, validate_inventory


def test_equation_check_rejects_finite_mask_mismatch():
    actual = np.array([[1.0, np.nan]], dtype=np.float32)
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="finite-pixel mask mismatch"):
        equation_max_error(actual, expected, "synthetic")


def test_inventory_check_rejects_unrecorded_fits(tmp_path: Path):
    expected_path = tmp_path / "expected.fits"
    extra_path = tmp_path / "extra.fits"
    expected_path.touch()
    extra_path.touch()
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        validate_inventory(tmp_path, {expected_path.name}, "Synthetic")
