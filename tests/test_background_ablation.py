import inspect
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from astr_ir.evaluation.background_ablation import (paper_input, products, null_raw,
                                                   score_trial, accepted)
from astr_ir.asteris.paper_pipeline import (PaperAsterisConfig, denoise_registered_exposures,
                                           prepare_paper_dataset)
from astr_ir.evaluation.weak_detection import Exposure, analytic_psf


def test_ablation_input_is_exact_paper_control():
    rng = np.random.default_rng(17)
    stack = rng.normal(0, 3, (8, 32, 32)).astype(np.float32)
    stack[:, 13, 14] += 300
    valid = np.ones_like(stack, bool)
    valid[:, 0, 0] = False
    image, mask = paper_input(stack, valid)
    reference, _, expected, _ = denoise_registered_exposures(stack, valid, nn.Identity(),
        replace(PaperAsterisConfig(), inference_tile_size=32, inference_overlap=0, amp=False), device='cpu')
    assert np.array_equal(image, reference, equal_nan=True)
    assert np.array_equal(mask, expected)


def test_processing_signature_has_no_truth_or_catalog():
    assert set(inspect.signature(products).parameters) == {'raw', 'valid', 'offsets', 'methods', 'models'}


def test_null_is_reproducible_does_not_recycle_stars_or_change_inputs():
    rng = np.random.default_rng(1)
    raw = rng.normal(10000, 100, (8, 64, 64)).astype(np.float32)
    raw[:, 30, 31] += 1e7
    valid = np.ones_like(raw, bool)
    valid[:, 10, 10] = False
    before = raw.copy()
    first = null_raw(raw, valid, np.random.default_rng(42))
    second = null_raw(raw, valid, np.random.default_rng(42))
    assert np.array_equal(first, second, equal_nan=True)
    assert np.isnan(first[:, 10, 10]).all()
    assert np.nanmax(first) < 20000
    assert np.array_equal(raw, before)


def test_frozen_manifest_validation_prevents_unsafe_or_missing_frames(tmp_path):
    manifest = pd.DataFrame(dict(frame_id=['s:0'], sequence=['s'], frame_index=[0],
        split=['train'], filename=['a.fits'], product_path=['../outside.fits'],
        alignment_dx=[0.], alignment_dy=[0.]))
    with pytest.raises(ValueError, match='unsafe'):
        prepare_paper_dataset(tmp_path/'input', tmp_path/'raw', tmp_path/'output',
                              sequences=['s'], frozen_manifest=manifest)
    manifest.loc[0, 'product_path'] = 'missing.fits'
    with pytest.raises(ValueError, match='Missing'):
        prepare_paper_dataset(tmp_path/'input', tmp_path/'raw', tmp_path/'output',
                              sequences=['s'], frozen_manifest=manifest)


def test_parallel_frame_order_and_values_match_serial(monkeypatch):
    import astr_ir.evaluation.background_ablation as module
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(29)
    raw = rng.normal(20000, 20, (3, 144, 144)).astype(np.float32)
    valid = np.ones_like(raw, bool)
    valid[:, 60, 60] = False
    monkeypatch.setenv('ASTR_IR_FRAME_WORKERS', '1')
    serial = module.flicker_stack(raw, valid)
    serial_background = module.background_stack(serial, valid, np.zeros((3,2)), 'new32_single')
    with ProcessPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(module, '_FRAME_POOL', pool)
        monkeypatch.setenv('ASTR_IR_FRAME_WORKERS', '2')
        parallel = module.flicker_stack(raw, valid)
        parallel_background = module.background_stack(parallel, valid, np.zeros((3,2)), 'new32_single')
    assert np.array_equal(serial, parallel, equal_nan=True)
    assert np.array_equal(serial_background, parallel_background, equal_nan=True)
