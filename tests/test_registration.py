import numpy as np
from astr_ir.registration import masked_shift, masked_gaussian


def test_masked_value_cannot_leak_into_subpixel_neighbours():
    image = np.ones((24, 24)) * 7
    valid = np.ones_like(image, bool)
    valid[12, 12] = False
    image[12, 12] = 1e30
    shifted, good, support, _ = masked_shift(image, valid, (0.3, -0.4))
    assert np.allclose(shifted[good], 7)
    assert np.any((support > 0.5) & (support < 1))
    assert np.allclose(masked_gaussian(image, valid, 2), 7)


def test_variance_uses_squared_weights_and_no_coverage_is_nan():
    image = np.ones((12, 12))
    shifted, valid, support, var = masked_shift(image, image.astype(bool), (0.5, 0.5), variance=4)
    assert np.allclose(var[valid], 1)  # Four independent quarter-weight samples.
    assert np.isnan(shifted[~valid]).all()
    assert not valid[0].any()


def test_identity_preserves_holes_without_inventing_coverage():
    image = np.arange(100).reshape(10, 10).astype(float)
    mask = np.ones_like(image, bool)
    mask[2:4, 2:4] = False
    shifted, valid, _, var = masked_shift(image, mask, (0, 0), variance=3)
    assert np.array_equal(valid, mask)
    assert np.array_equal(shifted[valid], image[valid])
    assert np.all(var[valid] == 3)
