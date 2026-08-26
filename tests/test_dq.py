from pathlib import Path

from astropy.io import fits
import numpy as np

from astr_ir.dq import (
    DQ_DETECTOR_BAD,
    DQ_DO_NOT_USE,
    DQ_NO_COVERAGE,
    DQ_PARTIAL_COVERAGE,
    build_dq,
    read_dq,
    write_fits_with_dq,
)


def test_dq_distinguishes_detector_bad_missing_and_partial_coverage(tmp_path: Path):
    detector = np.zeros((4, 5), dtype=bool)
    missing = np.zeros_like(detector)
    partial = np.zeros_like(detector)
    detector[1, 2] = True
    missing[2, 3] = True
    partial[3, 4] = True
    dq = build_dq(
        detector.shape,
        detector_bad=detector,
        no_coverage=missing,
        partial_coverage=partial,
    )
    assert dq[1, 2] == DQ_DO_NOT_USE | DQ_DETECTOR_BAD
    assert dq[2, 3] == DQ_DO_NOT_USE | DQ_NO_COVERAGE
    assert dq[3, 4] == DQ_PARTIAL_COVERAGE
    assert dq[0, 0] == 0

    science = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / "science.fits"
    write_fits_with_dq(path, science, fits.Header(), dq)
    assert np.array_equal(fits.getdata(path), science)
    assert np.array_equal(read_dq(path, science.shape), dq)
    with fits.open(path) as hdul:
        assert hdul["DQ"].header["DQBIT0"] == "DO_NOT_USE"
        assert hdul["DQ"].header["DQBIT3"] == "PARTIAL"
