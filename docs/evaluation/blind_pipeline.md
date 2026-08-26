# Blind pre-ASTERIS pipeline, 2026-08-26

Main notebook: `notebooks/evaluation/02_blind_pre_asteris_pipeline.ipynb`.

Section 5 now provides an independent, offline catalog-position visualization of the frozen products.
It reads the previously saved 12 weak-source labels for each of 90000002 and 90000003 only after processing.
Automatic image stars transfer the historical catalog grid into the current first-exposure grid; weak-source positions are never snapped to peaks or reselected.
Outputs are PNG/CSV/JSON in `figures/catalog_validation_output/`, with separate exposure/preprocessing labels and registration diagnostics.
This display-only module does not change any science data or the catalog-free processing described below.

## Results and limitations of this run

All 400 frames completed both preprocessing stages. The 1,600 preprocessor FITS and 25 joint-detection FITS passed inventory/DQ checks; both float32 subtraction equations have zero error. There are 20 retained old science FITS (two per sequence per stage). The executed main notebook contains no cell errors. No training was launched.

**A background-flatness regression was measured:** across the ten retained background science frames, the median 64-pixel block-location scatter increased from approximately **12 DN to 29 DN** with the more conservative 64-pixel mesh (using the same detector/edge masks for old and new). This is not evidence of improved weak-source preservation. The defaults are an experimental conservative baseline, not a validated performance improvement; a 32/64-mesh source-injection and background-scale ablation is needed before training on it. Exact paired values and paths are in `data/processed/pre_asteris_blind_v2/old_new_comparison.csv`.

All five joint maps retain shared-structure warnings (empirical noise scales about 3.4–4.3). In `90000005_2`, 69/80 frames use the fallback PSF and all 80 use default transparency. Positive detections, half-split agreement and negative peaks must not be equated to true-source completeness. Full diagnostics are in `data/processed/pre_asteris_blind_v2/validation.json`.

- Science processing never reads the catalog or target measurement CSV. Legacy `target` arguments in 1/f and background APIs are ignored. Automatic image-derived source masks remain enabled.
- Blind pixels and input DO_NOT_USE flags are excluded before Gaussian filtering, registration and coaddition. Bilinear registration divides by valid kernel support (minimum 0.5), returns coverage and propagates diagonal variance with squared weights. It does not claim uncorrelated output pixels.
- Background mesh defaults to 64 pixels (previously 32), with a five-cell median filter. Multi-scale source masks are automatic. Bright detection now excludes bad pixels before mask dilation. Noise and large-scale-improvement gates remain; known-target photometry gates are disabled.
- Future N2N/ASTERIS manifest preparation uses automatic stellar features relative to the first exposure, not CSV target tracks. Frame split labels are unchanged. Known-source patch preference and known-source clipping protection are not used by newly prepared blind manifests. Existing checkpoints/manifests/coadds are not regenerated in this run.
- Future empirical-PSF evaluation selects stars automatically from the first twelve training frames per sequence; it no longer selects frames or source masks from catalog SNR/target tracks. This keeps the downstream evaluation compatible with the new catalog-free manifests.

## Commands

```powershell
python scripts/maintenance/run_pre_asteris.py --workers 2
# Only to recover an interrupted run with its backup still intact:
python scripts/maintenance/run_pre_asteris.py --workers 2 --resume
python scripts/evaluation/run_blind_joint_detection.py
python scripts/validation/validate_blind_pipeline.py
python -m pytest -q
```

First/last OLD science frames per sequence and per stage are copied to
`data/processed/comparison_before_blind_v2/{flicker,background}/{sequence}/old_*.fits`.
Other previous preprocessor products are overwritten by the recomputation, not archived.
The originals in `data/raw` and all learned-model outputs are untouched.
Progress/config are in `data/processed/pre_asteris_blind_v2`.

## Joint detection baseline

Reference: Zackay & Ofek, *How to coadd images? I. Optimal source detection and photometry using ensembles of images*, [arXiv:1512.06872](https://arxiv.org/abs/1512.06872), equations 27 and 29. Companion proper-coaddition paper: [arXiv:1512.06879](https://arxiv.org/abs/1512.06879). Original arXiv PDFs are in `D:/Astr_IR/paper`.

For each frame, fit circular Gaussian PSFs to automatically detected stars, estimate native-pixel background noise, and obtain a robust relative transparency from uniquely matched stars. The fallback PSF sigma is 2.5 pixels and transparency 1 only when too few fits/matches are available; every fallback is reported. Gross registration failures raise an error instead of silently assigning zero shift.

Detection evaluates `N=sum(F*P*I/V)` and `D=sum(F²*P²/V)` on native pixels with DQ support. Subpixel translation goes into the PSF kernel; only an integer translation is applied to the filtered statistic. Thus bilinear covariance is not introduced in this detection path. The visual weighted coadd separately uses mask-normalized bilinear interpolation and its diagonal variance. Neither output is Paper II's Fourier proper coadd.

The default detector also fits a **local constant background nuisance parameter** uniformly at every trial position, rather than interpreting a common residual pedestal as point-source flux. For a finite PSF window, let `B=sum(P/V)`, `C=sum(1/V)`, `Q=sum(I/V)`. It replaces each numerator with `N-F*B*Q/C` and information with `D-F²*B²/C`. This is the weighted two-parameter [PSF, background] fit, including the loss of information from fitting the background; it is an extension of the ideal known-background Paper I statistic, not an exact reproduction. The regression tests check fractional shifts, masked samples, flux response and noise scale. It removes constant local backgrounds, not arbitrary source confusion or structured noise.

Products per sequence: weighted coadd, PSF flux estimate `N/D`, nominal significance `N/sqrt(D)`, empirical-MAD-normalized significance, odd/even flux-difference null, coverage/information extensions, frame diagnostics, blind detections, negative peaks and DS9 regions. All positions are relative to the FIRST exposure; CSV coordinates are zero-based. A fixed empirical threshold of 5 and minimum 80% frame coverage are used everywhere. Catalog positions are never inputs.

The nominal model assumes independent background-dominated native noise. Fitted PSF uncertainty, preprocessing correlations, temporal/systematic residuals and confusion are not solved by a MAD scale. Central-distribution empirical normalization is **not** a calibrated Gaussian-tail false-alarm probability. Positive/negative peak counts and odd/even checks are diagnostics, not proof that each peak is a real source. Random source-injection and matched-sample completeness/false-positive tests are required before claiming a detection-depth improvement. The current 80-frame baseline must not be compared as equal exposure to old 16-frame ASTERIS coadds.

In particular, fitting a finite-window local background gives the detection response negative sidelobes around bright sources. Negative-peak counts therefore are not an unbiased false-positive estimate near such sources. An empirical scale above 2 is flagged as a shared-structure warning; this threshold is a diagnostic convention, not a significance test.

## Historical entry points

The legacy N2N/standard-ASTERIS target-photometry blend-calibration functions are **not** part of this blind workflow and must not be reused with new blind manifests. They remain for reproducing historical experiments and need a separate catalog-free calibration design before future reuse. ASTERIS-paper does not use those blend-calibration functions. No legacy training, strength calibration, inference or learned-model output was run or changed by this batch.
