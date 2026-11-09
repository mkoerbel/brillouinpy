# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Poisson maximum-likelihood fitting (`loss='poisson'`)** on `DHO`, `Lorentzian`,
  `Gaussian` and `PeakFit` for raw photon-counting spectra. Each pixel is first fitted by
  least squares, then refined by minimising the Poisson negative log-likelihood (L-BFGS-B).
  The returned covariance is the inverse Fisher information. On synthetic Poisson DHO
  spectra the shift / linewidth scatter reaches the Cramér-Rao bound (least squares:
  +14 % / +38 % above it). Works with `irf=` (1-D and per pixel). A negative `Background`
  lower bound is clipped to 0, and non-count input triggers a warning.
  `expected_peaks='auto'` with `loss='poisson'` raises `NotImplementedError`.
  `loss='squares'` stays the default, so existing results are unchanged.

### Fixed
- **`utils.brillouin_spectral_axis` channel spacing is now `2L/N` instead of `2L/(N−1)`.**
  The axis was built with `np.linspace(-L, L, N)`, which puts the outermost channels
  exactly at ±L. According to the TFP and GHOST manuals, however, the scan range `2L` is
  divided into `N` channels of equal duration. The axis now returns the channel centres,
  symmetric about 0 (the old axis × (N−1)/N). Previously, all frequency parameters (BFS,
  linewidth, spectral offset) came out too large by N/(N−1): +0.4 % at 256, +0.2 % at
  512 and +0.1 % at 1024 channels (water: ≈ +30 / +15 / +7 MHz on 7.58 GHz). This also
  produced a spurious BFS offset between measurements with different channel counts
  (water, 256 vs 1024 channels: 21 MHz before the fix, < 1 MHz after).
  `io.tfp.brillouin_spectral_axis_from_meta` is affected the same way. **Results change
  for existing data**, so re-run the fits for any values you want to compare with
  previously analysed data.
- **Zero-count channels are no longer dropped from fits.** `_prep_pixel_spectrum`
  (`analysis/fit/core.py`) used to turn every 0 into `NaN`. On raw photon-counting data
  most wing channels are legitimately 0, so this biased fits towards a positive background
  and a too-narrow line (synthetic Poisson test: HWHM −9 %, background +0.8 counts;
  affects `DHO(irf='auto')` on raw TFP data in particular). Only masked / `NaN` channels
  are excluded now. A pixel with no counts at all still yields `NaN` parameters.
- **`IRF_Remover`** now sets removed channels to `NaN` instead of 0 (like
  `Deconvoluter_IRF`), works on any spatial dimensionality (incl. the raw
  `(x, y, z, t, spectral)` shape), clips `start - offset` at 0 instead of wrapping around,
  and accepts the documented `offset=None` (`Deconvoluter_IRF` too).
- **'from_brim' now handles nD frequency axes.** Importing brimfile data via `from_brim` 
  with non-unique frequency axes no longer causes an error because the spectral axis is 
  expected to be 1D. The frequency axes will be interpolated if the used brimfile 
  version >= 1.7.6, otherwise return `AttributeError`.


### Changed
- **Behaviour change:** on raw TFP data the shutter-blocked channels (exactly 0) were
  previously dropped implicitly by the zero→`NaN` rule. They now count as data unless
  blanked explicitly, e.g. by choosing an `IRF_Remover`/`Deconvoluter_IRF` `offset` that
  covers them, or by masking / `Cropper`.
- **Python 3.10 is no longer supported.** BrillouinPy now requires Python >= 3.11, the
  minimum `brimfile` needs. Python 3.10 has reached its end of life.

## [0.6.0] - 2026-09-21

### Added
- **`tests/test_io_legacy.py`, `tests/test_io_multimodal.py`, extended
  `tests/test_io_tfp.py`** cover the filename-coordinate-encoded `.DAT`/`.csv`/`.txt`
  loaders (`io.legacy.load_spectral_image`/`load_spectral_image_brio2`,
  `io.tfp.extract_coordinates`/`import_DAT_File`/`prepare_brillouin_data`,
  `io.multimodal.prepare_raman_data`), previously almost entirely untested.
- **Coverage reporting in CI.** New `pytest-cov` test dependency, `[tool.coverage.run]`
  (`source = ["brillouinpy"]`, omitting `utils.py`/`benchmarks/`), and the
  `integration` CI job now runs `pytest --cov --cov-report=term-missing
  --cov-report=xml` (the tolerant `test` matrix job is unchanged).
- **`tests/test_export_misc.py`** covers `io.export.fit_to_tiff` and
  `_dho_parameter_names` (previously untested): file naming, `parameter_names`
  override/length-mismatch, wrong-`ndim`/missing-`expected_peaks` errors, and
  masked-array handling.
- **New tutorial page** [Multimodal data and channels](docs/tutorial/multimodal-channels.md)
  covers the `channels` API end to end: attaching channels, grid-conformant vs.
  non-conformant behaviour through spatial operations, `apply_to_channel`/
  `drop_channel`, `as_channel`/`from_channel` for label/mask maps, and the
  pickle round-trip. New companion script `examples/18_multimodal_channels.py`.
- **`tests/test_ramanspy_interop.py`** verifies the duck-typing compatibility
  with RamanSPy that `docs/design/multimodal_container.md` claims: a
  brillouinpy `SpectralObject` passed straight into a RamanSPy preprocessing
  step. `ramanspy` is a new (test-only) extra. This includes passing a
  **channel value** (`container.channels[name]`) to a RamanSPy step directly, or
  via `apply_to_channel` - no conversion code needed, since a channel is already
  a full `SpectralObject`; see `examples/17_channel_ramanspy_interop.py`.
- **`as_channel`/`from_channel`** wrap a per-pixel array with no spectral axis
  (classification labels, masks, fit-result maps) as a channel-ready
  `SpectralObject` and back. `SpectralContainer.with_channel`/`drop_channel`/
  `apply_to_channel` add, remove, or transform a single named channel,
  returning a new object and leaving the original untouched.

### Changed
- **`channels` values must now be `SpectralObject`s.** A bare `numpy.ndarray`
  (previously accepted silently, then failing later in `flat`, `__getitem__`,
  or stacking) now raises `TypeError` pointing at `as_channel` at
  construction time.

### Fixed
- **`prepare_raman_data` no longer crashes on a single-data-row Raman CSV file.**
  Extracting the wavelength axis (and, per-file, the spectrum) from a file with
  exactly one `(wavelength, intensity)` row hit `np.loadtxt(..., usecols=0)`
  collapsing to a 0-d array, which `[::-1]` can't slice - `IndexError` outside the
  per-file `try/except` (a hard crash), or silently caught-and-masked inside it
  (silent data loss) depending on which occurrence. Both now go through
  `np.atleast_1d(...)` first.
- **`load_spectral_image_brio2`'s docstring no longer describes a `NameError` bug
  that doesn't reproduce.** Verified empirically (mismatched file count under both
  loaders) - the sanity check it referenced already defines `z_dim`/`timepoint`
  correctly and warns as intended; the docstring's "Notes" section was stale.
- **`fit_to_tiff` now honours a masked `fit_result`.** It called `np.asarray()`
  on its input, which silently strips the mask off a `numpy.ma.masked_array` -
  the `np.ma.filled(..., np.nan)` further down was then a no-op, and masked
  pixels were written with their raw underlying value instead of `NaN`. Now
  uses `np.ma.asarray()`, which preserves a mask if present.
- **`from_channel` no longer silently mangles a real spectral channel.** Given
  an object with a genuine spectral axis (e.g. Raman) instead of one built by
  `as_channel`, it returned `spectral_data[..., 0]` - the right spatial shape,
  plausible-looking values, wrong data. It now raises `ValueError` for any
  object whose spectral axis isn't length 1.

## [0.5.0] - 2026-09-10

### Added
- **New `Gaussian` fit model** - a Gaussian doublet with the same interface,
  parameter layout, `irf=` support and return values as `DHO` / `Lorentzian`;
  also selectable as `segmented_fit(model='gaussian')`.
- **`elastic=True`** on `DHO` / `Lorentzian` / `Gaussian` fits the `*_elastic`
  variant, whose shared baseline is `Background + elastic_slope * (x -
  axis_shift)` (one extra trailing parameter) to absorb the sloping residual wing
  of an un-blanked elastic peak. Incompatible with `irf=`.
- **Lineshape registry.** `fit.lineshapes.register_lineshape(name, core=...)`
  adds a lineshape family from a single single-mode core function; the 1/2/3-mode
  model functions are assembled automatically. Fit any registered family with the
  new generic `fit.PeakFit(model=...)`.
- `estimate_p0` gained a `model=` keyword (default `'dho'`, unchanged behaviour;
  the `*_elastic` families get an extra trailing `0.0`).

### Fixed
- **`to_brim` no longer crashes on a flat metadata dict.** It expected
  `spectral_object.metadata` to be keyed by brim categories, so a plain vendor /
  `META.json` dict (e.g. attached by a loader) raised `KeyError` on the first
  non-category key. Such a dict is now normalised: well-known keys (`Date`,
  `Sample`, laser wavelength/power, ...) are mapped onto the brim schema and the
  rest is bundled into `Experiment.Info` so nothing is lost.
- **`to_hdf5_bls`** no longer masks a validation error (e.g. a `fit_result`
  without `expected_peaks`) with a `WrapperError_Save` from the cleanup path.
- **`brillouinpy.plot`** used `plt.cm.get_cmap()`, removed in matplotlib 3.9, in
  `plot.image`/`plot.volume`/`plot.spectra` - every default-colour plot raised on
  matplotlib >= 3.9. Switched to `plt.get_cmap()`; `matplotlib>=3.7` is now the
  declared lower bound.
- **`plot.peak_dist`** raised `TypeError` when given a single spectral object
  (as its own docstring/signature allow) instead of a list; it now wraps a bare
  object like the other plot functions.
- **`plot.spectra` / `plot.mean_spectra` with `plot_type="stacked"`** raised
  `AttributeError` (`Figure` has no `set_yscale`); the y-scale is now applied to
  each stacked sub-axes.
- **`plot.save`** was defined but never exported from `brillouinpy.plot`, so
  `plot.save(...)` raised `AttributeError`.
- **`HDF5_BLS` test/optional dependency** now requires Python >= 3.12 (was
  `>= 3.11`): the installed package's own code uses a nested same-quote
  f-string, which is a `SyntaxError` on 3.10/3.11 despite its declared
  `requires-python = ">=3.6"` - an upstream bug
  ([bio-brillouin/HDF5_BLS](https://github.com/bio-brillouin/HDF5_BLS)).
  `tests/test_export_hdf5_bls.py` now skips cleanly instead of failing
  collection when this happens.

### CI / tests
- CI now runs the test suite on Python 3.10, 3.11 and 3.12 (was 3.12 only), and
  installs `brimfile` / `HDF5_BLS` on 3.11+ so `tests/test_export_brim.py` and
  the new `tests/test_export_hdf5_bls.py` actually run instead of skipping
  everywhere. Added a `test` extra (`pip install ".[test]"`).
- New `tests/test_analysis_fit_deprecation.py` locks in the backwards
  compatibility of the old `brillouinpy.analysis.fitmodel` / `.lineshapes` /
  `.segmented` / `.FitStep` import paths.
- New `tests/test_plot_smoke.py` (every public `brillouinpy.plot` function,
  including `save`, all `plot_type`s, on the Agg backend) and
  `tests/test_plot_core.py` (the internal `plot._core` helpers, including an
  `xfail(strict=False)` reproducing the `np.unique` ragged-shift-axes case
  tracked in #10). The plot modules previously had no coverage - that's where
  the `plt.cm.get_cmap`, `peak_dist`, `plot_type="stacked"` and `plot.save`
  export bugs above were hiding.
- Added a `Documentation` URL (GitHub Pages site) to the project metadata.

### Changed
- **Fitting reorganised into the `brillouinpy.analysis.fit` subpackage.** The
  lineshape models moved out of the old `fitmodel` module into
  `fit.lineshapes`; the fitters (`DHO`, `Lorentzian`, `Gaussian`, `PeakFit`,
  `estimate_p0`, `estimate_peak_count`, `irf_kernel`) are in `fit.core`;
  `segmented_fit` is in `fit.segmented`; the `FitStep` base class is in
  `fit.step`. Everything is re-exported flat, so `bp.analysis.fit.DHO`,
  `bp.analysis.fit.segmented_fit`, `bp.analysis.segmented_fit` all work. The old
  paths (`brillouinpy.analysis.fitmodel`, `.segmented`, `.lineshapes`,
  `.FitStep`) still import as thin re-exports with a `DeprecationWarning` and
  will be removed in a future release.
- **Packaging moved to PEP 621.** `setup.py` is removed; all metadata now lives
  in `pyproject.toml` (`[build-system]` + `[project]`). `requires-python` is
  `>=3.10` (was an inconsistent `>=3.8`), the license is declared as the SPDX
  expression `BSD-3-Clause`, and `LICENSE` / `NOTICE.md` / `THIRD_PARTY_LICENSES`
  ship as metadata license files. `pysptools` is no longer a hard dependency: it
  only backs the optional `PPI`/`FIPPI`/`NFINDR` endmember extractors and is now
  the `brillouinpy[unmix-eea]` extra (VCA and all abundance methods are pure
  NumPy/SciPy and unaffected).
- `LICENSE` is now a verbatim BSD-3-Clause text (the trailing third-party
  reference block moved wholly into `NOTICE.md`) so automated license detection
  recognises it.

## [0.4.0] - 2026-09-07

Bundles everything on `develop` since 0.3.1: spectral phasor analysis, VCA/NMF/PCA/k-means
exploratory tools, `segmented_fit`, IRF-convolution fitting, automatic peak-count
selection, and the `analysis.mechanics` module - plus a review-driven cleanup of
the `DHO`/`Lorentzian` fit models. **Breaking:** the fit models' parameter
conventions are now pinned down (`Asymmetry` → `axis_shift`, `LineWidth` is the
HWHM, `Background` counted once, real Lorentzian doublet) and the derived
`analysis.mechanics` / `io.export` linewidths change accordingly - refit any data
whose fit parameters were stored with 0.3.x. See below.

### Added
- **`segmented_fit(..., model=...)`.** The segmented fit can now use a Lorentzian
  lineshape (`model='lorentzian'`) instead of the DHO (`model='dho'`, default).
  `SegmentedFitResult` is unchanged (same parameter layout; `linewidth` is the
  HWHM either way).
- **Tutorial: new "Multi-component and mixed images" page** covering the three
  spatial scales of a Brillouin image (phonon wavelength / voxel / structure,
  after Prevedel et al. 2019), why fitting the maximum peak count everywhere
  fails, `segmented_fit`, `estimate_peak_count`, `DHO(expected_peaks='auto')` and
  how they compare to the model-free methods. The "Fitting spectra" page gains
  sections on the parameter conventions (HWHM, `axis_shift`), `DHO` vs
  `Lorentzian`, and `max_workers`.

### Changed
- **`fitmodel` internals consolidated.** The pixel-wise fit now runs through a
  single `_fit_concurrent` (lineshape family selected by `model=`) plus the
  separate `_fit_concurrent_DHO_auto` for peak-count selection; the three unused
  runners (`_fit_concurrent_DHO`, `_fit_concurrent_DHO3` with its
  `shared_memory` machinery, `_fit_concurrent_lorentzian`) are removed. No
  public API change. `DHO` and `Lorentzian` gain an optional `max_workers=`
  argument (default: a quarter of the visible CPUs, as before). `Lorentzian`
  gains `irf=` support and now applies the same masked-array / NaN / zero-channel
  handling as `DHO` (previously it silently returned zeros for failed pixels);
  its `p0` and `bounds` are now optional. Per-pixel diagnostics go through the
  `brillouinpy.analysis.fitmodel` logger instead of `print()`; a wrong-length
  `p0` now raises a `UserWarning` instead of printing.

### Changed (breaking)
- **DHO / Lorentzian fit models, parameter conventions made explicit.** These
  change the meaning and, for the Lorentzian, the values of fitted parameters -
  refit any data whose parameters were stored from an earlier version.
  - The shared trailing parameter formerly called `Asymmetry` is renamed
    `axis_shift` (it was always a rigid shift of the frequency axis, `x -
    axis_shift`, never an asymmetry). Affects `DHO`/`Lorentzian` results,
    `estimate_p0`, and the `io.export` parameter names. No compatibility alias.
  - `LineWidth` is now documented as the **half width at half maximum** (HWHM,
    `Gamma / 2`); the FWHM is `2 * LineWidth`. The DHO formula is unchanged;
    this only pins down a convention that was previously implicit.
  - `Background` is now added **once** for multi-mode models (`_DHO_2/_DHO_3`,
    `_Lorentzian_2/_Lorentzian_3`) instead of once per mode, matching the
    IRF-convolved model. A quantitative background fitted with an older version
    was the true value divided by the number of modes.
  - `Lorentzian` is now an actual Lorentzian doublet (two Lorentzians of HWHM
    `LineWidth` at `axis_shift +/- freqShift`) rather than an unnormalised
    DHO-shaped variant.
- **`analysis.mechanics`** now converts `LineWidth` (HWHM) to the FWHM internally,
  so `loss_tangent`, `loss_modulus` and `longitudinal_viscosity` are a factor of
  two larger than before and now equal `tan(delta) = Gamma / shift` etc.
  directly.
- **`io.export`** writes the FWHM (`2 * LineWidth`) into brim / HDF5_BLS
  `width` / `linewidth` fields, which use the FWHM convention.

## [0.3.1] - 2026-09-03

Documentation and repository-workflow changes only; no code changes.

### Changed
- Development now happens on a `develop` branch; `main` is release-only and
  protected. Added [`RELEASING.md`](RELEASING.md) and a "Branching and pull
  requests" section to `CONTRIBUTING.md`. Refreshed the `README.md` quick-start
  example (META.json-derived frequency axis, `metadata`, a DHO fit).
- Restructured the documentation site: the front page is now a short landing page
  instead of the full README, installation moved to its own page, and the
  single-file tutorial was split into per-stage pages under `docs/tutorial/`
  (loading / preprocessing / exploratory analysis / fitting / exporting / custom
  loader). The GitLab wiki mirror of the tutorial is retired in favour of the
  GitHub Pages site.

## [0.3.0] - 2026-09-03

Makes acquisition metadata and pixel size a first-class part of the data model,
adds generic multimodal support to `SpectralContainer`, and reorganises the
loading/export functions under a new `brillouinpy.io` subpackage.

### Added
- `SpectralContainer` (and subclasses) now carry `metadata` and `px_size_um`
  dicts as first-class attributes, settable via the constructor. Both always
  exist on an instance (default: `{}` and `{"x": None, "y": None, "z": None}`),
  and derived objects (spatial slices, `flat`, `mean`, `variance`, `tolist`,
  `from_stack`, `from_image_stack`, `layer`) inherit a copy. Pickles written
  before these attributes existed load with the defaults filled in.
- `io.from_brim` now also attaches the data group's pixel size to the returned
  object as a `px_size_um` dict (`{"x"/"y"/"z": float_or_None}`, in micrometers),
  alongside the existing `metadata` dict.
- `SpectralContainer` can now carry co-acquired data in a `channels` dict
  (`{name: SpectralObject}`) - e.g. a Raman or fluorescence channel on the same
  sample. The mechanism is generic (no modality-specific types or reserved
  names); the primary `spectral_data` stays the Brillouin data. A channel whose
  spatial `shape` matches the container's follows along through indexing,
  `flat` and stacking; other channels are passed through untouched. Introspect
  with the new `channels_grid_conformant` property and `repr()`. `mean` /
  `variance` carry no channels. `io.to_brim` / `io.to_hdf5_bls` do not export
  channels (pickle `save`/`load` round-trips them). See
  [`docs/design/multimodal_container.md`](docs/design/multimodal_container.md).
- `Spectrum` / `SpectralImage` / `SpectralVolume` now validate their
  dimensionality at construction (`ndim` 1 / 3 / 4) and raise a clear
  `ValueError` on a mismatch, instead of the class name silently not matching the
  data shape. `core._create_data` still picks the class by `ndim` (base
  `SpectralContainer` for 2D / 5D+); preprocessing and analysis steps are
  unaffected as they deep-copy rather than reconstruct.
- `SpectralContainer.peaks()` is now available on any spectral object, not just
  `Spectrum`; for an object with spatial dimensions it operates on the `mean`
  spectrum.
- A new "Writing your own loader" section in the tutorial (`docs/tutorial.md`),
  explaining the minimal `(intensity_array, spectral_axis)` contract a custom
  loader needs to satisfy to work with the rest of the package, using
  `io.tfp.prepare_brillouin_data` as a worked example.
- `CONTRIBUTING.md`, outlining development setup, project layout and conventions.

### Changed
- `io.to_brim` reads `metadata` and the x/y/z pixel steps off the passed object
  when the corresponding arguments are omitted, so a
  `from_brim` -> process -> `to_brim` round-trip preserves metadata and pixel
  size without the caller re-supplying them. Explicit arguments still take
  precedence.
- **Reorganised acquisition-side loaders and export/interop functions under a new
  `brillouinpy.io` subpackage**, split by purpose:
  - `io.tfp` - this group's actively used tandem Fabry-Perot loading path
    (`prepare_brillouin_data`, `read_meta`, `brillouin_spectral_axis_from_meta`,
    `extract_coordinates`, `import_DAT_File`).
  - `io.legacy` - superseded loaders kept for reading old datasets
    (`load_spectral_image`, `load_spectral_image_brio2`).
  - `io.multimodal` - loaders for modalities other than the group's primary setup
    (`prepare_raman_data`).
  - `io.export` (formerly the top-level `brillouinpy.export` module) - `brim`/
    `HDF5_BLS` interoperability and TIFF export; the most commonly used functions
    (`to_brim`, `from_brim`, `list_brim_measurements`, `to_hdf5_bls`,
    `from_hdf5_bls`, `list_measurements`) are also re-exported directly on
    `brillouinpy.io` for convenience.

  `brillouinpy.utils` keeps only the generic, setup-independent functions
  (`is_aligned`, `wavelength_to_wavenumber`/`wavenumber_to_wavelength`,
  `raman_spectral_axis`, `fsr`, `brillouin_spectral_axis`, `TFP_IRF_Analysis`).
  The old `brillouinpy.utils.*` names for the moved functions, and the
  `brillouinpy.export` module itself, keep working unchanged for now - both emit
  a `DeprecationWarning` pointing at the new location. These compatibility
  shims may be removed in a future release; update your code to call
  `brillouinpy.io.*` directly when convenient.
- Removed a stray leftover `print(data)` debug statement from
  `prepare_brillouin_data`'s CSV-loading branch (carried over unnoticed from
  `load_spectral_image`; harmless but noisy).

### Fixed
- `io.to_brim` now always writes an `Experiment.Datetime` metadata field (from the
  new `acquisition_datetime` argument, a value already present in `metadata`, or
  the current local time). Files written without any Experiment metadata made
  BrimView fail with `AttributeError: 'NoneType' object has no attribute 'get'`.
- `io.from_brim` no longer includes empty metadata categories in the returned
  `metadata` dict.

## [0.2.0] - 2026-08-27

First public release. The project is now developed openly on GitHub, with the
existing GitLab instance kept as an LUH-internal mirror.

### Added
- Public repository at [github.com/timm-landes/brillouinpy](https://github.com/timm-landes/brillouinpy),
  now the canonical, publicly accessible home of the project (mirrored
  internally on `gitlab.uni-hannover.de/phytophotonics/brillouinpy`, reachable
  only from within the LUH network/SSO).
- `.github/workflows/docs.yml`, automatically building and publishing the
  Sphinx documentation to GitHub Pages
  (`https://timm-landes.github.io/brillouinpy/`) on every push to `main`.

### Changed
- **Renamed the package from `BrillouinAnalyzer` to `BrillouinPy`** (import name
  `brillouinpy`). `brillouinanalyzer` remains installable as a thin backwards-
  compatibility shim - `import brillouinanalyzer as bp` still works, but now
  emits a `DeprecationWarning` and re-exports everything from `brillouinpy`.
  Update your imports to `import brillouinpy as bp` when convenient; the shim
  may be removed in a future release. The GitLab project itself (and its URL)
  are unaffected by this change for now.

### Added
- `examples/` restructured into nine focused, runnable scripts covering the full
  workflow: loading, preprocessing, IRF removal, fitting, export, spectral
  unmixing (VCA), decomposition (NMF, PCA) and clustering (k-means). Each script
  falls back to synthetic data (`examples/_synthetic_data.py`) when no real
  dataset is configured, so every example runs out of the box.
- A step-by-step, illustrated Tutorial (`docs/tutorial.md`), also mirrored on the
  project's GitLab Wiki, walking through the `examples/` scripts with generated
  figures (`docs/generate_tutorial_images.py`).
- `utils.read_meta` and `utils.brillouin_spectral_axis_from_meta`, which read the
  interferometer scan parameters (mirror spacing, scan amplitude, laser
  wavelength) for `brillouin_spectral_axis` from a measurement's `META.json`
  automatically, transparently handling both the current (nested) and legacy
  (flat) metadata schema found across existing datasets.
- `LICENSE` (BSD 3-Clause) and `NOTICE.md`/`THIRD_PARTY_LICENSES/` documenting the
  RamanSPy- and VCA-derived portions of the codebase.
- `export.to_brim`/`export.from_brim`/`export.list_brim_measurements`, for
  reading/writing the [brim](https://github.com/brillouin-imaging/Brillouin-standard-file)
  format (a Zarr-based standard for Brillouin microscopy data, also readable by
  the napari/Fiji brim viewer plugins), via the optional `brimfile` package
  (`pip install brillouinanalyzer[brim]`; needs Python >= 3.11).
- `export.to_brim(..., as_zip=True)`, to write a single `.brim.zip` archive
  instead of a `.brim.zarr` directory.

### Fixed
- `analysis.unmix.NFINDR` called a non-existent `pysptools.eea.nfindr.NFINDR` API
  and passed data in the wrong shape; it now uses `eea.NFINDR().extract(...)` with
  a correctly shaped input cube.
- `export.to_brim` now writes the root `Subtype` attribute explicitly (as
  `'none'`) - `brimfile`'s own `File.create()`/`create_data_group()` never write
  it unless one of the `brimfile.subtypes.*` helpers is used, and while
  `brimfile`'s own reader tolerates that, at least one downstream viewer
  (BrimView, as of `brimfile` 1.7.0) doesn't and fails to open the file with
  `ValueError: Invalid subtype: None`.
- `export.to_brim` no longer writes a `null` pixel size for a single z-slice
  (e.g. a plain `SpectralImage`, where the z-axis is a placeholder to begin
  with) when `z_step_um` isn't given, since that also tripped up BrimView.

### Changed
- `setup.py`'s license metadata corrected from a placeholder `MIT` classifier to
  `BSD-3-Clause`, matching the added `LICENSE` file.

## [0.1.1]

- Added support for missing data points. The pipeline now warns the user when
  points are missing instead of stopping execution; missing values are treated
  as `None`.

## [0.1]

- Initial upload.
