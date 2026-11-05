
# BrillouinPy

BrillouinPy is a Python package for analyzing Brillouin light scattering (BLS) microscopy and imaging data - from raw spectra to instrument response deconvolution, damped harmonic oscillator (DHO) fitting, and multivariate analysis.

Its data model and pipeline architecture originate as a fork of [RamanSPy](https://github.com/barahona-research-group/RamanSPy), a package built for Raman spectroscopy. Raman and Brillouin analysis share a lot of structure - spectral containers, preprocessing pipelines, decomposition/clustering workflows - but the two modalities differ in the physics that matters: Brillouin spectra don't need baseline correction the way Raman spectra do, since the background in BLS experiments is typically flatter, but they do require removal of the elastically scattered light (Rayleigh or reference-beam peak) and are usually interpreted through a Lorentzian/DHO line shape rather than discrete Raman bands. BrillouinPy keeps the parts of RamanSPy's architecture that generalize well and replaces or extends the rest with Brillouin-specific preprocessing and analysis: masked-array/NaN-aware pipelines, instrumental response function deconvolution, and Lorentzian/DHO peak fitting. See [`NOTICE.md`](NOTICE.md) for a file-by-file breakdown of what was adapted from RamanSPy and what changed.

**Note:** this package was renamed from `BrillouinAnalyzer` to `BrillouinPy` earlier than v0.2.0. Existing code doing `import brillouinanalyzer as bp` keeps working unchanged (it now emits a `DeprecationWarning` and re-imports everything from `brillouinpy`) - update your imports to `import brillouinpy as bp` when convenient.

This project is licensed under the BSD 3-Clause License (see [`LICENSE`](LICENSE)); see [`NOTICE.md`](NOTICE.md) for the third-party (RamanSPy, VCA) components it incorporates and their respective licenses.

**Repository:** the canonical, publicly accessible repository is [github.com/timm-landes/brillouinpy](https://github.com/timm-landes/brillouinpy). A mirror also exists on the Leibniz University Hannover GitLab instance (`gitlab.uni-hannover.de/phytophotonics/brillouinpy`) for internal development - note that this mirror is only reachable from within the LUH network/SSO and is not usable by external contributors or users.

**Contributing:** bug reports and pull requests are welcome on the GitHub repository - see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, coding conventions, and how to run the test suite/linter locally.

## Documentation
A step-by-step [Tutorial](docs/tutorial/) walking through the `examples/` folder, and the full API reference (built from the doc strings in this package), are published via GitHub Pages: **https://timm-landes.github.io/brillouinpy/**. They're rebuilt automatically on every push to `main`, i.e. on each release (see [`.github/workflows/docs.yml`](.github/workflows/docs.yml)); day-to-day development happens on `develop`. The same build also runs on the internal GitLab mirror's CI (find the link under **Deploy → Pages** in the GitLab project), reachable only from within the LUH network.

To build it locally instead:
```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
Then open `docs/_build/html/index.html` in your browser.

## Installation
The instructions below use [Anaconda](https://www.anaconda.com/download/success) as the Python distribution, which is recommended if you are new to Python. Any other Python ≥ 3.11 environment works just as well.

### Prerequisites
 1. Install Anaconda: [download](https://www.anaconda.com/download/success).
 2. Create a new conda environment: `conda create --name <env-name>`, replacing `<env-name>` with a name of your choice.
 3. Activate the environment: `conda activate <env-name>`.
 4. Install pip and git: `conda install pip git`.

You can then install BrillouinPy in one of two ways:
 1. From the **Git repository** - recommended if you only use the package and want to stay up to date with the latest version.
 2. From a **local clone** - recommended if you intend to modify the package code itself.

**Note:** unless you have a specific reason to upgrade, keep your conda installation as-is - upgrading conda itself can break existing environments.

### Install from the Git repository
```bash
pip install git+https://github.com/timm-landes/brillouinpy.git
```

### Install from a local folder

1. Choose a folder to hold the clone and navigate to it:
   ```bash
   cd <your folder path>
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/timm-landes/brillouinpy.git
   ```
3. Install the package:
   ```bash
   pip install .
   ```
   Or, for an editable install that picks up local changes without reinstalling:
   ```bash
   pip install -e .
   ```

**LUH-internal note:** if you're working from inside the university network and prefer the GitLab mirror instead, replace the URL above with `git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinpy.git` (requires a GitLab SSH key registered with your LUH account - see [this tutorial](https://www.youtube.com/watch?v=Vmt0V6a3ppE) if you haven't set one up). This is not reachable outside the LUH network/SSO.

## Updates
1. Activate your environment: `conda activate <env-name>`.
2. If installed from the git repository:
   ```bash
   pip install --upgrade git+https://github.com/timm-landes/brillouinpy.git
   ```
3. If installed from a local folder, navigate to it and run:
   ```bash
   pip install --upgrade .
   ```
   or, for an editable installation:
   ```bash
   pip install -e .
   ```

## Example
The [`examples`](examples/) folder contains complete, runnable scripts covering typical workflows, including multivariate/ML-based analysis. For a guided, illustrated walkthrough, see the [Tutorial](https://timm-landes.github.io/brillouinpy/tutorial/).

The snippet below loads a measurement, preprocesses it and fits the Brillouin
doublet. The [`examples/`](examples/) scripts cover each step in more depth (and
fall back to synthetic data, so they run without a dataset configured). Example
data is available [here](https://seafile.projekt.uni-hannover.de/d/ae7aff2e3bf14f119ed5/)
(password `brillouin_test_data`).

```python
import numpy as np
import matplotlib.pyplot as plt

import brillouinpy as bp

if __name__ == '__main__':
    # Raw measurement files live in <project_path>/data, with the (x, y, z, t)
    # coordinates encoded in each filename (e.g. "Bri_0_0_0_0.DAT").
    project_path = r'complete_path_to_your_project'

    # Load every spectrum under <project_path>/data into one masked
    # (x, y, z, t, spectral) array. .squeeze() drops the singleton z / timepoint
    # axes of a plain 2D scan, leaving (x, y, spectral); keep them (and use
    # bp.SpectralVolume below) for a z-stack.
    intensity = bp.io.prepare_brillouin_data(project_path, 'Brillouin').squeeze()

    # The frequency-shift axis is derived from the interferometer scan
    # parameters. If the measurement ships a META.json, read them automatically:
    try:
        spectral_axis = bp.io.brillouin_spectral_axis_from_meta(
            project_path, no_of_channels=intensity.shape[-1],
        )
    except (FileNotFoundError, KeyError):
        # otherwise enter them by hand - these must match your setup!
        spectral_axis = bp.utils.brillouin_spectral_axis(
            mirror_spacing=3e-3,    # [m]
            scan_amplitude=309e-9,  # [m]
            no_of_channels=intensity.shape[-1],
        )

    brillouin_data = bp.SpectralImage(intensity, spectral_axis)  # bp.SpectralVolume for a z-stack

    # Anything else in META.json (sample name, operator, acquisition date) can be
    # attached to the object; it rides along through processing and is written
    # out again on export (io.to_brim / io.to_hdf5_bls).
    try:
        brillouin_data.metadata = bp.io.read_meta(project_path)
    except FileNotFoundError:
        pass

    bp.plot.mean_spectra(brillouin_data, title='Raw Brillouin data', yscale='log')
    plt.show()

    # Preprocessing pipeline: normalise per pixel, remove the instrument
    # response function (IRF) / elastic peak, then smooth.
    pipeline = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
        bp.preprocessing.misc.Deconvoluter_IRF(offset=65, iterations=4, padding=None),
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
    ])
    pp_brillouin_data = pipeline.apply(brillouin_data)

    bp.plot.mean_spectra(pp_brillouin_data, title='Processed Brillouin data', yscale='linear')
    plt.show()

    # Fit one damped-harmonic-oscillator doublet per pixel.
    model = bp.analysis.fit.DHO(
        expected_peaks=1,
        p0=[20, 8, 0.5, 0, 0],  # [amplitude, shift (GHz), HWHM (GHz), background, axis_shift]
        bounds=([0, 5, 0.1, -5, -5], [100, 15, 10, 5, 5]),
    )
    fit_result, metrics = model.apply(pp_brillouin_data)
    print(f"mean Brillouin frequency shift: {np.nanmean(fit_result[..., 1]):.3f} GHz")
```

## Usage in Spyder
To use the package's environment as the Spyder console, install the Spyder kernels into that environment (via conda, not pip):
```bash
conda install spyder-kernels
```
Spyder 5.x requires a specific kernels version instead:
```bash
conda install spyder-kernels=2.5
```
Then, in Spyder, open a new console on that environment (Spyder 6), or set it as the default environment via the interpreter selector in the bottom status bar (Spyder 5).


## Changelog
See [`CHANGELOG.md`](CHANGELOG.md).