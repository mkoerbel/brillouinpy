# Fitting spectra

*Script: `examples/09_fit_spectra.py`*

`analysis.fit.DHO` fits a Damped Harmonic Oscillator lineshape - the
physically correct model for a Brillouin peak, describing the material's acoustic
phonon mode as a damped oscillator driven by thermal fluctuations - to every
spectrum in a spectral object, in parallel across a process pool. This step goes
back to the single-mode data from [loading](loading.md)/[preprocessing](preprocessing.md),
so `expected_peaks=1` here - for the two-doublet data explored in
[exploratory analysis](exploratory-analysis.md), `expected_peaks=2` would be the
right setting instead (see below):

```python
dho_fit = bp.analysis.fit.DHO(
    expected_peaks=1,
    p0=[0.005, 8.5, 1, 0, 0],  # [Amplitude, FreqShift (GHz), LineWidth = HWHM (GHz), Background, axis_shift]
    bounds=None,
)
fitted_parameters, covariances = dho_fit.apply(preprocessed_image)
# fitted_parameters has shape (x, y, 5)
```

```{image} /_static/tutorial/09_fit_overlay.png
:alt: The preprocessed mean spectrum (purple) with the mean DHO fit result overlaid as a red line, closely tracking both peaks of the Stokes/anti-Stokes doublet.
:width: 480px
:align: center
```

Note that `expected_peaks` counts *modes* (i.e. symmetric peak pairs), not
individual peaks in the plot - `expected_peaks=1` is correct for the single-mode
data used here, even though it visibly produces two peaks (see the note on the
[tutorial overview](index.md)). For `expected_peaks > 1` (e.g. two overlapping
materials each contributing their own mode), `p0`/`bounds` grow to length
`expected_peaks * 3 + 2` (one `[Amplitude, FreqShift, LineWidth]` triplet per mode,
followed by the shared `[Background, axis_shift]`) - one triplet per doublet found
in the variance spectrum, roughly centred on the peak positions read off the mean
spectrum.

A good `p0` (initial guess) matters: `scipy.optimize.curve_fit` (used internally)
is a local optimiser, so a wildly wrong starting frequency shift or linewidth can
make it converge on a spurious local minimum, or fail outright and get discarded as
a `RuntimeError` (the fit logs a debug message per failed pixel via the
`brillouinpy.analysis.fit.core` logger and returns `NaN` there; a wrong-length `p0`
raises a `UserWarning` and is ignored). `bounds` can additionally constrain the
search space (e.g. to keep the frequency shift within a physically plausible range)
once you have a rough idea of where the fit should land.

Instead of hand-writing `p0`, `analysis.fit.estimate_p0(object, expected_peaks)`
derives one from peak detection on the mean spectrum - the same routine
`09_fit_spectra.py` uses.

Because fitting spawns a process pool, fitting scripts must guard their entry point
with `if __name__ == '__main__':` (as all the example scripts do) - required on
Windows in particular, where child processes re-import the launching script from
scratch. Pass `max_workers=N` to `DHO`/`Lorentzian` to cap the pool size (the
default is a quarter of the visible CPUs); set it explicitly on an HPC node or in a
CPU-limited container, where `os.cpu_count()` reports the whole machine.

## Parameter conventions

The fitted parameter vector for each mode is `[I0, freqShift, LineWidth]`, then the
single shared `[Background, axis_shift]` for the whole spectrum:

| Parameter | Meaning |
|---|---|
| `I0` | Peak-area-like amplitude. The peak *height* above the background is `I0 / (pi * LineWidth)` for both models. |
| `freqShift` | Brillouin shift ν_B (GHz). One mode is the symmetric doublet at `axis_shift ± freqShift`. |
| `LineWidth` | **Half width at half maximum** (HWHM, Γ/2) - *not* the FWHM. The full width is `2 * LineWidth`. This matches the HDF5_BLS/BMicro convention, so `analysis.mechanics.loss_tangent` = `Γ_FWHM / ν_B` = tan δ directly. |
| `Background` | Additive constant, counted **once** for the whole model (not once per mode). |
| `axis_shift` | Rigid shift of the frequency axis (GHz); the doublet is symmetric about this value. Named `Asymmetry` in versions ≤ 0.3.1. |

`io.export` and `analysis.mechanics` convert `LineWidth` to the FWHM wherever a
physical linewidth is meant, so you never have to double it yourself
(`examples/15_mechanical_properties.py`, `analysis.mechanics.from_dho_fit`).

## Choosing the lineshape: `DHO` vs `Lorentzian` vs `Gaussian`

`analysis.fit.Lorentzian` and `analysis.fit.Gaussian` have the exact
same interface, return values and `irf=` support as `DHO`:

```python
lor_fit = bp.analysis.fit.Lorentzian(expected_peaks=1, p0=p0, bounds=None)
```

- **`DHO`** is the physically correct model for a Brillouin peak: it describes the
  acoustic phonon as a damped harmonic oscillator driven by thermal fluctuations.
  Its wings fall off faster than a Lorentzian's, and the loss tangent
  `LineWidth / freqShift` is a genuine ratio of the loss and storage moduli. Use it
  by default, and always when you go on to derive mechanical quantities.
- **`Lorentzian`** (a doublet of two Lorentzians of HWHM `LineWidth`) is the simpler,
  more forgiving fit. It is a good choice when the instrument response dominates the
  measured width, when you only need the shift and a rough width, or to cross-check a
  `DHO` result. The two models return the same parameter layout, so nothing
  downstream changes.

- **`Gaussian`** (a doublet of two Gaussians of HWHM `LineWidth`) is mostly useful
  when the measured width is dominated by a Gaussian-like instrument response and
  you are not modelling the IRF explicitly.

For a spectrum that is IRF-limited, the cleanest option is none of these - it is a
`DHO` fit of the *IRF-convolved* model (`irf=`), covered on
[removing the instrument response](irf-fit-comparison.md).

## The elastic peak: `elastic=True`

If the elastic / Rayleigh peak is not blanked, its wing leaves a *sloping*
background under the Brillouin doublet that a single constant `Background` cannot
absorb. Passing `elastic=True` to `DHO` / `Lorentzian` / `Gaussian` fits the
`*_elastic` variant, whose shared baseline is
`Background + elastic_slope * (x - axis_shift)`; the fitted parameter vector then
carries one extra trailing value, `elastic_slope`. Blanking the elastic channels
(`preprocessing.misc.IRF_Remover`) and using the plain model is still preferable
where possible; `elastic=True` is incompatible with `irf=`.

## Raw photon counts: `loss='poisson'`

By default every fit is unweighted least squares (`loss='squares'`). That is the
best estimator only when the noise has the same variance in every channel. Raw TFP
spectra are photon counts, which are Poisson-distributed: the variance equals the
expected count, so it is ~16 at the peak but only ~0.1-0.3 in the wings, where most
channels hold 0 or 1 count. Least squares weights all channels equally and so
largely ignores the low-count wings, which carry most of the linewidth information.

`loss='poisson'` (on `DHO`, `Lorentzian`, `Gaussian` and `PeakFit`) instead
minimises the Poisson negative log-likelihood `sum(mu_k - y_k ln mu_k)` over the
finite channels, starting from the least-squares solution:

```python
dho_fit = bp.analysis.fit.DHO(expected_peaks=1, p0=p0, bounds=bounds, irf='auto', loss='poisson')
```

This maximum-likelihood fit reaches the Cramér-Rao bound, the lowest scatter any
unbiased estimator can achieve. On single TFP spectra of water (512 ms, IRF-convolved
DHO), it reduces the standard deviation of the shift by ~14 % and of the linewidth
by ~30 % compared with least squares. Least squares would need ~1.35× (shift) and
~2× (linewidth) the measurement time to match. The returned covariance is the
inverse Fisher information, which predicts the actual scatter of the estimate.
On count data, the least-squares covariance from `curve_fit` underestimates it
considerably.

- **Only for raw counts**, e.g. straight from `prepare_brillouin_data`, possibly with
  the elastic channels blanked by `IRF_Remover`. Sums of spectra are still Poisson.
  After `Deconvoluter_IRF`, normalisation, background subtraction or averaging the
  noise is no longer Poisson, so stay with `loss='squares'`. A warning flags
  non-integer or negative input.
- The expected count must stay positive, so a negative lower bound on `Background`
  is clipped to 0 (with a warning).
- Because the wings carry more weight, the Poisson fit is also more sensitive to
  model mismatch there (IRF truncation, scan nonlinearity). If the two losses give
  noticeably different mean linewidths on real data while they agree on simulated
  data, the lineshape model is the likely cause, not the estimator.
- Not yet available with `expected_peaks='auto'`.

## Adding your own lineshape

The families above are entries in a registry. To fit a lineshape that is not
built in, write a single-mode doublet core `f(xs, I0, freqShift, LineWidth)` (on
the axis-shifted coordinate `xs = x - axis_shift`), register it, and fit it with
the generic `PeakFit`:

```python
import numpy as np
from brillouinpy.analysis.fit import lineshapes, PeakFit

def pseudo_voigt_core(xs, I0, freqShift, LineWidth, eta=0.3):
    g = lineshapes.LINESHAPES["gaussian"].core(xs, I0, freqShift, LineWidth)
    l = lineshapes.LINESHAPES["lorentzian"].core(xs, I0, freqShift, LineWidth)
    return eta * l + (1 - eta) * g

lineshapes.register_lineshape("pseudo_voigt", core=pseudo_voigt_core)

p0 = bp.analysis.fit.estimate_p0(image, expected_peaks=1, model="pseudo_voigt")
params, cov = PeakFit(model="pseudo_voigt", expected_peaks=1, p0=p0, bounds=(0, np.inf)).apply(image)
```

The 1/2/3-mode model functions `curve_fit` sees are assembled automatically. The
`core` (and any custom `baseline`) must be importable module-level functions -
not lambdas - so they survive being sent to the fit worker processes. A custom
baseline with extra shared parameters is declared via `tail_params=` (see
`register_lineshape`), exactly as the `*_elastic` families do.

## How many modes? (`expected_peaks`)

`expected_peaks` is a fixed integer here because the single-mode tutorial data has
exactly one mode everywhere. When the mode count is unknown or varies across the
image, three tools help:

| Approach | What it does | Use when |
|---|---|---|
| [exploratory analysis](exploratory-analysis.md) (variance spectrum, `variance_explained`) | Model-free count for the whole image | First guess, always cheap |
| `fitmodel.estimate_peak_count(object)` | Fits 1/2/3 modes to the **mean** spectrum, picks the count by information criterion (BIC/AIC) with a robust noise estimate | The count is a property of the sample, not the pixel |
| `DHO(expected_peaks='auto')` | Runs the same model selection **per pixel**, returns a NaN-padded array; `fitmodel.fitted_peak_count()` reads the per-pixel count back | The count genuinely varies pixel to pixel |
| [`segmented_fit`](multi-component.md) | Classify first, then fit exactly the number of modes each class holds | A constant background plus components *added* on top in sub-regions |

Per-pixel model selection (`'auto'`) is noisy and biased upward by anything the DHO
does not capture (a residual elastic wing, a non-Lorentzian real lineshape, an
un-deconvolved IRF); `estimate_peak_count` on the mean spectrum is the steadier
default. The dedicated page on [multi-component fitting](multi-component.md) works
through the differences.

```{image} /_static/tutorial/09_fit_maps.png
:alt: Three side-by-side heatmaps of the fitted amplitude, frequency shift and linewidth per pixel. Amplitude and linewidth show mild pixel-to-pixel noise around a consistent value; frequency shift is mostly uniform around 8.5 GHz with a handful of clear outlier pixels where the fit converged on the wrong value.
:width: 900px
:align: center
```

The per-pixel parameter maps are the actual payoff of fitting an *imaging* dataset:
here, amplitude and linewidth vary mildly pixel-to-pixel around a consistent value
(as expected - the synthetic data has no real spatial structure in those
parameters), while the frequency-shift map is mostly uniform but shows a handful of
clear outlier pixels where noise made the fit converge on the wrong value entirely.
This is normal for real data too - a quick look at each parameter map (or the
returned `covariances`, an estimate of the fit's parameter uncertainty per pixel)
is worth doing before trusting a fit result at face value, e.g. to catch and mask
out such outlier pixels rather than propagating them into further analysis.
