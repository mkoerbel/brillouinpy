# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:14:03 2025

@author: Timm
"""
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, fftconvolve, peak_prominences, peak_widths
from scipy.special import xlogy
from tqdm import tqdm

from .step import FitStep
from .lineshapes import (  # noqa: F401  (re-exported for backwards compatibility)
    LINESHAPES,
    MAX_EXPECTED_PEAKS,
    Lineshape,
    _DHO_1,
    _DHO_2,
    _DHO_3,
    _Lorentzian_1,
    _Lorentzian_2,
    _Lorentzian_3,
    _dho_line,
    _lorentz_line,
    assemble_model,
    model_funcs,
    register_lineshape,
)

_logger = logging.getLogger(__name__)

#: Deprecated alias - lineshape lookup now lives in :mod:`brillouinpy.analysis.fit.lineshapes`.
_model_funcs = model_funcs


def _resolve_max_workers(max_workers):
    """Worker-process count for a pixel-wise fit. ``None`` -> a quarter of the
    visible CPUs (keeps the machine usable during a long fit); an explicit value
    is clamped to at least 1. Pass an int on an HPC node or in a CPU-limited
    container, where ``os.cpu_count()`` is misleading."""
    if max_workers is None:
        return max(1, int((os.cpu_count() or 4) * 0.25))
    return max(1, int(max_workers))


def _prep_pixel_spectrum(intensity_data_slice):
    """One pixel's spectrum as a plain float array with masked channels set to NaN
    (``curve_fit(nan_policy='omit')`` then drops them). Zero counts are valid data
    (on raw photon-counting spectra most wing channels are legitimately 0) and are
    kept; channels to exclude must be blanked explicitly (NaN / masked, e.g. by
    ``IRF_Remover``). Returns ``None`` if the pixel has no usable channel or no
    counts at all (an empty / padding pixel)."""
    y = np.ma.filled(intensity_data_slice, np.nan).astype(float)
    if not np.any(y[np.isfinite(y)]):
        return None
    return y


def estimate_p0(spectral_object, expected_peaks, model="dho"):
    """
    Estimates an initial parameter guess (``p0``) for :class:`DHO`/:class:`Lorentzian` fits.

    Runs peak detection once on the mean spectrum of ``spectral_object`` and derives, for each
    expected peak, an amplitude/frequency-shift/linewidth guess from the detected peak's
    height/position/FWHM, plus a shared background and axis-shift guess. Returned in the
    parameter order the fit models expect: ``[I0, freqShift, LineWidth, ...,  Background,
    axis_shift]``.

    If fewer than ``expected_peaks`` distinct peaks are found (e.g. an overlapping doublet
    showing up as a single broad peak), the remaining slots fall back to evenly spaced
    positions with a width derived from the spectral range, so the estimate degrades
    gracefully instead of raising.

    Parameters
    ----------
    spectral_object : core.SpectralObject
        The data to estimate a p0 for. Its mean spectrum (:attr:`core.SpectralContainer.mean`)
        is used. ``expected_peaks`` is typically known ahead of time, e.g. from the number of
        components found via NMF/VCA.
    expected_peaks : int
        The number of peaks to estimate parameters for (1, 2, or 3).
    model : str, optional
        Lineshape family the guess is for (default ``'dho'``). Only affects the
        length of the shared tail: the ``*_elastic`` families get an extra
        trailing ``elastic_slope`` guess of ``0.0``.

    Returns
    -------
    list[float]
        The estimated p0, ready to pass into :class:`DHO`/:class:`Lorentzian`.
    """
    if expected_peaks not in (1, 2, 3):
        raise ValueError(f"expected_peaks must be 1, 2 or 3, got {expected_peaks}.")

    mean_spectrum = spectral_object.mean
    return _estimate_p0_from_mean(mean_spectrum.spectral_data, mean_spectrum.spectral_axis,
                                  expected_peaks, model=model)


def _estimate_p0_from_mean(intensity, axis, expected_peaks, model="dho"):
    """Array-based core of :func:`estimate_p0` - takes a 1-D mean-intensity array
    and its axis directly (so callers that already have the mean spectrum, e.g.
    :func:`_fit_concurrent_DHO_auto`, don't rebuild a spectral object)."""
    axis = np.asarray(axis)
    intensity = np.ma.filled(intensity, np.nan).astype(float)

    if np.all(np.isnan(intensity)):
        raise ValueError("Cannot estimate p0: mean spectrum is entirely masked/NaN.")

    background = float(np.nanpercentile(intensity, 5))
    intensity = np.where(np.isnan(intensity), background, intensity)
    spacing = float(np.mean(np.diff(axis)))

    # The DHO model is symmetric about zero (a mode at +f also produces a peak at
    # -f), so on a two-sided Brillouin axis one mode = one Stokes-side peak. Only
    # consider peaks at positive shift, past the central elastic region, so
    # ``expected_peaks`` counts modes rather than visible peaks.
    two_sided = axis.min() < 0 < axis.max()
    stokes_threshold = max(spacing, 0.05 * float(np.nanmax(np.abs(axis)))) if two_sided else -np.inf

    candidate_peaks, _ = find_peaks(intensity)
    candidate_peaks = candidate_peaks[axis[candidate_peaks] > stokes_threshold]
    if len(candidate_peaks) > 0:
        prominences = peak_prominences(intensity, candidate_peaks)[0]
        candidate_peaks = candidate_peaks[np.argsort(prominences)[::-1]]

    detected_peaks = list(candidate_peaks[:expected_peaks])
    detected_widths = dict(zip(
        detected_peaks,
        peak_widths(intensity, detected_peaks, rel_height=0.5)[0] if detected_peaks else [],
    ))

    # Pad with evenly spaced fallback positions if too few distinct peaks were detected.
    padded_peaks = []
    if len(detected_peaks) < expected_peaks:
        min_spacing = max(len(intensity) // (2 * expected_peaks), 1)
        remaining = sorted(
            (i for i in range(len(intensity)) if i not in detected_peaks and axis[i] > stokes_threshold),
            key=lambda i: intensity[i], reverse=True,
        )
        for i in remaining:
            if len(detected_peaks) + len(padded_peaks) >= expected_peaks:
                break
            if all(abs(i - p) > min_spacing for p in detected_peaks + padded_peaks):
                padded_peaks.append(i)
        stokes_start = int(np.searchsorted(axis, max(stokes_threshold, axis[0])))
        while len(detected_peaks) + len(padded_peaks) < expected_peaks:
            slot = len(detected_peaks) + len(padded_peaks)
            padded_peaks.append(stokes_start + int((len(intensity) - stokes_start) * (slot + 0.5) / expected_peaks))

    fallback_width_samples = len(intensity) / (4 * expected_peaks)

    p0 = []
    for peak_idx in sorted(detected_peaks + padded_peaks):
        width_samples = detected_widths.get(peak_idx, fallback_width_samples)
        I0 = float(max(intensity[peak_idx] - background, spacing))
        freqShift = float(axis[peak_idx])
        # peak_widths / the fallback give a FWHM in samples; LineWidth is the HWHM.
        LineWidth = float(max(0.5 * width_samples * spacing, spacing))
        p0.extend([I0, freqShift, LineWidth])

    # Shared tail: Background, axis_shift, then a 0.0 for any further tail param
    # (e.g. elastic_slope on the '*_elastic' families).
    n_tail = len(LINESHAPES[model].tail_params) if model in LINESHAPES else 2
    p0.extend([background, 0.0] + [0.0] * (n_tail - 2))

    return p0


# --------------------------------------------------------------------------- #
# Automatic peak-count selection (expected_peaks='auto')
# --------------------------------------------------------------------------- #

def _estimate_noise_variance(y):
    """Robust, model-free noise variance from the spectrum's second differences
    (var of the 2nd difference of white noise is 6 sigma^2). Returns ``None`` if
    there is too little data."""
    y = np.asarray(y, dtype=float)
    d = np.diff(y[np.isfinite(y)], n=2)
    if d.size < 4:
        return None
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = 1.4826 * mad / np.sqrt(6.0)
    return float(sigma ** 2) if sigma > 0 else None


def _information_criterion(rss, n, k, criterion, noise_variance=None):
    """AIC / AICc / BIC, lower is better. With a ``noise_variance`` the fit term
    is a proper chi-square ``RSS / sigma^2`` (so an extra peak that only trims
    RSS by less than the penalty's worth of noise is not selected); without one
    it falls back to the Gaussian ``n * ln(RSS/n)``."""
    fit_term = rss / noise_variance if noise_variance else n * np.log(rss / n)
    if criterion == "aic":
        return fit_term + 2 * k
    if criterion == "aicc":
        return fit_term + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)
    if criterion == "bic":
        return fit_term + k * np.log(n)
    raise ValueError(f"criterion must be 'aic', 'aicc' or 'bic', got {criterion!r}.")


def _dho_peak_heights(popt, n_peaks):
    """Per-peak height ``I0 / (pi * LineWidth)`` - bounded by the data (unlike I0)
    even for a degenerate narrow-spike fit."""
    return [abs(popt[3 * i]) / (np.pi * max(abs(popt[3 * i + 2]), 1e-12)) for i in range(n_peaks)]


def _select_count(entries, min_improvement, min_peak_fraction):
    """From ``{count: (criterion_value, popt)}`` (lower criterion is better), the
    smallest count. A larger count is only accepted if it beats the current best
    by more than ``min_improvement`` *and* every one of its fitted peaks has a
    height of at least ``min_peak_fraction`` of the tallest (so a peak that only
    mops up noise or model mismatch is not counted as a real mode)."""
    counts = sorted(entries)
    best = counts[0]
    for c in counts[1:]:
        score, popt = entries[c]
        if score >= entries[best][0] - min_improvement:
            continue
        heights = _dho_peak_heights(popt, c)
        if min(heights) < min_peak_fraction * max(heights):
            continue
        best = c
    return best


def _auto_bounds(axis, p0, n_peaks, shift_window=3.0):
    """Per-peak bounds for the automatic fit: non-negative amplitude/linewidth,
    and each shift constrained to ``+/- shift_window`` GHz of its ``p0`` value so
    a peak cannot escape to the spectral edge (where it would fit an artefact and
    make the model-selection prefer a spurious extra peak)."""
    axis = np.asarray(axis, dtype=float)
    spacing = float(np.mean(np.abs(np.diff(axis)))) or 1e-3
    max_width = 0.5 * float(np.nanmax(np.abs(axis)))
    lo, hi = [], []
    for i in range(n_peaks):
        s = abs(float(p0[1 + 3 * i]))
        lo += [0.0, max(spacing, s - shift_window), spacing]
        hi += [np.inf, s + shift_window, max_width]
    return lo + [-np.inf, -2.0], hi + [np.inf, 2.0]


def _pad_auto_params(popt, n_peaks, max_peaks):
    """Pad a length-``3n+2`` parameter vector to ``3*max_peaks+2`` with NaN peak
    triplets, keeping the trailing [Background, axis_shift] last."""
    popt = list(np.asarray(popt, dtype=float))
    return popt[:3 * n_peaks] + [np.nan] * (3 * (max_peaks - n_peaks)) + popt[3 * n_peaks:]


def _pad_auto_cov(pcov, n_peaks, max_peaks):
    variables = 3 * max_peaks + 2
    out = np.full((variables, variables), np.nan)
    src = list(range(3 * n_peaks)) + [3 * n_peaks, 3 * n_peaks + 1]
    dst = list(range(3 * n_peaks)) + [variables - 2, variables - 1]
    out[np.ix_(dst, dst)] = np.asarray(pcov, dtype=float)[np.ix_(src, src)]
    return out


def _fitDHO_auto(idx, spectral_axis, intensity_data_slice, p0_by_count, bounds_by_count,
                 fit_funcs, criterion, min_improvement, min_peak_fraction, max_peaks):
    variables = 3 * max_peaks + 2
    nan_params = np.full(variables, np.nan)
    nan_cov = np.full((variables, variables), np.nan)

    y = _prep_pixel_spectrum(intensity_data_slice)
    if y is None:
        return idx, nan_params, nan_cov

    noise_variance = _estimate_noise_variance(y)
    fits, scores = {}, {}
    for n_peaks, p0 in p0_by_count.items():
        lo, hi = bounds_by_count[n_peaks]
        try:
            popt, pcov = curve_fit(fit_funcs[n_peaks], spectral_axis, y,
                                   p0=list(np.clip(p0, lo, hi)), bounds=(lo, hi), nan_policy="omit")
        except (RuntimeError, ValueError):
            continue
        resid = y - fit_funcs[n_peaks](spectral_axis, *popt)
        ok = np.isfinite(resid)
        n = int(ok.sum())
        k = 3 * n_peaks + 2
        rss = float(np.sum(resid[ok] ** 2))
        if n <= k + 1 or not np.isfinite(rss) or rss <= 0:
            continue
        fits[n_peaks] = (popt, pcov)
        scores[n_peaks] = _information_criterion(rss, n, k, criterion, noise_variance)

    if not scores:
        return idx, nan_params, nan_cov
    n_peaks = _select_count({c: (scores[c], fits[c][0]) for c in scores},
                            min_improvement, min_peak_fraction)
    popt, pcov = fits[n_peaks]
    return idx, np.asarray(_pad_auto_params(popt, n_peaks, max_peaks)), _pad_auto_cov(pcov, n_peaks, max_peaks)


def _fit_concurrent_DHO_auto(intensity_data, spectral_axis, *, max_peaks=3, criterion="bic",
                             min_improvement=6.0, min_peak_fraction=0.25, irf=None, max_workers=None):
    if max_peaks not in (2, 3):
        raise ValueError(f"expected_peaks='auto' needs max_peaks in (2, 3), got {max_peaks}.")
    axis = spectral_axis[0]
    data = np.ma.filled(intensity_data, np.nan).astype(float)
    spatial_shape = data.shape[:-1]
    variables = 3 * max_peaks + 2

    base_funcs = model_funcs("dho")
    if irf is not None:
        irf_arr = np.asarray(irf) if not isinstance(irf, (tuple, list, str)) else irf
        if isinstance(irf_arr, np.ndarray) and irf_arr.ndim > 1:
            raise NotImplementedError(
                "expected_peaks='auto' does not support a per-pixel irf yet; pass a shared 1-D "
                "kernel or a parametric ('lorentzian', fwhm) spec, or select the peak count with "
                "fitmodel.estimate_peak_count and fit with a fixed expected_peaks.")
        kernel = irf_kernel(irf_arr, axis)
        fit_funcs = {c: _ConvolvedModel(base_funcs[c], kernel, axis) for c in range(1, max_peaks + 1)}
    else:
        fit_funcs = base_funcs

    mean_intensity = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    p0_by_count, bounds_by_count = {}, {}
    for c in range(1, max_peaks + 1):
        try:
            p0_by_count[c] = _estimate_p0_from_mean(mean_intensity, axis, c)
        except ValueError:
            p0_by_count[c] = [1.0] * (3 * c + 2)
        bounds_by_count[c] = _auto_bounds(axis, p0_by_count[c], c)

    fit_results = np.full(spatial_shape + (variables,), np.nan)
    cov_results = np.full(spatial_shape + (variables, variables), np.nan)
    _logger.info("Starting multiprocessing pool (first dispatch can take a few seconds).")
    with ProcessPoolExecutor(max_workers=_resolve_max_workers(max_workers)) as executor:
        tasks = [executor.submit(_fitDHO_auto, idx, axis, data[idx + (slice(None),)],
                                 p0_by_count, bounds_by_count, fit_funcs, criterion,
                                 min_improvement, min_peak_fraction, max_peaks)
                 for idx in np.ndindex(spatial_shape)]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Fitting Spectral data (auto peak count)"):
            idx, popt, pcov = future.result()
            fit_results[idx] = popt
            cov_results[idx] = pcov
    return fit_results, cov_results


def fitted_peak_count(fit_parameters):
    """Number of peaks actually fitted at each pixel of an ``expected_peaks='auto'``
    result (a NaN peak triplet means that slot was not used)."""
    p = np.asarray(fit_parameters, dtype=float)
    max_peaks = (p.shape[-1] - 2) // 3
    return np.isfinite(p[..., 1:3 * max_peaks:3]).sum(axis=-1)


def estimate_peak_count(spectral_object, *, criterion="bic", candidates=(1, 2, 3),
                        min_improvement=6.0, min_peak_fraction=0.25, bounds=None, return_scores=False):
    """
    Pick the number of Brillouin modes (``expected_peaks``) that best explains the
    *mean* spectrum of ``spectral_object``, by information criterion.

    Much cheaper than ``DHO(expected_peaks='auto')`` (one fit per candidate on a
    single spectrum, not per pixel) and more robust to per-pixel noise - use it
    when the mode count is a property of the sample rather than something that
    varies pixel to pixel.

    This is a heuristic. Information criteria assume the model is otherwise
    correct, so anything the single DHO does *not* capture - a residual elastic /
    Rayleigh wing, an asymmetric or non-Lorentzian real lineshape, an IRF that
    was not deconvolved - reduces the residual when an extra peak is added and
    biases the count upward. On such data, prefer a count from prior knowledge or
    from the variance spectrum (see ``04_classical_analysis.py``), or raise
    ``min_peak_fraction``.

    Parameters
    ----------
    spectral_object : core.SpectralObject
    criterion : {'bic', 'aic', 'aicc'}
        'bic' (default) penalises extra peaks most heavily.
    candidates : iterable of int
        Peak counts to try (each must be 1, 2 or 3).
    min_improvement : float
        An extra peak must improve the criterion by more than this (default 6).
    min_peak_fraction : float
        An extra peak must also have a height of at least this fraction of the
        tallest peak (default 0.25), so a peak that only mops up noise or
        lineshape mismatch is not counted as a real mode.
    bounds : tuple, optional
        Shared ``(lo, hi)`` bounds; if omitted, a default (non-negative
        amplitude/linewidth, shift near each p0) is used per candidate.
    return_scores : bool
        If True, also return ``{count: criterion_value}``.

    Returns
    -------
    int, or (int, dict)
    """
    mean = spectral_object.mean
    axis = np.asarray(mean.spectral_axis)
    y = np.ma.filled(mean.spectral_data, np.nan).astype(float)
    fit_funcs = model_funcs("dho")
    noise_variance = _estimate_noise_variance(y)

    entries, scores = {}, {}
    for c in candidates:
        if c not in (1, 2, 3):
            raise ValueError(f"candidate peak counts must be 1, 2 or 3, got {c}.")
        try:
            p0 = _estimate_p0_from_mean(y, axis, c)
        except ValueError:
            p0 = [1.0] * (3 * c + 2)
        lo, hi = bounds if bounds is not None else _auto_bounds(axis, p0, c)
        try:
            popt, _ = curve_fit(fit_funcs[c], axis, y, p0=list(np.clip(p0, lo, hi)),
                                bounds=(lo, hi), nan_policy="omit")
        except (RuntimeError, ValueError):
            continue
        resid = y - fit_funcs[c](axis, *popt)
        ok = np.isfinite(resid)
        n, k = int(ok.sum()), 3 * c + 2
        rss = float(np.sum(resid[ok] ** 2))
        if n <= k + 1 or not np.isfinite(rss) or rss <= 0:
            continue
        scores[c] = _information_criterion(rss, n, k, criterion, noise_variance)
        entries[c] = (scores[c], popt)

    if not entries:
        raise RuntimeError("no candidate peak count could be fitted to the mean spectrum.")
    best = _select_count(entries, min_improvement, min_peak_fraction)
    return (best, scores) if return_scores else best


_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def irf_kernel(irf, spectral_axis):
    """
    Build a normalised, centred convolution kernel from an instrument response function (IRF).

    Pass the result as the ``irf`` argument of :class:`DHO` to fit the *convolved* model
    ``(intrinsic lineshape) * IRF + background`` to the raw spectrum, so the fitted linewidth
    is the intrinsic Brillouin linewidth with the instrumental broadening accounted for in the
    forward model - rather than deconvolving the data first (which amplifies noise). See the
    ``benchmarks/irf_convolution.py`` comparison.

    Parameters
    ----------
    irf : array-like or tuple
        Either

        - a 1-D array: a measured IRF sample (e.g. the background-subtracted elastic /
          Rayleigh / reference-beam peak), on the *same* channel spacing as ``spectral_axis``.
          It is baseline-subtracted, clipped to non-negative, trimmed to a symmetric window
          about its maximum and normalised.
        - ``('lorentzian', fwhm)`` / ``('gaussian', fwhm)`` / ``('voigt', fwhm_lorentz,
          fwhm_gauss)``: a parametric IRF with the given full width(s) at half maximum, in the
          units of ``spectral_axis`` (GHz).

    spectral_axis : array-like
        The (uniformly spaced) spectral axis the model will be evaluated on. A non-uniform
        axis raises ``ValueError`` - resample first.

    Returns
    -------
    numpy.ndarray
        Odd-length 1-D kernel, summing to 1, centred on its middle sample.
    """
    axis = np.asarray(spectral_axis, dtype=float)
    diffs = np.diff(axis)
    dx = float(np.mean(diffs))
    if dx <= 0 or not np.allclose(diffs, dx, rtol=1e-3, atol=0.0):
        raise ValueError(
            "irf_kernel needs a uniformly spaced spectral_axis; got spacing varying by "
            f"{float(np.ptp(diffs)):.3g}. Resample the spectrum onto a regular grid first.")

    if isinstance(irf, (tuple, list)) and len(irf) >= 2 and isinstance(irf[0], str):
        shape = irf[0].lower()
        if shape in ("lorentzian", "gaussian"):
            fwhm = float(irf[1])
            if fwhm <= 0:
                raise ValueError(f"irf fwhm must be positive, got {fwhm}.")
            # Lorentzian tails are heavy - truncating at 5x FWHM drops enough area to
            # bias the renormalised kernel narrow; go wider. A Gaussian is essentially
            # zero by 5 sigma.
            support = 12.0 if shape == "lorentzian" else 5.0
            half_n = max(3, int(np.ceil(support * fwhm / dx)))
            grid = np.arange(-half_n, half_n + 1) * dx
            if shape == "lorentzian":
                kernel = 1.0 / (1.0 + (grid / (fwhm / 2.0)) ** 2)
            else:
                kernel = np.exp(-0.5 * (grid / (fwhm / _FWHM_TO_SIGMA)) ** 2)
        elif shape == "voigt":
            from scipy.special import voigt_profile
            fwhm_l, fwhm_g = float(irf[1]), float(irf[2])
            if fwhm_l <= 0 or fwhm_g <= 0:
                raise ValueError(f"voigt irf fwhms must be positive, got ({fwhm_l}, {fwhm_g}).")
            half_n = max(3, int(np.ceil(5.0 * (fwhm_l + fwhm_g) / dx)))
            grid = np.arange(-half_n, half_n + 1) * dx
            kernel = voigt_profile(grid, fwhm_g / _FWHM_TO_SIGMA, fwhm_l / 2.0)
        else:
            raise ValueError(
                f"parametric irf shape must be 'lorentzian', 'gaussian' or 'voigt', got {irf[0]!r}.")
    else:
        kernel = np.asarray(irf, dtype=float).ravel()
        if kernel.size < 3:
            raise ValueError(f"a measured irf needs at least 3 samples, got {kernel.size}.")
        kernel = np.nan_to_num(kernel, nan=0.0)
        nz = np.flatnonzero(kernel > kernel.min())
        if nz.size < 3:
            raise ValueError("measured irf has no usable samples.")
        # Restrict to the detected support (a detected elastic-peak region is
        # otherwise mostly zeros spanning the whole spectrum).
        kernel = kernel[nz[0]:nz[-1] + 1]
        # Subtract a *linear* baseline through the support's two ends: a detected
        # elastic peak sits on the sloping inner wing of the Brillouin doublet,
        # not a flat pedestal, and leaving that in makes the kernel too broad
        # (the fit then over-corrects the linewidth).
        idx = np.arange(kernel.size, dtype=float)
        e = max(1, kernel.size // 8)
        ends_x = np.concatenate([idx[:e], idx[-e:]])
        ends_y = np.concatenate([kernel[:e], kernel[-e:]])
        slope, intercept = np.polyfit(ends_x, ends_y, 1)
        kernel = np.clip(kernel - (slope * idx + intercept), 0.0, None)
        # Drop channels below a small fraction of the peak (residual baseline noise
        # in the shoulders - kept low so genuine Lorentzian wings survive), then
        # trim to a symmetric window about the peak so it is centred.
        kernel[kernel < 0.005 * kernel.max()] = 0.0
        nz = np.flatnonzero(kernel > 0)
        if nz.size < 3:
            raise ValueError("measured irf has no usable samples after baseline subtraction.")
        kernel = kernel[nz[0]:nz[-1] + 1]
        peak = int(np.argmax(kernel))
        reach = min(peak, kernel.size - 1 - peak)
        if reach < 1:
            raise ValueError("measured irf peak is at the very edge of its support; give a wider window.")
        kernel = kernel[peak - reach:peak + reach + 1]

    total = kernel.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("irf kernel has non-positive total; check the input.")
    return kernel / total


class _ConvolvedModel:
    """Picklable wrapper turning a bare lineshape ``f(x, *peak_params, Background, axis_shift)``
    into ``conv(f(., *peak_params, 0, axis_shift), kernel) + Background``, evaluated on a stored
    uniform grid and interpolated to whatever sample points ``curve_fit`` passes (which may be
    a subset of the grid, e.g. with the elastic-peak channels dropped).

    ``Background`` is passed as 0 into the intrinsic lineshape and added once after the
    convolution, so it is counted exactly once - matching the plain (non-convolved) models."""

    def __init__(self, base_func, kernel, full_axis):
        self.base_func = base_func
        self.kernel = np.asarray(kernel, dtype=float)
        self.full_axis = np.asarray(full_axis, dtype=float)

    def __call__(self, x, *params):
        x = np.asarray(x, dtype=float)
        background = params[-2]
        peaks = self.base_func(self.full_axis, *params[:-2], 0.0, params[-1])
        n = self.kernel.size
        padded = np.pad(peaks, n, mode="edge")
        convolved = fftconvolve(padded, self.kernel, mode="same")[n:-n]
        model = convolved + background
        if x.shape == self.full_axis.shape and np.array_equal(x, self.full_axis):
            return model
        return np.interp(x, self.full_axis, model)


class DHO(FitStep):
    """
    Fit of one or multiple damped-harmonic-oscillator (DHO) modes::

        I0 * 4 * LineWidth * freqShift**2
            / (np.pi * ((xs**2 - freqShift**2)**2 + 4 * (LineWidth * xs)**2)) + Background

    with ``xs = x - axis_shift``. ``LineWidth`` is the **HWHM** (half width at
    half maximum, Gamma / 2; the FWHM is ``2 * LineWidth``); the peak height is
    ``I0 / (pi * LineWidth)``; ``Background`` is a single shared additive
    constant; ``axis_shift`` (formerly ``Asymmetry``) rigidly shifts the
    frequency axis. One mode is a symmetric Stokes/anti-Stokes doublet, so
    ``expected_peaks`` counts modes. See
    :mod:`brillouinpy.analysis.fit.lineshapes` for the full parameter layout and for
    registering additional lineshape families.

    Parameters
    ----------
    expected_peaks : int or ``'auto'``
        The number of Brillouin modes (symmetric peak pairs) to fit. ``'auto'``
        fits 1 .. ``max_peaks`` per pixel and keeps whichever count minimises the
        information ``criterion`` (see below); ``p0``/``bounds`` are then derived
        automatically (from :func:`estimate_p0` on the mean spectrum) and any
        values passed are ignored. The returned parameter array is padded to
        ``3 * max_peaks + 2``, with a NaN ``[I0, freqShift, LineWidth]`` triplet
        for each unused slot - :func:`fitted_peak_count` reads the per-pixel
        count back. Per-pixel model selection is noisy; :func:`estimate_peak_count`
        (one number for the whole image) is the cheaper, steadier option.
    p0 : list[floats] or None
        The initial guess (ignored when ``expected_peaks='auto'``).
    bounds : list[list[floats], list[floats]] or None
        Parameter bounds (ignored when ``expected_peaks='auto'``).
    max_peaks : int, optional
        For ``expected_peaks='auto'``: the largest peak count to consider (2 or 3;
        default 3).
    criterion : {'bic', 'aic', 'aicc'}, optional
        For ``expected_peaks='auto'``: the information criterion to minimise.
        ``'bic'`` (default) penalises extra peaks most heavily.
    min_improvement : float, optional
        For ``expected_peaks='auto'``: an extra peak is accepted only if it
        improves ``criterion`` by more than this (default 6) *and* is at least
        ``min_peak_fraction`` as tall as the biggest peak. Raise to be more
        conservative about adding modes.
    min_peak_fraction : float, optional
        For ``expected_peaks='auto'``: the minimum height (relative to the
        tallest peak) an added peak must have to count as a real mode rather
        than noise / lineshape mop-up. Default 0.25 - raise it (e.g. to 0.4) if a
        known single mode is still being split by residual elastic scattering or
        lineshape mismatch on real data.
    irf : array-like, tuple, ``'auto'`` or None, optional
        Instrument response function. If given, the model fitted to each spectrum is the
        intrinsic DHO lineshape *convolved with the IRF* plus a background, so the fitted
        ``LineWidth`` is the intrinsic Brillouin linewidth with the instrumental broadening
        modelled explicitly - instead of deconvolving the data first (which amplifies noise;
        contrast :class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF`).

        - ``'auto'``: use the fitted object's ``instrument_response_function`` attribute
          (set by :class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF` /
          :class:`~brillouinpy.preprocessing.misc.IRF_Remover` with ``store_irf=True``, or
          by :func:`~brillouinpy.preprocessing.misc.assign_irf`). A per-pixel
          ``(*spatial, B)`` IRF gives each pixel its own kernel; a shared 1-D one is used
          everywhere. Falls back to a plain fit (with a warning) if the object carries none.
        - a 1-D array: a measured IRF sample (e.g. the elastic peak) on the spectrum's
          channel spacing.
        - ``('lorentzian'|'gaussian', fwhm)`` / ``('voigt', fwhm_l, fwhm_g)``: a parametric
          IRF, widths in GHz.

        The elastic-peak channels should still be blanked to ``NaN`` (e.g. via
        :class:`~brillouinpy.preprocessing.misc.IRF_Remover`) so they are excluded from the
        residual; the convolved model itself is evaluated on the full axis. ``None`` (default)
        keeps the plain, unconvolved fit.
    max_workers : int or None, optional
        Number of worker processes for the pixel-wise fit. ``None`` (default) uses
        a quarter of the visible CPUs; set it explicitly on an HPC node or in a
        CPU-limited container.
    loss : {'squares', 'poisson'}, optional
        The fit objective.

        - ``'squares'`` (default): unweighted least squares (``scipy.optimize.curve_fit``).
          The returned covariance is ``curve_fit``'s, scaled by the residual variance.
        - ``'poisson'``: Poisson maximum likelihood, i.e. minimise
          ``sum(mu_k - y_k * ln(mu_k))`` over the finite channels, started from the
          least-squares solution. On raw photon counts this weights every channel by
          its actual statistics (the low-count wings carry most of the linewidth
          information that least squares under-weights) and reaches the Cramér-Rao
          bound; on TFP water spectra it cuts the BFS scatter by ~14 % and the
          linewidth scatter by ~30 %. The returned covariance is the inverse Fisher
          information at the optimum (absolute, not rescaled by the residuals). A
          negative lower bound on ``Background`` is clipped to 0 (the expected count
          must stay positive). Only for raw counts (or sums of them) - *not* after
          deconvolution, normalisation, background subtraction or averaging; a
          warning flags non-integer or negative input. Not supported with
          ``expected_peaks='auto'``.

    """
    def __init__(self, *, expected_peaks, p0=None, bounds=None, irf=None, max_workers=None,
                 elastic=False, max_peaks=3, criterion='bic', min_improvement=6.0,
                 min_peak_fraction=0.25, loss='squares'):
        _check_loss(loss)
        if isinstance(expected_peaks, str) and expected_peaks == 'auto':
            if elastic:
                raise NotImplementedError("expected_peaks='auto' does not support elastic=True.")
            if loss != 'squares':
                raise NotImplementedError(
                    "expected_peaks='auto' does not support loss='poisson' yet (its information "
                    "criterion is built on least-squares residuals); pick the peak count with "
                    "estimate_peak_count and fit with a fixed expected_peaks.")
            super().__init__(_fit_concurrent_DHO_auto, max_peaks=max_peaks, criterion=criterion,
                             min_improvement=min_improvement, min_peak_fraction=min_peak_fraction,
                             irf=irf, max_workers=max_workers)
        else:
            super().__init__(_fit_concurrent, expected_peaks=expected_peaks, p0=p0, bounds=bounds,
                             irf=irf, model="dho_elastic" if elastic else "dho",
                             max_workers=max_workers, loss=loss)


class Lorentzian(FitStep):
    """
    Fit of one or multiple Lorentzian doublets: for each mode, two Lorentzians of
    HWHM ``LineWidth`` centred at ``axis_shift +/- freqShift`` (peak height
    ``I0 / (pi * LineWidth)`` each), plus one shared ``Background``. Same
    parameter conventions, flat layout, ``irf`` handling and return values as
    :class:`DHO` (see :mod:`brillouinpy.analysis.fit.lineshapes` and the ``DHO`` docstring).

    Parameters
    ----------
    expected_peaks : int
        The number of modes (symmetric peak pairs) to fit.
    p0 : list[float] or None
        Initial guess, length ``expected_peaks * 3 + 2``. ``None`` starts from ones
        (see :func:`estimate_p0` for a better guess).
    bounds : tuple or None
        ``(lo, hi)`` parameter bounds passed to ``scipy.optimize.curve_fit``.
    irf : array-like, tuple, ``'auto'`` or None, optional
        Instrument response function - as for :class:`DHO`.
    elastic : bool, optional
        If True, fit ``lorentzian_elastic``: the shared baseline gains a linear
        ``elastic_slope`` term (last parameter) that absorbs the sloping residual
        wing of an un-blanked elastic peak. Incompatible with ``irf``.
    max_workers : int or None, optional
        Worker processes for the pixel-wise fit (see :class:`DHO`).
    loss : {'squares', 'poisson'}, optional
        Fit objective - as for :class:`DHO`.
    """
    def __init__(self, *, expected_peaks, p0=None, bounds=None, irf=None, elastic=False,
                 max_workers=None, loss='squares'):
        _check_loss(loss)
        super().__init__(_fit_concurrent, expected_peaks=expected_peaks, p0=p0, bounds=bounds,
                         irf=irf, model="lorentzian_elastic" if elastic else "lorentzian",
                         max_workers=max_workers, loss=loss)


class Gaussian(FitStep):
    """
    Fit of one or multiple Gaussian doublets: for each mode, two Gaussians of
    HWHM ``LineWidth`` centred at ``axis_shift +/- freqShift`` (peak height
    ``I0 / (pi * LineWidth)`` each), plus one shared ``Background``. Same
    parameter conventions, flat layout, ``irf`` handling and return values as
    :class:`DHO` / :class:`Lorentzian`.

    Parameters
    ----------
    expected_peaks : int
        The number of modes (symmetric peak pairs) to fit.
    p0 : list[float] or None
        Initial guess, length ``expected_peaks * 3 + 2`` (one more when
        ``elastic=True``). ``None`` starts from ones.
    bounds : tuple or None
        ``(lo, hi)`` parameter bounds passed to ``scipy.optimize.curve_fit``.
    irf : array-like, tuple, ``'auto'`` or None, optional
        Instrument response function - as for :class:`DHO`.
    elastic : bool, optional
        Fit ``gaussian_elastic`` (extra trailing ``elastic_slope``). Incompatible
        with ``irf``.
    max_workers : int or None, optional
        Worker processes for the pixel-wise fit (see :class:`DHO`).
    loss : {'squares', 'poisson'}, optional
        Fit objective - as for :class:`DHO`.
    """
    def __init__(self, *, expected_peaks, p0=None, bounds=None, irf=None, elastic=False,
                 max_workers=None, loss='squares'):
        _check_loss(loss)
        super().__init__(_fit_concurrent, expected_peaks=expected_peaks, p0=p0, bounds=bounds,
                         irf=irf, model="gaussian_elastic" if elastic else "gaussian",
                         max_workers=max_workers, loss=loss)


class PeakFit(FitStep):
    """
    Fit of one or multiple modes of an arbitrary **registered** lineshape family
    (see :func:`~brillouinpy.analysis.fit.lineshapes.register_lineshape`). Use this
    for lineshapes you add yourself; :class:`DHO` / :class:`Lorentzian` /
    :class:`Gaussian` are the convenience wrappers for the built-ins.

    Parameters
    ----------
    model : str
        Name of a family in :data:`brillouinpy.analysis.fit.lineshapes.LINESHAPES`.
    expected_peaks : int
        Number of modes (symmetric peak pairs) to fit.
    p0, bounds, irf, max_workers, loss
        As for :class:`DHO`. ``irf`` requires the family's baseline to be the
        standard ``(Background, axis_shift)``.
    """
    def __init__(self, *, model, expected_peaks, p0=None, bounds=None, irf=None, max_workers=None,
                 loss='squares'):
        _check_loss(loss)
        super().__init__(_fit_concurrent, expected_peaks=expected_peaks, p0=p0, bounds=bounds,
                         irf=irf, model=model, max_workers=max_workers, loss=loss)


# --------------------------------------------------------------------------- #
# Poisson maximum likelihood (loss='poisson')
# --------------------------------------------------------------------------- #

_LOSSES = ("squares", "poisson")

#: Floor on the expected count in the Poisson likelihood (``ln(mu)`` needs ``mu > 0``).
_MIN_EXPECTED_COUNT = 1e-12


def _check_loss(loss):
    if loss not in _LOSSES:
        raise ValueError(f"loss must be 'squares' or 'poisson', got {loss!r}.")


def _numeric_jacobian(fit_func, x, p):
    """Central-difference Jacobian ``d fit_func(x, *p) / dp``, shape ``(len(p), len(x))``,
    with each step relative to the parameter's magnitude."""
    p = np.asarray(p, dtype=float)
    jac = np.empty((p.size, len(x)))
    for i, value in enumerate(p):
        h = 1e-5 * max(abs(value), 1.0)
        up, down = p.copy(), p.copy()
        up[i] += h
        down[i] -= h
        jac[i] = (fit_func(x, *up) - fit_func(x, *down)) / (2 * h)
    return jac


def _poisson_refine(fit_func, x, y, p_start, bounds):
    """Poisson maximum-likelihood fit of ``fit_func`` to the finite channels of the
    count spectrum ``y``, started from ``p_start`` (the least-squares solution).
    Returns ``(popt, pcov)`` with ``pcov`` the inverse Fisher information
    ``(J diag(1/mu) J^T)^-1`` at the optimum."""
    finite = np.isfinite(y)
    x_f, y_f = np.asarray(x, dtype=float)[finite], y[finite]
    p_start = np.asarray(p_start, dtype=float)
    lo = np.broadcast_to(np.asarray(bounds[0], dtype=float), p_start.shape)
    hi = np.broadcast_to(np.asarray(bounds[1], dtype=float), p_start.shape)
    p_start = np.clip(p_start, lo, hi)

    # Subtracting the saturated-model term makes the objective half the Poisson
    # deviance (>= 0, ~ n_channels / 2 at the optimum) instead of a large
    # offset, which keeps L-BFGS-B's relative stopping criterion meaningful.
    saturated = float(np.sum(y_f - xlogy(y_f, y_f)))

    def half_deviance(p):
        mu = np.maximum(fit_func(x_f, *p), _MIN_EXPECTED_COUNT)
        return float(np.sum(mu - y_f * np.log(mu))) - saturated, mu

    def objective(p):
        value, mu = half_deviance(p)
        return value, _numeric_jacobian(fit_func, x_f, p) @ (1.0 - y_f / mu)

    res = minimize(objective, p_start, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)))
    # L-BFGS-B can report a line-search "failure" at an already converged point;
    # take its result whenever it is at least as good as the start.
    popt = res.x if np.isfinite(res.fun) and res.fun <= half_deviance(p_start)[0] else p_start

    mu = np.maximum(fit_func(x_f, *popt), _MIN_EXPECTED_COUNT)
    jac = _numeric_jacobian(fit_func, x_f, popt)
    pcov = np.linalg.pinv(jac @ (jac / mu).T)
    return popt, pcov


def _poisson_bounds(bounds, p0, n_params, background_index):
    """Bounds / p0 for a Poisson fit: the ``Background`` lower bound (and start
    value) clipped to >= 0, since the likelihood needs a positive expected count
    in every channel (and a negative one is unphysical anyway)."""
    bounds = (0, np.inf) if bounds is None else bounds
    lo = np.array(np.broadcast_to(np.asarray(bounds[0], dtype=float), n_params))
    hi = np.array(np.broadcast_to(np.asarray(bounds[1], dtype=float), n_params))
    if lo[background_index] < 0:
        warnings.warn(
            f"loss='poisson' needs a non-negative Background; clipping its lower bound "
            f"{lo[background_index]:g} to 0.", stacklevel=3)
        lo[background_index] = 0.0
    if p0 is not None and p0[background_index] < 0:
        p0 = list(p0)
        p0[background_index] = 0.0
    return (lo, hi), p0


def _warn_if_not_counts(intensity_data):
    """Warn when ``loss='poisson'`` is used on data that cannot be raw photon counts."""
    y = np.ma.filled(intensity_data, np.nan)
    y = y[np.isfinite(y)]
    if y.size and (np.any(y < 0) or np.max(np.abs(y - np.round(y))) > 1e-6):
        warnings.warn(
            "loss='poisson' expects raw photon counts (non-negative integers), but the data "
            "contain negative or non-integer values. After deconvolution, normalisation, "
            "background subtraction or averaging the noise is no longer Poisson - use "
            "loss='squares' there.", stacklevel=3)


def _fit_pixel(idx, spectral_axis, intensity_data_slice, n_params, fit_func, p0, bounds,
               loss="squares"):
    """Fit one pixel's spectrum with ``fit_func``. Returns ``(idx, popt, pcov)``;
    an all-masked / empty pixel or a non-converging fit yields NaN arrays of the
    right shape. ``loss='poisson'`` refines the least-squares result to the
    Poisson maximum-likelihood estimate (:func:`_poisson_refine`)."""
    nan_params = np.full(n_params, np.nan)
    nan_cov = np.full((n_params, n_params), np.nan)

    y = _prep_pixel_spectrum(intensity_data_slice)
    if y is None:
        return idx, nan_params, nan_cov

    p0 = [1.0] * n_params if p0 is None else p0
    bounds = (0, np.inf) if bounds is None else bounds

    try:
        popt, pcov = curve_fit(fit_func, spectral_axis, y, p0=p0, bounds=bounds, nan_policy="omit")
    except RuntimeError:
        _logger.debug("fit did not converge for pixel %s", idx)
        return idx, nan_params, nan_cov
    if loss == "poisson":
        try:
            popt, pcov = _poisson_refine(fit_func, spectral_axis, y, popt, bounds)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            _logger.debug("Poisson refinement failed for pixel %s", idx)
            return idx, nan_params, nan_cov
    return idx, popt, pcov


def _fit_concurrent(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None,
                    irf=None, *, model="dho", max_workers=None, loss="squares"):
    """Pixel-wise fit of a fixed number of modes across a stacked spectral image;
    backs :class:`DHO`, :class:`Lorentzian`, :class:`Gaussian` and :class:`PeakFit`.

    ``model`` picks the lineshape family (``'dho'`` / ``'lorentzian'``). ``irf`` is
    ``None``, a parametric spec / 1-D kernel (one convolved model for every pixel)
    or a per-pixel ``(*spatial, B)`` kernel array (a pixel with no usable kernel
    falls back to the plain model). ``loss`` is ``'squares'`` (least squares) or
    ``'poisson'`` (Poisson maximum likelihood, see :class:`DHO`)."""
    _check_loss(loss)
    fit_funcs = model_funcs(model)
    if expected_peaks not in fit_funcs:
        raise ValueError(f"expected_peaks must be 1..{MAX_EXPECTED_PEAKS}, got {expected_peaks!r}.")
    n_tail = len(LINESHAPES[model].tail_params)
    if irf is not None and n_tail != 2:
        raise NotImplementedError(
            f"irf convolution is not supported for the '{model}' lineshape (it has a "
            f"non-standard baseline: {list(LINESHAPES[model].tail_params)}). Blank the "
            f"elastic-peak channels and fit the non-elastic '{model.replace('_elastic', '')}' "
            f"family with irf instead.")
    n_params = int(expected_peaks * 3 + n_tail)
    spatial_shape = intensity_data.shape[:-1]
    fit_results = np.full(spatial_shape + (n_params,), np.nan)
    cov_results = np.full(spatial_shape + (n_params, n_params), np.nan)

    base_func = fit_funcs[expected_peaks]
    spectral_axis = spectral_axis[0]

    if p0 is not None and len(p0) != n_params:
        warnings.warn(
            f"p0 has {len(p0)} entries but the {expected_peaks}-mode '{model}' model needs "
            f"{n_params}; ignoring it and starting curve_fit from ones.", stacklevel=2)
        p0 = None

    if loss == "poisson":
        # tail_params[0] is the additive background by the lineshape-registry contract
        bounds, p0 = _poisson_bounds(bounds, p0, n_params, background_index=3 * expected_peaks)
        _warn_if_not_counts(intensity_data)

    irf_arr = np.asarray(irf) if (irf is not None and not isinstance(irf, (tuple, list, str))) else irf
    per_pixel_irf = isinstance(irf_arr, np.ndarray) and irf_arr.ndim > 1
    if irf is None:
        shared_fit_func = base_func
    elif per_pixel_irf:
        shared_fit_func = None
    else:
        shared_fit_func = _ConvolvedModel(base_func, irf_kernel(irf_arr, spectral_axis), spectral_axis)

    def _pixel_fit_func(idx):
        if not per_pixel_irf:
            return shared_fit_func
        try:
            return _ConvolvedModel(base_func, irf_kernel(irf_arr[idx], spectral_axis), spectral_axis)
        except ValueError:
            return base_func  # no usable IRF for this pixel

    _logger.info("Starting multiprocessing pool (first dispatch can take a few seconds).")
    with ProcessPoolExecutor(max_workers=_resolve_max_workers(max_workers)) as executor:
        tasks = [
            executor.submit(_fit_pixel, idx, spectral_axis, intensity_data[idx + (slice(None),)],
                            n_params, _pixel_fit_func(idx), p0, bounds, loss)
            for idx in np.ndindex(spatial_shape)
        ]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Fitting spectral data"):
            idx, popt, pcov = future.result()
            fit_results[idx] = popt
            cov_results[idx] = pcov

    return fit_results, cov_results
