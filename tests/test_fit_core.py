import numpy as np
import pytest

from brillouinpy.analysis.fit.core import (
    _DHO_1,
    _DHO_2,
    _DHO_3,
    _Lorentzian_1,
    DHO,
    estimate_p0,
    estimate_peak_count,
    fitted_peak_count,
    irf_kernel,
)
from brillouinpy.core import Spectrum, SpectralImage


def test_dho1_peaks_near_freq_shift():
    # The model is symmetric about axis_shift, so it peaks at both +freqShift and -freqShift;
    # restrict to the positive half to check the location of one of them.
    x = np.linspace(0, 20, 2001)
    y = _DHO_1(x, I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, axis_shift=0.0)

    assert x[np.argmax(y)] == pytest.approx(5.0, abs=0.05)


def test_dho1_linewidth_is_hwhm():
    # LineWidth is the half width at half maximum: the full width at half the
    # peak height above background is 2 * LineWidth.
    x = np.linspace(0, 20, 400001)
    peak = _DHO_1(np.array([6.0]), I0=2.0, freqShift=6.0, LineWidth=0.5, Background=1.0, axis_shift=0.0)[0]
    y = _DHO_1(x, I0=2.0, freqShift=6.0, LineWidth=0.5, Background=1.0, axis_shift=0.0)
    half = y >= 1.0 + (peak - 1.0) / 2.0
    fwhm = x[half][-1] - x[half][0]
    assert fwhm == pytest.approx(2 * 0.5, abs=0.02)


def test_dho1_approaches_background_far_from_peak():
    y_far = _DHO_1(np.array([1000.0]), I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, axis_shift=0.0)

    assert y_far[0] == pytest.approx(1.0, abs=1e-3)


def test_dho2_shares_a_single_background():
    x = np.linspace(-20, 20, 50)
    params = dict(I0=1.0, freqShift=5.0, LineWidth=0.3, I02=2.0, freqShift2=8.0, LineWidth2=0.5,
                  Background=0.5, axis_shift=0.1)

    combined = _DHO_2(x, **params)
    expected = (
        _DHO_1(x, params["I0"], params["freqShift"], params["LineWidth"],
               params["Background"], params["axis_shift"])
        + _DHO_1(x, params["I02"], params["freqShift2"], params["LineWidth2"],
                 0.0, params["axis_shift"])
    )

    assert np.allclose(combined, expected)


def test_dho3_is_dho2_plus_one_backgroundless_mode():
    x = np.linspace(-20, 20, 50)
    params = dict(I0=1.0, freqShift=5.0, LineWidth=0.3, I02=2.0, freqShift2=8.0, LineWidth2=0.5,
                  I03=0.5, freqShift3=12.0, LineWidth3=0.2, Background=0.5, axis_shift=0.0)

    combined = _DHO_3(x, **params)
    expected = (
        _DHO_2(x, params["I0"], params["freqShift"], params["LineWidth"], params["I02"], params["freqShift2"],
               params["LineWidth2"], params["Background"], params["axis_shift"])
        + _DHO_1(x, params["I03"], params["freqShift3"], params["LineWidth3"],
                 0.0, params["axis_shift"])
    )

    assert np.allclose(combined, expected)


def test_lorentzian1_peaks_near_freq_shift():
    # Same symmetry consideration as the DHO model above.
    x = np.linspace(0, 20, 2001)
    y = _Lorentzian_1(x, I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, axis_shift=0.0)

    assert x[np.argmax(y)] == pytest.approx(5.0, abs=0.05)


def test_lorentzian1_linewidth_is_hwhm():
    x = np.linspace(0, 20, 400001)
    peak = _Lorentzian_1(np.array([6.0]), I0=2.0, freqShift=6.0, LineWidth=0.5, Background=1.0, axis_shift=0.0)[0]
    y = _Lorentzian_1(x, I0=2.0, freqShift=6.0, LineWidth=0.5, Background=1.0, axis_shift=0.0)
    half = y >= 1.0 + (peak - 1.0) / 2.0
    fwhm = x[half][-1] - x[half][0]
    assert fwhm == pytest.approx(2 * 0.5, abs=0.03)


def test_dho_fit_recovers_known_parameters():
    true_params = dict(I0=5.0, freqShift=6.0, LineWidth=0.4, Background=1.0, axis_shift=0.0)
    axis = np.linspace(-20, 20, 400)
    intensity = _DHO_1(axis, **true_params)

    spectrum = Spectrum(intensity, axis)

    fit = DHO(expected_peaks=1, p0=[1, 5, 1, 0.5, 0], bounds=(0, np.inf))
    popt, _ = fit.apply(spectrum)

    assert popt[0] == pytest.approx(true_params["I0"], rel=0.05)
    assert popt[1] == pytest.approx(true_params["freqShift"], rel=0.05)
    assert popt[2] == pytest.approx(true_params["LineWidth"], rel=0.05)
    assert popt[3] == pytest.approx(true_params["Background"], abs=0.1)


def test_estimate_p0_single_peak():
    # Restricted to the positive half: the DHO model is symmetric about axis_shift and would
    # otherwise also peak at -freqShift (see test_dho1_peaks_near_freq_shift).
    true_params = dict(I0=5.0, freqShift=6.0, LineWidth=0.4, Background=1.0, axis_shift=0.0)
    axis = np.linspace(0, 20, 400)
    intensity = _DHO_1(axis, **true_params)

    spectrum = Spectrum(intensity, axis)
    p0 = estimate_p0(spectrum, expected_peaks=1)

    assert len(p0) == 5
    assert p0[1] == pytest.approx(true_params["freqShift"], abs=0.2)
    assert p0[3] == pytest.approx(true_params["Background"], abs=0.5)


def test_estimate_p0_two_peaks_recover_positions():
    true_params = dict(I0=1.0, freqShift=5.0, LineWidth=0.3, I02=2.0, freqShift2=12.0, LineWidth2=0.5,
                        Background=0.5, axis_shift=0.0)
    axis = np.linspace(0, 20, 800)
    intensity = _DHO_2(axis, **true_params)

    spectrum = Spectrum(intensity, axis)
    p0 = estimate_p0(spectrum, expected_peaks=2)

    assert len(p0) == 8
    # Peaks are returned ordered by ascending frequency shift.
    assert p0[1] == pytest.approx(true_params["freqShift"], abs=0.2)
    assert p0[4] == pytest.approx(true_params["freqShift2"], abs=0.2)


def test_estimate_p0_falls_back_when_peaks_overlap():
    # A single, unresolved broad peak: expecting 2 peaks must not raise.
    axis = np.linspace(0, 20, 400)
    intensity = _DHO_1(axis, I0=5.0, freqShift=10.0, LineWidth=3.0, Background=1.0, axis_shift=0.0)

    spectrum = Spectrum(intensity, axis)
    p0 = estimate_p0(spectrum, expected_peaks=2)

    assert len(p0) == 8
    assert all(np.isfinite(p0))


def test_estimate_p0_rejects_invalid_expected_peaks():
    axis = np.linspace(-20, 20, 100)
    spectrum = Spectrum(np.ones_like(axis), axis)

    with pytest.raises(ValueError):
        estimate_p0(spectrum, expected_peaks=4)


# --- IRF convolution -------------------------------------------------------- #

def test_irf_kernel_parametric_and_measured_are_normalised_and_centred():
    axis = np.linspace(-15, 15, 400)

    for spec in (("lorentzian", 0.5), ("gaussian", 0.5), ("voigt", 0.3, 0.4)):
        k = irf_kernel(spec, axis)
        assert k.size % 2 == 1
        assert k.sum() == pytest.approx(1.0)
        assert np.argmax(k) == k.size // 2  # symmetric, centred

    # a measured (noisy, off-centre, sloping-baseline) sample still normalises/centres
    grid = np.arange(-20, 21) * (axis[1] - axis[0])
    measured = 1.0 / (1.0 + (grid / 0.25) ** 2) + 0.05 + 0.001 * grid
    k = irf_kernel(measured, axis)
    assert k.sum() == pytest.approx(1.0)
    assert np.argmax(k) == k.size // 2


def test_irf_kernel_rejects_non_uniform_axis():
    axis = np.concatenate([np.linspace(-10, 0, 50), np.linspace(0.5, 10, 50)])
    with pytest.raises(ValueError):
        irf_kernel(("gaussian", 0.5), axis)


def test_convolved_dho_fit_recovers_intrinsic_linewidth():
    # A DHO spectrum broadened by a known IRF: the plain fit sees the broadened width,
    # the irf= fit recovers the intrinsic one.
    rng = np.random.default_rng(1)
    axis = np.linspace(-15, 15, 400)
    true_shift, true_lw, amp, bg, irf_fwhm = 6.0, 0.6, 1.0, 0.05, 0.5

    intrinsic = _DHO_1(axis, amp, true_shift, true_lw, 0.0, 0.0)
    k = irf_kernel(("lorentzian", irf_fwhm), axis)
    n = k.size
    broadened = np.convolve(np.pad(intrinsic, n, mode="edge"), k, mode="same")[n:-n] + bg

    scale = 400.0 / broadened.max()
    noisy = rng.poisson(np.clip(np.tile(broadened, (5, 5, 1)) * scale, 0, None)) / scale
    image = SpectralImage(noisy.astype(float), axis)

    p0 = [amp, true_shift, 1.0, bg, 0.0]
    bounds = ([0, 3, 0.05, -1, -1], [10, 14, 5, 2, 1])

    plain, _ = DHO(expected_peaks=1, p0=p0, bounds=bounds).apply(image)
    conv, _ = DHO(expected_peaks=1, p0=p0, bounds=bounds, irf=("lorentzian", irf_fwhm)).apply(image)

    plain_lw = float(np.nanmedian(plain[..., 2]))
    conv_lw = float(np.nanmedian(conv[..., 2]))

    assert plain_lw > true_lw + 0.1                       # plain fit is broadened
    assert conv_lw == pytest.approx(true_lw, abs=0.08)    # convolution fit recovers it
    assert float(np.nanmedian(conv[..., 1])) == pytest.approx(true_shift, abs=0.05)


def test_dho_irf_auto_reads_container_attribute():
    import warnings

    rng = np.random.default_rng(2)
    axis = np.linspace(-15, 15, 400)
    true_shift, true_lw, amp, bg, irf_fwhm = 6.0, 0.6, 1.0, 0.05, 0.5

    intrinsic = _DHO_1(axis, amp, true_shift, true_lw, 0.0, 0.0)
    k = irf_kernel(("lorentzian", irf_fwhm), axis)
    n = k.size
    broadened = np.convolve(np.pad(intrinsic, n, mode="edge"), k, mode="same")[n:-n] + bg
    scale = 400.0 / broadened.max()
    noisy = (rng.poisson(np.clip(np.tile(broadened, (4, 4, 1)) * scale, 0, None)) / scale).astype(float)

    p0 = [amp, true_shift, 1.0, bg, 0.0]
    bounds = ([0, 3, 0.05, -1, -1], [10, 14, 5, 2, 1])

    # a stored IRF is axis-aligned: embed the kernel at the centre of a full-length array
    k_full = np.zeros(axis.size)
    lo = axis.size // 2 - k.size // 2
    k_full[lo:lo + k.size] = k
    with_irf = SpectralImage(noisy, axis, instrument_response_function=k_full)
    auto, _ = DHO(expected_peaks=1, p0=p0, bounds=bounds, irf='auto').apply(with_irf)
    assert float(np.nanmedian(auto[..., 2])) == pytest.approx(true_lw, abs=0.08)

    # no IRF on the object -> warns and falls back to a plain (broadened) fit
    without = SpectralImage(noisy, axis)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback, _ = DHO(expected_peaks=1, p0=p0, bounds=bounds, irf='auto').apply(without)
    assert any("instrument_response_function" in str(w.message) for w in caught)
    assert float(np.nanmedian(fallback[..., 2])) > true_lw + 0.1


def test_dho_per_pixel_irf_falls_back_to_plain_where_absent():
    rng = np.random.default_rng(3)
    axis = np.linspace(-15, 15, 300)
    intrinsic = _DHO_1(axis, 1.0, 6.0, 0.6, 0.0, 0.05)
    k = irf_kernel(("lorentzian", 0.5), axis)
    n = k.size
    broadened = np.convolve(np.pad(intrinsic, n, mode="edge"), k, mode="same")[n:-n]
    scale = 400.0 / broadened.max()
    noisy = (rng.poisson(np.clip(np.tile(broadened, (3, 3, 1)) * scale, 0, None)) / scale).astype(float)

    # per-pixel IRF: real kernel everywhere except one all-zero pixel
    per_pixel = np.zeros((3, 3, axis.size))
    lo = axis.size // 2 - k.size // 2
    per_pixel[..., lo:lo + k.size] = k
    per_pixel[0, 0] = 0.0

    image = SpectralImage(noisy, axis, instrument_response_function=per_pixel)
    params, _ = DHO(expected_peaks=1, p0=[1.0, 6.0, 1.0, 0.05, 0.0],
                    bounds=([0, 3, 0.05, -1, -1], [10, 14, 5, 2, 1]), irf='auto').apply(image)

    assert np.isfinite(params[0, 0, 2])          # the no-IRF pixel still fits (plain)
    assert params[1, 1, 2] == pytest.approx(0.6, abs=0.1)  # an IRF pixel recovers the intrinsic width


def test_dho_keeps_zero_count_channels_on_poisson_data():
    # Raw photon counts: most wing channels are legitimately 0. Dropping them biased the
    # fit towards a positive background and a too-narrow line (HWHM -9 %, bg ~0.8).
    from brillouinpy.core import SpectralContainer
    rng = np.random.default_rng(0)
    axis = np.linspace(-21.1, 21.1, 1024)
    truth = [16.0, 7.57, 0.30, 0.0, 0.0]
    counts = rng.poisson(_DHO_1(axis, *truth), size=(200, 1, 1, 1, axis.size)).astype(float)
    counts[..., np.abs(axis) < 4] = np.nan  # blanked elastic region

    params, _ = DHO(expected_peaks=1, p0=[20, 7.6, 0.3, 0.1, 0.0],
                    bounds=([0, 0, 0, -10, -1], [np.inf, 9, 2, np.inf, 1])).apply(
        SpectralContainer(counts, axis))
    params = params.reshape(-1, 5)

    assert np.mean(counts[~np.isnan(counts)] == 0) > 0.5  # the regime this guards
    assert np.mean(params[:, 2]) == pytest.approx(truth[2], rel=0.02)
    assert abs(np.mean(params[:, 3])) < 0.05


def test_dho_all_zero_pixel_yields_nan():
    axis = np.linspace(-15, 15, 200)
    data = np.tile(_DHO_1(axis, 1.0, 6.0, 0.5, 0.05, 0.0), (2, 2, 1))
    data[0, 0] = 0.0
    params, _ = DHO(expected_peaks=1, p0=[1.0, 6.0, 0.5, 0.05, 0.0],
                    bounds=([0, 3, 0.05, -1, -1], [10, 14, 5, 2, 1])).apply(SpectralImage(data, axis))

    assert np.all(np.isnan(params[0, 0]))
    assert params[1, 1, 2] == pytest.approx(0.5, abs=0.01)


# --- Poisson maximum likelihood (loss='poisson') -------------------------------

_POISSON_AXIS = np.linspace(-21.1, 21.1, 1024)
_POISSON_WINDOW = (np.abs(_POISSON_AXIS) > 4) & (np.abs(_POISSON_AXIS) < 15)
_POISSON_TRUTH = np.array([16.0, 7.57, 0.30, 0.0, 0.0])  # ~750 counts per spectrum, no background
_POISSON_BOUNDS = ([0, 0, 0, 0, -1], [np.inf, 9, 2, np.inf, 1])


def _poisson_counts(model, n_spectra, seed):
    """Poisson count spectra of ``model`` at the truth, blanked outside the fit window."""
    counts = np.random.default_rng(seed).poisson(
        model(_POISSON_AXIS, *_POISSON_TRUTH), size=(n_spectra, _POISSON_AXIS.size)).astype(float)
    counts[:, ~_POISSON_WINDOW] = np.nan
    return counts


def _cramer_rao_sd(model):
    """Cramér-Rao lower bound on each parameter's SD for Poisson data at the truth."""
    from brillouinpy.analysis.fit.core import _numeric_jacobian
    x = _POISSON_AXIS[_POISSON_WINDOW]
    mu = model(x, *_POISSON_TRUTH)
    jac = _numeric_jacobian(model, x, _POISSON_TRUTH)
    return np.sqrt(np.diag(np.linalg.inv(jac @ (jac / mu).T)))


@pytest.fixture(scope="module")
def plain_poisson_fits():
    from brillouinpy.core import SpectralContainer
    container = SpectralContainer(_poisson_counts(_DHO_1, 500, seed=10), _POISSON_AXIS)
    fits = {}
    for loss in ("squares", "poisson"):
        params, cov = DHO(expected_peaks=1, p0=list(_POISSON_TRUTH), bounds=_POISSON_BOUNDS,
                          loss=loss, max_workers=4).apply(container)
        fits[loss] = (np.asarray(params), np.asarray(cov))
    return fits


def test_poisson_loss_reaches_cramer_rao_bound(plain_poisson_fits):
    crb = _cramer_rao_sd(_DHO_1)
    sd = {loss: params.std(axis=0, ddof=1) for loss, (params, _) in plain_poisson_fits.items()}

    assert sd["poisson"][1] == pytest.approx(crb[1], rel=0.1)   # BFS
    assert sd["poisson"][2] == pytest.approx(crb[2], rel=0.1)   # HWHM
    assert sd["poisson"][1] < sd["squares"][1]
    assert sd["poisson"][2] < 0.85 * sd["squares"][2]           # the wings carry the linewidth


def test_poisson_loss_is_unbiased_and_pcov_is_inverse_fisher(plain_poisson_fits):
    params, cov = plain_poisson_fits["poisson"]
    shift = params[:, 1]
    assert abs(shift.mean() - _POISSON_TRUTH[1]) < 3 * shift.std(ddof=1) / np.sqrt(shift.size)
    # pcov is the inverse Fisher information, so it predicts the actual scatter
    assert np.sqrt(np.median(cov[:, 1, 1])) == pytest.approx(shift.std(ddof=1), rel=0.15)
    assert np.all(params[:, 3] >= 0)


@pytest.mark.parametrize("per_pixel", [False, True])
def test_poisson_loss_with_irf(per_pixel):
    from brillouinpy.analysis.fit.core import _ConvolvedModel
    from brillouinpy.core import SpectralContainer
    # a measured-style IRF: a kernel embedded at the centre of a full-length array
    k = irf_kernel(("lorentzian", 0.25), _POISSON_AXIS)
    irf = np.zeros(_POISSON_AXIS.size)
    lo = _POISSON_AXIS.size // 2 - k.size // 2
    irf[lo:lo + k.size] = k
    model = _ConvolvedModel(_DHO_1, irf_kernel(irf, _POISSON_AXIS), _POISSON_AXIS)

    counts = _poisson_counts(model, 300, seed=11)
    if per_pixel:
        data = SpectralImage(counts.reshape(20, 15, -1), _POISSON_AXIS,
                             instrument_response_function=np.tile(irf, (20, 15, 1)))
        irf_arg = 'auto'
    else:
        data, irf_arg = SpectralContainer(counts, _POISSON_AXIS), irf

    sd, shift = {}, None
    for loss in ("squares", "poisson"):
        params, _ = DHO(expected_peaks=1, p0=list(_POISSON_TRUTH), bounds=_POISSON_BOUNDS,
                        irf=irf_arg, loss=loss, max_workers=4).apply(data)
        params = np.asarray(params).reshape(-1, 5)
        sd[loss] = params.std(axis=0, ddof=1)
        shift = params[:, 1]

    crb = _cramer_rao_sd(model)
    assert sd["poisson"][1] == pytest.approx(crb[1], rel=0.15)
    assert sd["poisson"][2] < 0.85 * sd["squares"][2]
    assert abs(shift.mean() - _POISSON_TRUTH[1]) < 3 * shift.std(ddof=1) / np.sqrt(shift.size)


def test_default_loss_is_least_squares_unchanged():
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(12)
    axis = np.linspace(-15, 15, 200)
    data = _DHO_1(axis, 1.0, 6.0, 0.5, 0.05, 0.0) + rng.normal(scale=0.01, size=(3, 3, axis.size))
    image = SpectralImage(data, axis)
    kwargs = dict(expected_peaks=1, p0=[1.0, 6.0, 0.5, 0.05, 0.0],
                  bounds=([0, 3, 0.05, -1, -1], [10, 14, 5, 2, 1]))

    default, default_cov = DHO(**kwargs).apply(image)
    squares, squares_cov = DHO(**kwargs, loss='squares').apply(image)
    direct, _ = curve_fit(_DHO_1, axis, data[1, 2], p0=kwargs["p0"], bounds=kwargs["bounds"])

    np.testing.assert_array_equal(default, squares)
    np.testing.assert_array_equal(np.asarray(default_cov), np.asarray(squares_cov))
    np.testing.assert_allclose(default[1, 2], direct, rtol=1e-12)


def test_poisson_loss_clips_negative_background_bound_and_warns_on_non_counts():
    from brillouinpy.core import SpectralContainer
    counts = _poisson_counts(_DHO_1, 4, seed=13)
    bounds = ([0, 0, 0, -10, -1], [np.inf, 9, 2, np.inf, 1])
    p0 = [16.0, 7.57, 0.3, -0.5, 0.0]

    with pytest.warns(UserWarning, match="non-negative Background"):
        params, _ = DHO(expected_peaks=1, p0=p0, bounds=bounds, loss='poisson',
                        max_workers=1).apply(SpectralContainer(counts, _POISSON_AXIS))
    assert np.all(np.asarray(params)[:, 3] >= 0)

    with pytest.warns(UserWarning, match="raw photon counts"):
        DHO(expected_peaks=1, p0=list(_POISSON_TRUTH), bounds=_POISSON_BOUNDS, loss='poisson',
            max_workers=1).apply(SpectralContainer(counts / counts[np.isfinite(counts)].max(),
                                                   _POISSON_AXIS))


def test_loss_validation():
    from brillouinpy.analysis.fit.core import Gaussian, Lorentzian, PeakFit
    for cls, extra in [(DHO, {}), (Lorentzian, {}), (Gaussian, {}), (PeakFit, {"model": "dho"})]:
        with pytest.raises(ValueError, match="loss"):
            cls(expected_peaks=1, loss='absolute', **extra)
        cls(expected_peaks=1, loss='poisson', **extra)  # accepted
    with pytest.raises(NotImplementedError, match="poisson"):
        DHO(expected_peaks='auto', loss='poisson')


# --- automatic peak-count selection (expected_peaks='auto') -------------------

def _multi_mode_image(shifts, linewidths, nx=8, ny=8, n_channels=250, noise=3e-4, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.linspace(-20, 20, n_channels)
    spectrum = np.zeros(n_channels)
    for s, lw in zip(shifts, linewidths):
        spectrum += _DHO_1(axis, 5e-3, s, lw, 0.0, 0.0)
    data = np.tile(spectrum, (nx, ny, 1)) + rng.normal(scale=noise, size=(nx, ny, n_channels))
    return SpectralImage(np.clip(data, 0, None), axis)


def test_estimate_peak_count_one_and_two_modes():
    one = _multi_mode_image([7.5], [0.8], seed=1)
    two = _multi_mode_image([6.0, 10.0], [0.8, 0.9], seed=2)

    assert estimate_peak_count(one) == 1
    assert estimate_peak_count(two) == 2

    best, scores = estimate_peak_count(two, return_scores=True)
    assert best == 2 and scores[2] < scores[1]


def test_estimate_peak_count_min_improvement_controls_parsimony():
    two = _multi_mode_image([6.0, 10.0], [0.8, 0.9], seed=3)
    # a huge threshold refuses to ever add a peak
    assert estimate_peak_count(two, min_improvement=1e9) == 1


def test_dho_auto_returns_padded_params_and_recovers_modes():
    two = _multi_mode_image([6.0, 10.0], [0.8, 0.9], nx=6, ny=6, seed=4)

    params, cov = DHO(expected_peaks='auto', max_peaks=3).apply(two)
    cov = np.asarray(cov)

    assert params.shape == (6, 6, 11)          # padded to 3 * max_peaks + 2
    assert cov.shape == (6, 6, 11, 11)
    counts = fitted_peak_count(params)
    assert counts.shape == (6, 6)
    assert np.median(counts) == 2

    two_peak = counts == 2
    s1 = np.abs(params[..., 1])[two_peak]
    s2 = np.abs(params[..., 4])[two_peak]
    lo, hi = np.minimum(s1, s2), np.maximum(s1, s2)
    assert np.nanmedian(lo) == pytest.approx(6.0, abs=0.4)
    assert np.nanmedian(hi) == pytest.approx(10.0, abs=0.4)
    # the unused third peak triplet (amplitude, shift, width at indices 6, 7, 8) is NaN
    assert np.all(np.isnan(params[two_peak][:, 6:9]))


def test_dho_auto_single_mode_picks_one_almost_everywhere():
    one = _multi_mode_image([8.0], [1.0], nx=8, ny=8, seed=5)
    params, _ = DHO(expected_peaks='auto').apply(one)
    counts = fitted_peak_count(params)
    # per-pixel model selection is noisy; the overwhelming majority is 1 mode
    assert np.mean(counts == 1) >= 0.9
