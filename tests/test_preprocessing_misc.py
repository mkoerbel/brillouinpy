import numpy as np
import pytest

from brillouinpy.preprocessing.misc import Deconvoluter_IRF, IRF_Remover, assign_irf
from brillouinpy.core import Spectrum, SpectralImage, SpectralContainer


def _spectrum_with_irf(n_channels=100, irf_index=50, irf_width=2):
    axis = np.linspace(-20, 20, n_channels)
    peak = np.exp(-0.5 * ((np.arange(n_channels) - (irf_index - 20)) / 3) ** 2)
    irf = np.exp(-0.5 * ((np.arange(n_channels) - irf_index) / irf_width) ** 2)
    data = 0.1 * peak + irf
    data[irf_index - 10:irf_index - 7] = 0
    data[irf_index + 7:irf_index + 10] = 0
    return data, axis


def test_deconvoluter_irf_blanks_irf_region_on_spectrum():
    data, axis = _spectrum_with_irf()
    spectrum = Spectrum(data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(spectrum)

    assert np.all(np.isnan(result.spectral_data[45:55]))
    assert not np.any(np.isnan(result.spectral_data[:40]))


def test_deconvoluter_irf_works_on_spectral_image():
    data, axis = _spectrum_with_irf()
    image_data = np.tile(data, (3, 3, 1))
    image = SpectralImage(image_data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(image)

    assert isinstance(result, SpectralImage)
    assert result.spectral_data.shape == image_data.shape
    assert np.all(np.isnan(result.spectral_data[1, 1, 45:55]))


def test_deconvoluter_irf_works_on_raw_5d_shape():
    data, axis = _spectrum_with_irf()
    volume_data = np.tile(data, (2, 2, 1, 1, 1))
    container = SpectralContainer(volume_data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(container)

    assert result.spectral_data.shape == volume_data.shape
    assert np.all(np.isnan(result.spectral_data[0, 1, 0, 0, 45:55]))


def test_deconvoluter_irf_padding_preserves_shape_and_stays_finite():
    data, axis = _spectrum_with_irf()
    spectrum = Spectrum(data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=15).apply(spectrum)

    assert result.spectral_data.shape == data.shape
    valid = ~np.isnan(result.spectral_data)
    assert np.all(np.isfinite(result.spectral_data[valid]))
    assert np.all(np.isnan(result.spectral_data[45:55]))


def test_deconvoluter_irf_padding_reduces_edge_artefact():
    # A peak whose flank is cut off right at the array's edge (index 0) - the case padding
    # is meant to help with, since the unpadded deconvolution treats that as a hard boundary.
    n_channels = 100
    axis = np.linspace(-20, 20, n_channels)
    irf_index = 50
    irf = np.exp(-0.5 * ((np.arange(n_channels) - irf_index) / 2) ** 2)
    edge_peak = np.exp(-0.5 * ((np.arange(n_channels) - 3) / 3) ** 2)
    data = 0.1 * edge_peak + irf
    data[irf_index - 10:irf_index - 7] = 0
    data[irf_index + 7:irf_index + 10] = 0
    spectrum = Spectrum(data, axis)

    unpadded = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(spectrum)
    padded = Deconvoluter_IRF(offset=2, iterations=4, padding=20).apply(spectrum)

    # Padding changes the result near the edge - it isn't silently ignored.
    assert not np.allclose(unpadded.spectral_data[:5], padded.spectral_data[:5])


# --- IRF carried on the container ---------------------------------------------

def test_irf_remover_store_irf_attaches_per_pixel_irf():
    data, axis = _spectrum_with_irf()
    image = SpectralImage(np.tile(data, (3, 3, 1)), axis)

    result = IRF_Remover(offset=2, store_irf=True).apply(image)

    irf = result.instrument_response_function
    assert irf is not None and irf.shape == image.spectral_data.shape
    # non-zero only around the detected IRF region
    assert np.count_nonzero(irf[1, 1]) > 0
    assert np.count_nonzero(irf[1, 1, :30]) == 0

    # opt-out
    assert IRF_Remover(offset=2, store_irf=False).apply(image).instrument_response_function is None
    # Deconvoluter_IRF does not store it by default (would be a double correction)
    assert Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(
        image).instrument_response_function is None


def test_irf_remover_blanks_to_nan_and_keeps_zeros_elsewhere():
    data, axis = _spectrum_with_irf()
    data[:5] = 0  # genuine zero counts far from the IRF
    result = IRF_Remover(offset=2, store_irf=False).apply(Spectrum(data, axis))

    assert np.all(np.isnan(result.spectral_data[45:55]))
    assert np.all(result.spectral_data[:5] == 0)
    assert not np.any(np.isnan(result.spectral_data[:40]))


def test_irf_remover_works_on_raw_5d_shape():
    data, axis = _spectrum_with_irf()
    volume_data = np.tile(data, (2, 2, 1, 3, 1))
    result = IRF_Remover(offset=2, store_irf=True).apply(SpectralContainer(volume_data, axis))

    assert result.spectral_data.shape == volume_data.shape
    assert np.all(np.isnan(result.spectral_data[1, 0, 0, 2, 45:55]))
    assert result.instrument_response_function.shape == volume_data.shape


def test_irf_remover_offset_none_and_no_wraparound():
    # IRF right at the start of the array: start - offset < 0 must not wrap to the end
    n_channels = 100
    axis = np.linspace(-20, 20, n_channels)
    data = 0.1 * np.ones(n_channels)
    data[:4] = [5.0, 3.0, 1.0, 0.5]
    data[4] = 0

    clipped = IRF_Remover(offset=10, store_irf=False).apply(Spectrum(data, axis))
    assert np.all(np.isnan(clipped.spectral_data[:14]))
    assert not np.any(np.isnan(clipped.spectral_data[14:]))

    bare = IRF_Remover(offset=None, store_irf=False).apply(Spectrum(data, axis))
    assert np.all(np.isnan(bare.spectral_data[:4]))
    assert not np.any(np.isnan(bare.spectral_data[4:]))


def test_cropper_would_invalidate_stored_irf(recwarn):
    from brillouinpy.preprocessing.misc import Cropper
    data, axis = _spectrum_with_irf()
    image = SpectralImage(np.tile(data, (2, 2, 1)), axis,
                          instrument_response_function=np.tile(data, (2, 2, 1)))

    result = Cropper(region=(-10, 10)).apply(image)

    assert result.instrument_response_function is None
    assert any("instrument_response_function" in str(w.message) for w in recwarn)


def test_assign_irf_single_measurement_is_shared():
    axis = np.linspace(-20, 20, 40)
    image = SpectralImage(np.random.rand(3, 4, 40), axis)
    irf = np.hanning(40)

    result = assign_irf(image, irf)

    assert result.instrument_response_function.shape == (40,)
    np.testing.assert_array_equal(result.instrument_response_function, irf)
    assert image.instrument_response_function is None  # input untouched


def test_assign_irf_multiple_measurements_expand_to_per_pixel():
    axis = np.linspace(-20, 20, 40)
    image = SpectralImage(np.random.rand(4, 5, 40), axis)  # 20 pixels
    meas = np.stack([np.full(40, 1.0), np.full(40, 3.0)])

    nearest = assign_irf(image, meas, at=[0, 19], method='nearest')
    assert nearest.instrument_response_function.shape == (4, 5, 40)
    flat = nearest.instrument_response_function.reshape(20, 40)
    assert flat[0, 0] == 1.0 and flat[-1, 0] == 3.0

    linear = assign_irf(image, meas, at=[0, 19], method='linear')
    mid = linear.instrument_response_function.reshape(20, 40)[10, 0]
    assert 1.0 < mid < 3.0

    with pytest.raises(ValueError):  # missing at=
        assign_irf(image, meas)
