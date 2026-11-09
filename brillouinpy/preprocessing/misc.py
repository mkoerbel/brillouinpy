import copy
from numbers import Number
from typing import Tuple
import numpy as np
from skimage import restoration

from . import PreprocessingStep
from ..core import Spectrum


# BackgroundSubtractor and Cropper below are adapted, near-verbatim, from
# RamanSPy (https://github.com/barahona-research-group/RamanSPy), Copyright (c)
# 2023 Dimitar Georgiev, licensed under the BSD 3-Clause License (see LICENSE and
# THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository root).
# IRF_Remover and Deconvoluter_IRF further below are original to this package.
class BackgroundSubtractor(PreprocessingStep):
    """
    Subtract a fixed reference background.

    Parameters
    ----------
    background : Spectrum
        The reference background to subtract.
    """

    def __init__(self, *, background: Spectrum):
        super().__init__(_subtract_background, background=background)


class Cropper(PreprocessingStep):
    """
    Crop the intensity values and the shift axis associated with the band range(s) specified.

    Parameters
    ----------
    region : tuple of two elements
        The band intervals to crop (in GHz).

        For instance:

            - [(None, 300)] - keeps the bands < 300cm-1
            - [(3000, None)] - keeps the bands > 3000cm-1
            - [(700, 1800)] - keeps the bands between 700 and 1800 (i.e. the "fingerprint" region)
    """

    def __init__(self, *, region: Tuple[Number or None, Number or None]): # type: ignore
        if len(region) != 2:
            raise ValueError("The region must be a tuple of two elements")

        super().__init__(_crop, region=region)

class IRF_Remover(PreprocessingStep):
    """
    Remove the intensity values corresponding to the Instrument Response Function (IRF) including an optional little offset.

    The removed channels are set to ``numpy.nan`` (not 0), so fits drop them while genuine
    zero-count channels elsewhere in the spectrum are kept as data. On raw TFP data choose
    ``offset`` large enough to also cover the shutter-blocked channels next to the IRF.
    Works on any spatial dimensionality, including the raw ``(x, y, z, t, spectral)`` shape.

    Parameters
    ----------
    offset : number of spectral channels which gets removed
        The frequency channels to crop (in channels).

        For instance:
            - None - only the IRF gets detected, saved and removed from the spectral data
            - 10 - the IRF and additional 10 channels get removed
    store_irf : bool, optional
        If ``True`` (default), the detected per-pixel IRF region is attached to
        the returned object as its ``instrument_response_function`` (a
        ``(*spatial, B)`` array, zero outside the detected IRF extent), for a
        later ``brillouinpy.analysis.fit.DHO(irf='auto')`` fit.
    """
    def __init__(self, *, offset: Number or None, store_irf: bool = True): # type: ignore

        super().__init__(_irf_remove, offset=offset, store_irf=store_irf)


class Deconvoluter_IRF(PreprocessingStep):
    """
    Deconvolve the Instrument Response Function (IRF) out of the intensity data using the
    Richardson-Lucy algorithm, then remove the (now deconvolved) IRF region from the spectrum.

    Unlike :class:`IRF_Remover`, which just crops the IRF away, this step first uses the
    detected IRF as the point-spread function to sharpen the rest of the spectrum via
    Richardson-Lucy deconvolution, and only then blanks out the IRF channels themselves
    (set to ``numpy.nan``) plus the requested offset around them. Downstream steps must be
    able to handle ``numpy.nan`` values (e.g. :class:`~brillouinpy.analysis.Step.AnalysisStep`
    drops NaN-containing channels automatically).

    Works on any spectral object (``Spectrum``, ``SpectralImage``, ``SpectralVolume``, or the
    raw ``(x, y, z, t, spectral)`` shape returned by
    :func:`brillouinpy.io.tfp.prepare_brillouin_data`) - it loops over whatever spatial
    dimensions are present.

    Parameters
    ----------
    offset : number of spectral channels which gets removed
        The frequency channels to blank out around the detected IRF (in channels), in
        addition to the IRF's own extent.
    iterations : int or None
        The number of Richardson-Lucy deconvolution iterations to run.
    padding : int or None
        Number of channels to reflect-pad the spectrum with on each side before running
        the deconvolution, then crop back off afterwards. Richardson-Lucy deconvolution
        otherwise treats the array edges as a hard boundary, which can produce artefacts
        (a spurious up/down swing) near the first/last few channels; padding with a
        reflected copy of the spectrum gives the algorithm room to work without that
        boundary, at the cost of a bit of extra computation. ``None`` or ``0`` disables
        padding. A value comparable to a few times the IRF's width is a reasonable start;
        it must be smaller than the spectrum's length.
    store_irf : bool, optional
        Default ``False``. If ``True``, the detected per-pixel IRF region is also
        attached to the returned object as its ``instrument_response_function``
        (a ``(*spatial, B)`` array, zero outside the detected IRF extent), for
        inspection or plotting. **Do not** then fit with
        ``brillouinpy.analysis.fit.DHO(irf='auto')``: this step has already
        deconvolved the IRF out, so a convolution fit on top would correct for it
        twice. The convolution-fit workflow uses :class:`IRF_Remover` (which only
        crops the elastic peak) instead - see ``benchmarks/irf_convolution.py``.
    """

    def __init__(self, *, offset: Number or None, iterations: Number or None, padding: Number or None, # type: ignore
                 store_irf: bool = False):
        super().__init__(_deconvolute_irf, offset=offset, iterations=iterations, padding=padding,
                         store_irf=store_irf)

def _subtract_background(original_intensity_data, original_spectral_axis, background: Spectrum):
    if not np.array_equal(background.spectral_axis, original_spectral_axis):
        raise ValueError("The spectral axis of the background must match that of the spectral object to process")

    return original_intensity_data[..., :] - background.spectral_data, original_spectral_axis


def _crop(intensity_data, spectral_axis, region):
    indices_to_leave = _get_indices_to_leave(spectral_axis, region)

    return intensity_data[..., indices_to_leave], spectral_axis[indices_to_leave]


def _get_indices_to_leave(spectral_axis, region):
    start = spectral_axis[0] if region[0] is None else region[0]
    end = spectral_axis[-1] if region[1] is None else region[1]

    if start > end:
        # swap
        start, end = end, start

    indices_to_leave = np.logical_and(
        start <= spectral_axis, spectral_axis <= end)

    return indices_to_leave

def _irf_remove(intensity_data, spectral_axis, offset, store_irf=True):
    '''
    Blank the Instrumental Response Function (IRF) plus ``offset`` channels on each side
    to ``numpy.nan`` (like :func:`_deconvolute_irf`, so the fit drops them while genuine
    zero counts stay in). Works on any spatial dimensionality.
    '''
    raw = np.ma.filled(intensity_data, 0).astype(float)
    corrected_intensity_data = np.ma.filled(intensity_data, np.nan).astype(float)
    spectral_response = np.zeros(raw.shape)
    n_channels = raw.shape[-1]
    offset = offset or 0

    for idx in np.ndindex(raw.shape[:-1]):
        start, end = _find_instrumental_response(raw[idx])
        spectral_response[idx + (slice(start, end),)] = raw[idx + (slice(start, end),)]
        corrected_intensity_data[
            idx + (slice(max(0, start - offset), min(n_channels, end + offset)),)
        ] = np.nan

    if store_irf:
        return corrected_intensity_data, spectral_axis, spectral_response
    return corrected_intensity_data, spectral_axis

def _find_instrumental_response(pixel_spectrum):
    max_index = np.argmax(pixel_spectrum)

    start = next((i for i in range(max_index, 0, -1) if pixel_spectrum[i] == 0), 0)
    end = next((i for i in range(max_index, len(pixel_spectrum)) if pixel_spectrum[i] == 0), len(pixel_spectrum))

    return start, end

def _deconvolute_irf(intensity_data, spectral_axis, offset, iterations=4, padding=None, store_irf=False):
    '''
    Deconvolute the intensity data with the Instrumental Response Function (IRF) using the Richardson-Lucy algorithm.
    Additionally, the IRF gets removed from the spectral data.
    Works on any spatial dimensionality (0D - a single Spectrum - through the 4D (x, y, z, t,
    spectral) shape returned by prepare_brillouin_data).

    With ``store_irf`` (default), also returns the detected per-pixel IRF region
    (``spectral_response``, zero outside the detected extent) as a third value,
    which :meth:`PreprocessingStep._process_object` attaches to the result as its
    ``instrument_response_function``.
    '''
    corrected_intensity_data = np.copy(np.asarray(intensity_data))
    spectral_response = np.zeros(intensity_data.shape)
    spatial_shape = intensity_data.shape[:-1]
    n_channels = intensity_data.shape[-1]
    pad = padding or 0
    offset = offset or 0

    for idx in np.ndindex(spatial_shape):
        pixel_spectrum = intensity_data[idx + (slice(None),)]
        start, end = _find_instrumental_response(pixel_spectrum)
        spectral_response[idx + (slice(start, end),)] = pixel_spectrum[start:end]

        image = corrected_intensity_data[idx + (slice(None),)]
        psf = spectral_response[idx + (slice(None),)]
        if pad:
            # Richardson-Lucy treats the array edges as a hard boundary, which produces
            # a spurious up/down swing near the first/last few channels; padding with a
            # reflected copy gives it room to work before cropping back to the original
            # range. The PSF is zero-padded to match, so the IRF's position within the
            # padded array stays aligned with the (also padded) spectrum.
            image = np.pad(image, pad, mode='reflect')
            psf = np.pad(psf, pad, mode='constant')

        # Apply Richardson-Lucy deconvolution
        deconvolved = restoration.richardson_lucy(
            image, psf, num_iter=iterations, clip=False, filter_epsilon=None
        )
        corrected_intensity_data[idx + (slice(None),)] = deconvolved[pad:pad + n_channels] if pad else deconvolved
        # Remove the IRF and additional offset
        corrected_intensity_data[
            idx + (slice(max(0, start - offset), min(intensity_data.shape[-1], end + offset)),)
        ] = np.nan

    if store_irf:
        return corrected_intensity_data, spectral_axis, spectral_response
    return corrected_intensity_data, spectral_axis


def _as_irf_array(irf_measurements, spectral_axis):
    """Coerce an IRF-measurement input to a ``(N, B)`` array aligned to ``spectral_axis``."""
    if hasattr(irf_measurements, "flat") and hasattr(irf_measurements, "spectral_data"):
        arr = np.ma.filled(irf_measurements.flat.spectral_data, np.nan).astype(float)
    elif isinstance(irf_measurements, (list, tuple)):
        arr = np.vstack([np.ma.filled(getattr(m, "spectral_data", m), np.nan).astype(float).ravel()
                         for m in irf_measurements])
    else:
        arr = np.asarray(irf_measurements, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != len(spectral_axis):
        raise ValueError(
            f"IRF measurements have {arr.shape[-1]} channels but the spectral axis has "
            f"{len(spectral_axis)}; they must be on the same axis.")
    return arr


def assign_irf(spectral_object, irf_measurements, *, at=None, method='nearest'):
    """
    Attach an IRF that was measured *separately* from the acquisition to a spectral
    object, expanding it to a per-pixel ``instrument_response_function``.

    This is the counterpart to :class:`Deconvoluter_IRF` / :class:`IRF_Remover`'s
    ``store_irf`` for spectrometers (e.g. VIPA) where the IRF is not recorded on
    every scan but once, before/after, or a handful of times during the
    acquisition for drift correction.

    Parameters
    ----------
    spectral_object : core.SpectralObject
        The acquisition to attach the IRF to. A copy is returned; the input is
        unchanged.
    irf_measurements : array_like, Spectrum, SpectralContainer or list of Spectrum
        One or more IRF spectra, on the same spectral axis as ``spectral_object``.
        Shape ``(B,)`` or ``(N, B)``.
    at : array_like or None
        For ``N > 1``: the position of each measurement in the acquisition, as a
        flattened (row-major) pixel index into ``spectral_object.flat`` (e.g.
        ``[0, n_pixels // 2, n_pixels - 1]`` for before / middle / after).
        Ignored - and may be ``None`` - when a single IRF is given, which is then
        stored as a shared 1-D kernel.
    method : {'nearest', 'linear'}
        How to map the measurements onto every pixel: ``'nearest'`` assigns each
        pixel the closest measurement; ``'linear'`` interpolates channel-wise
        between the two bracketing measurements (constant extrapolation outside).

    Returns
    -------
    core.SpectralObject
        A copy of ``spectral_object`` with ``instrument_response_function`` set -
        1-D ``(B,)`` for a single measurement, ``(*spatial, B)`` otherwise. Pass
        it to ``brillouinpy.analysis.fit.DHO(irf='auto')``.
    """
    axis = spectral_object.spectral_axis
    meas = _as_irf_array(irf_measurements, axis)

    out = copy.deepcopy(spectral_object)
    if meas.shape[0] == 1:
        out.instrument_response_function = meas[0]
        return out

    if at is None:
        raise ValueError("at= is required when more than one IRF measurement is given.")
    at = np.asarray(at, dtype=float)
    if at.shape[0] != meas.shape[0]:
        raise ValueError(f"at has {at.shape[0]} entries but {meas.shape[0]} IRF measurements were given.")

    n_pixels = int(np.prod(spectral_object.shape))
    pixels = np.arange(n_pixels, dtype=float)
    if method == 'nearest':
        nearest = np.argmin(np.abs(pixels[:, None] - at[None, :]), axis=1)
        per_pixel = meas[nearest]
    elif method == 'linear':
        order = np.argsort(at)
        per_pixel = np.column_stack([
            np.interp(pixels, at[order], meas[order, b]) for b in range(meas.shape[1])
        ])
    else:
        raise ValueError(f"method must be 'nearest' or 'linear', got {method!r}.")

    out.instrument_response_function = per_pixel.reshape(*spectral_object.shape, meas.shape[1])
    return out
