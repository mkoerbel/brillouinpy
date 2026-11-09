# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:35:14 2025

@author: Timm
"""

import warnings

import numpy as np
import scipy.optimize


# is_aligned() below is adapted, near-verbatim, from RamanSPy
# (https://github.com/barahona-research-group/RamanSPy), Copyright (c) 2023
# Dimitar Georgiev, licensed under the BSD 3-Clause License (see LICENSE and
# THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository root). The
# rest of this file is original.
def is_aligned(spectral_objects):
    """
    Checks whether a collection of spectral objects all share the same spectral axis.

    Parameters
    ----------
    spectral_objects : list of SpectralObject
        The spectral objects to compare.

    Returns
    -------
    bool
        ``True`` if all objects have an identical ``spectral_axis``, ``False`` otherwise.
    """
    unique_shift_axes = [
        np.array(unique) for unique in set(tuple(spectral_object.spectral_axis) for spectral_object in spectral_objects)]

    return len(unique_shift_axes) == 1


def wavelength_to_wavenumber(wavelengths, laser_excitation):
    """
    Converts wavelengths (nm) to Raman shift wavenumbers (cm :sup:`-1`) relative to a laser
    excitation wavelength.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths to convert, in nm.
    laser_excitation : float
        Excitation wavelength of the laser used to acquire the data, in nm.

    Returns
    -------
    array_like
        The corresponding Raman shift wavenumbers, in cm :sup:`-1`.
    """
    return 1e7/laser_excitation - 1e7/wavelengths


def wavenumber_to_wavelength(raman_shifts, laser_excitation):
    """
    Converts Raman shift wavenumbers (cm :sup:`-1`) back to wavelengths (nm), the inverse of
    :func:`wavelength_to_wavenumber`.

    Parameters
    ----------
    raman_shifts : array_like
        Raman shift wavenumbers to convert, in cm :sup:`-1`.
    laser_excitation : float
        Excitation wavelength of the laser used to acquire the data, in nm.

    Returns
    -------
    array_like
        The corresponding wavelengths, in nm.
    """
    return 1/(1 / laser_excitation - raman_shifts / 1e7)


def raman_spectral_axis(grating, wavelength):
    """
    Computes the wavelength axis of a Raman spectrometer for a given grating and center
    wavelength, based on the monochromator/CCD geometry hard-coded in this function.

    Parameters
    ----------
    grating : float
        Groove density of the grating, in grooves/mm.
    wavelength : float
        Center wavelength of the monochromator, in nm.

    Returns
    -------
    numpy.ndarray
        The 2048-channel wavelength axis (matching the CCD's horizontal pixel count), in nm.
    """
    # Hab jetzt den Bandpass berechnet, der passt fast immer bis auf 0.2 nm mit der Berechnung von Horiba überein.
    deviationAngle = 21.26 # deviation angle of mono in degree
    deviationAngleRad = deviationAngle * np.pi/180. # deviation angle in rad
    deviationHalfAngleRad = deviationAngleRad/2. # deviation angle in rad
    focalLength = 318.72 # output focal length of mono in mm
    inclineAngle = -5.5 # inclination angle of CCD in degree
    inclineAngleRad = inclineAngle * np.pi/180. # inclination angle of CCD in rad
    order = 1 # diffraction order
    detHPixel = 2048 # horizontal pixels
    pixelSize = 0.014 # in mm
    detWidth = detHPixel * pixelSize # detector width in mm
    alpha = np.arcsin((10**-6 * order * wavelength * grating)/(2 * np.cos(deviationHalfAngleRad))) - deviationAngleRad/2
    beta = deviationAngleRad+alpha
    linearDispersion = 10**6 * np.cos(beta) * np.cos(inclineAngleRad)/(focalLength * grating* order)
    monoBandpass = linearDispersion * detWidth
    pixelBandpass = monoBandpass / detHPixel
    lowerspectrumedge = wavelength - monoBandpass/2
    higherspectrumedge = wavelength + monoBandpass/2
    wavelength_axis = np.linspace(lowerspectrumedge, higherspectrumedge, detHPixel, endpoint=True)
    return wavelength_axis


def fsr(mirror_spacing):
    """
    Computes the Free Spectral Range (FSR) of a tandem Fabry-Perot interferometer.

    Parameters
    ----------
    mirror_spacing : float
        The mirror spacing of the interferometer, in m.

    Returns
    -------
    float
        The free spectral range, in Hz.
    """
    speed_of_light = 299792458  # m/s
    return speed_of_light / (2 * mirror_spacing)


def brillouin_spectral_axis(mirror_spacing, scan_amplitude, no_of_channels, laser_wavelength = 532.1e-9):
    """
    Computes the Brillouin frequency-shift axis of a tandem Fabry-Perot interferometer scan.

    Parameters
    ----------
    mirror_spacing : float
        The mirror spacing of the interferometer, in m.
    scan_amplitude : float
        The mirror scan amplitude, in the same units expected by :func:`fsr`'s
        ``mirror_spacing``/wavelength ratio (i.e. such that ``fsr(mirror_spacing) *
        scan_amplitude / laser_wavelength`` yields a frequency in Hz).
    no_of_channels : int
        Number of spectral channels to generate.
    laser_wavelength : float, optional
        Excitation laser wavelength, in m. Default is 532.1 nm (``532.1e-9``).

    Returns
    -------
    numpy.ndarray
        The symmetric frequency-shift axis, in GHz, of length ``no_of_channels``.

    Notes
    -----
    The scan spans ``2 * L`` with ``L = fsr * scan_amplitude / laser_wavelength``
    and is divided into ``no_of_channels`` channels of equal duration (TFP manual
    §4.9, §5.5; GHOST manual §3.1, §3.5.1). The returned values are the channel
    centres, spaced by ``2 * L / no_of_channels`` and symmetric about 0, so the
    outermost channels lie half a channel inside ``±L``. Whether 0 GHz falls on a
    channel centre or between two channels is not specified by the manuals; this
    does not affect fitted shifts, which carry their own spectral-offset parameter.
    """
    freq_limits = fsr(mirror_spacing) * scan_amplitude / laser_wavelength
    freq_limits_GHz = freq_limits * 1e-9
    step = 2 * freq_limits_GHz / no_of_channels
    return (np.arange(no_of_channels) - (no_of_channels - 1) / 2) * step


def TFP_IRF_Analysis(intensity_data, spectral_axis):
    '''
    Analyze the Instrumental Response Function (IRF) from the intensity data in terms of laser line shape and intenisty
    Assuming the data is 2D (x, y, spectral) or 3D (x, y, z, spectral) or 4D (x, y, z, t, spectral).
    '''
    spectral_response = np.zeros(intensity_data.shape)
    laser_fitness = np.zeros(intensity_data.shape[:-1])
    fwhm_values = np.zeros(intensity_data.shape[:-1])


    for x in range(intensity_data.shape[0]):
        for y in range(intensity_data.shape[1]):
            for z in range(intensity_data.shape[2]):
                for t in range(intensity_data.shape[3]):
                    start, end = _find_instrumental_response(intensity_data[x, y, z, t, :])
                    spectral_response[x, y, z, t, start:end] = intensity_data[x, y, z, t, start:end]
                    try:
                        spectrum = spectral_response[x, y, z, t, :]
                        p0 = [np.max(spectrum), spectral_axis[np.argmax(spectrum)], 0.1, np.min(spectrum)]
                        popt, _ = scipy.optimize.curve_fit(_gauss, spectral_axis, spectrum, p0=p0)
                        a, mu, sigma, c = popt
                        fwhm = 2.3548 * abs(sigma)  # FWHM for Gaussian
                        fwhm_values[x, y, z, t] = fwhm
                        laser_fitness[x, y, z, t] = a
                        # Optional: Save fitted curve if needed
                        # spectral_response[x, y, z, t, :] = _gauss(spectral_axis, *popt)
                    except Exception:
                        fwhm_values[x, y, z, t] = np.nan
                        laser_fitness[x, y, z, t] = np.nan
    return spectral_response, spectral_axis, laser_fitness, fwhm_values


def _gauss(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c


def _find_instrumental_response(pixel_spectrum):
    max_index = np.argmax(pixel_spectrum)

    start = next((i for i in range(max_index, 0, -1) if pixel_spectrum[i] == 0), 0)
    end = next((i for i in range(max_index, len(pixel_spectrum)) if pixel_spectrum[i] == 0), len(pixel_spectrum))

    return start, end


# --- Deprecated: moved to brillouinpy.io (2026-08-27) ---
#
# The acquisition-side loaders below moved to dedicated submodules under
# brillouinpy.io, split by what kind of data/setup they're for:
#   - brillouinpy.io.tfp        (this group's TFP setup - the actively used path)
#   - brillouinpy.io.legacy     (superseded loaders, kept for old datasets)
#   - brillouinpy.io.multimodal (Raman)
# These wrappers keep 'brillouinpy.utils.X(...)' callable unchanged, but emit a
# DeprecationWarning and forward to the new location - update your code to call
# brillouinpy.io.* directly when convenient. See CHANGELOG.md for details.

def _deprecated_move_warning(old_name, new_path):
    warnings.warn(
        f"'brillouinpy.utils.{old_name}' has moved to 'brillouinpy.io.{new_path}'. "
        f"Please update your code to call 'brillouinpy.io.{new_path}(...)' directly - "
        f"this compatibility wrapper may be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def read_meta(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.tfp.read_meta`."""
    from .io import tfp
    _deprecated_move_warning('read_meta', 'tfp.read_meta')
    return tfp.read_meta(*args, **kwargs)


def brillouin_spectral_axis_from_meta(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.tfp.brillouin_spectral_axis_from_meta`."""
    from .io import tfp
    _deprecated_move_warning('brillouin_spectral_axis_from_meta', 'tfp.brillouin_spectral_axis_from_meta')
    return tfp.brillouin_spectral_axis_from_meta(*args, **kwargs)


def extract_coordinates(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.tfp.extract_coordinates`."""
    from .io import tfp
    _deprecated_move_warning('extract_coordinates', 'tfp.extract_coordinates')
    return tfp.extract_coordinates(*args, **kwargs)


def import_DAT_File(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.tfp.import_DAT_File`."""
    from .io import tfp
    _deprecated_move_warning('import_DAT_File', 'tfp.import_DAT_File')
    return tfp.import_DAT_File(*args, **kwargs)


def prepare_brillouin_data(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.tfp.prepare_brillouin_data`."""
    from .io import tfp
    _deprecated_move_warning('prepare_brillouin_data', 'tfp.prepare_brillouin_data')
    return tfp.prepare_brillouin_data(*args, **kwargs)


def load_spectral_image(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.legacy.load_spectral_image`."""
    from .io import legacy
    _deprecated_move_warning('load_spectral_image', 'legacy.load_spectral_image')
    return legacy.load_spectral_image(*args, **kwargs)


def load_spectral_image_brio2(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.legacy.load_spectral_image_brio2`."""
    from .io import legacy
    _deprecated_move_warning('load_spectral_image_brio2', 'legacy.load_spectral_image_brio2')
    return legacy.load_spectral_image_brio2(*args, **kwargs)


def prepare_raman_data(*args, **kwargs):
    """Deprecated: moved to :func:`brillouinpy.io.multimodal.prepare_raman_data`."""
    from .io import multimodal
    _deprecated_move_warning('prepare_raman_data', 'multimodal.prepare_raman_data')
    return multimodal.prepare_raman_data(*args, **kwargs)
