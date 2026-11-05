# -*- coding: utf-8 -*-
"""
Export utilities for interoperability with external tools.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import tifffile

from .. import core


def _dho_parameter_names(expected_peaks: int) -> list:
    """Parameter names for the last axis of a DHO/Lorentzian FitStep result."""
    names = []
    for i in range(expected_peaks):
        suffix = "" if i == 0 else f"_{i + 1}"
        names += [f"I0{suffix}", f"freqShift{suffix}", f"LineWidth{suffix}"]
    names += ["Background", "axis_shift"]
    return names


_MEASUREMENT_GROUP_TYPES = {"Measure", "Calibration_spectrum", "Impulse_response"}


def _open_hdf5_bls(filepath):
    try:
        from HDF5_BLS import Wrapper
    except ImportError as exc:
        raise ImportError(
            "Reading/writing the HDF5_BLS format requires the 'HDF5_BLS' package. "
            "Install it with 'pip install HDF5_BLS'."
        ) from exc
    return Wrapper(filepath)


def _find_measurement_groups(structure: dict, path: str = "Brillouin") -> list:
    groups = []
    if structure.get("Brillouin_type") in _MEASUREMENT_GROUP_TYPES:
        groups.append(path)
    for key, child in structure.items():
        if key == "Brillouin_type" or not isinstance(child, dict):
            continue
        groups.extend(_find_measurement_groups(child, f"{path}/{key}"))
    return groups


def list_measurements(filepath: str) -> list:
    """
    List the measurement-like groups (Brillouin_type "Measure", "Calibration_spectrum"
    or "Impulse_response") contained in an HDF5_BLS file.

    Useful to find the ``measure_group`` path to pass to :func:`from_hdf5_bls` when a
    file contains more than one measurement.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file.

    Returns
    -------
    list[str]
        The full paths (e.g. "Brillouin/Measure") of the measurement groups found.
    """
    wrapper = _open_hdf5_bls(filepath)
    try:
        return _find_measurement_groups(wrapper.get_structure()["Brillouin"])
    finally:
        wrapper.close()


def from_hdf5_bls(filepath: str, measure_group: Optional[str] = None) -> core.SpectralObject:
    """
    Read a measurement stored in the HDF5_BLS format
    (https://github.com/bio-brillouin/HDF5_BLS) back into a brillouinpy
    spectral object, ready for further processing (preprocessing pipelines,
    fitting, ...).

    Requires the ``HDF5_BLS`` package (``pip install HDF5_BLS``).

    Parameters
    ----------
    filepath : str
        Path to the .h5 file.
    measure_group : str, optional
        Path of the group inside the file holding the data to read (e.g.
        "Brillouin/Measure"). If omitted, the file's only measurement group is
        used automatically; if the file contains several (e.g. a calibration
        spectrum next to the actual measurement), you must specify which one to
        read - use :func:`list_measurements` to see the available options.

    Returns
    -------
    SpectralObject
        A :class:`~brillouinpy.Spectrum`/:class:`~brillouinpy.SpectralImage`/
        :class:`~brillouinpy.SpectralVolume`/:class:`~brillouinpy.SpectralContainer`
        instance (chosen automatically based on the dimensionality of the stored
        data), with the group's attributes attached as a ``metadata`` dict.
    """
    wrapper = _open_hdf5_bls(filepath)
    try:
        if measure_group is None:
            candidates = _find_measurement_groups(wrapper.get_structure()["Brillouin"])
            if len(candidates) != 1:
                raise ValueError(
                    "Could not determine which measurement group to read automatically "
                    f"(found: {candidates}). Pass 'measure_group' explicitly - "
                    "use brillouinpy.io.list_measurements(filepath) to see the options."
                )
            measure_group = candidates[0]
        elif measure_group not in _find_measurement_groups(wrapper.get_structure()["Brillouin"]):
            raise ValueError(
                f"'{measure_group}' is not a measurement group in '{filepath}'. "
                "Use brillouinpy.io.list_measurements(filepath) to see the available options."
            )

        psd_names = wrapper.get_children_elements(measure_group, Brillouin_type="PSD")
        if not psd_names:
            psd_names = wrapper.get_children_elements(measure_group, Brillouin_type="Raw_data")
            if not psd_names:
                raise ValueError(f"No PSD or Raw_data dataset found under '{measure_group}'.")

        freq_names = wrapper.get_children_elements(measure_group, Brillouin_type="Frequency")
        if not freq_names:
            raise ValueError(f"No Frequency dataset found under '{measure_group}'.")

        intensity_data = np.asarray(wrapper[f"{measure_group}/{psd_names[0]}"])
        spectral_axis = np.asarray(wrapper[f"{measure_group}/{freq_names[0]}"])

        spectral_object = core._create_data(intensity_data, spectral_axis)
        spectral_object.metadata = wrapper.get_attributes(measure_group)

        return spectral_object
    finally:
        wrapper.close()


def to_hdf5_bls(
    spectral_object: core.SpectralObject,
    filepath: str,
    *,
    sample: Optional[str] = None,
    attributes: Optional[dict] = None,
    fit_result: Optional[np.ndarray] = None,
    expected_peaks: Optional[int] = None,
    parameter_names: Optional[Sequence[str]] = None,
    x_step: Optional[float] = None,
    y_step: Optional[float] = None,
    spatial_unit: str = "um",
    overwrite: bool = False,
) -> None:
    """
    Export a spectral object (and, optionally, a peak-fit result) to an HDF5 file
    compatible with the HDF5_BLS format (https://github.com/bio-brillouin/HDF5_BLS).

    Requires the ``HDF5_BLS`` package (``pip install HDF5_BLS``).

    Parameters
    ----------
    spectral_object : SpectralObject
        The (typically preprocessed) Brillouin data to export. Its ``spectral_data``
        is stored as the PSD and its ``spectral_axis`` as the frequency axis.
    filepath : str
        Destination path of the .h5 file.
    sample : str, optional
        Sample name, stored as a "Sample" attribute on the Measure group.
    attributes : dict, optional
        Additional free-form metadata stored as attributes on the Measure group.
    fit_result : numpy.ndarray of shape (..., n_params), optional
        The array of fitted parameters as returned by ``FitStep.apply`` (e.g.
        ``brillouinpy.analysis.fit.DHO``/``Lorentzian``). The last axis
        is expected to hold, per fitted peak, the triplet (I0, freqShift,
        LineWidth), followed by a single shared (Background, axis_shift) pair.
    expected_peaks : int, optional
        Number of fitted peaks in ``fit_result``. Required if ``fit_result`` is
        given and ``parameter_names`` isn't.
    parameter_names : list[str], optional
        Explicit names for the last axis of ``fit_result``, overriding the default
        DHO/Lorentzian naming derived from ``expected_peaks``.
    x_step, y_step : float, optional
        Real-world pixel spacing along x/y, used to build the spatial abscissae.
        If omitted, pixel indices are stored instead.
    spatial_unit : str, optional
        Unit of ``x_step``/``y_step``, by default "um".
    overwrite : bool, optional
        Overwrite ``filepath`` if it already exists, by default False.
    """
    wrapper = _open_hdf5_bls(None)
    saved = False
    try:
        wrapper.create_group("Measure", parent_group="Brillouin", brillouin_type="Measure")
        wrapper.add_PSD(np.ma.filled(spectral_object.spectral_data, np.nan), parent_group="Brillouin/Measure")
        wrapper.add_frequency(np.asarray(spectral_object.spectral_axis), parent_group="Brillouin/Measure")

        shape = spectral_object.shape
        if shape != (1,):
            if len(shape) >= 1:
                x_axis = np.arange(shape[0]) * (x_step if x_step else 1)
                wrapper.add_abscissa(
                    x_axis, parent_group="Brillouin/Measure", name="X",
                    unit=spatial_unit if x_step else "px", dim_start=0, dim_end=1,
                )
            if len(shape) >= 2:
                y_axis = np.arange(shape[1]) * (y_step if y_step else 1)
                wrapper.add_abscissa(
                    y_axis, parent_group="Brillouin/Measure", name="Y",
                    unit=spatial_unit if y_step else "px", dim_start=1, dim_end=2,
                )

        meta = dict(attributes or {})
        if sample is not None:
            meta["Sample"] = sample
        if meta:
            wrapper.add_attributes(meta, parent_group="Brillouin/Measure")

        if fit_result is not None:
            fit_result = np.asarray(fit_result)
            if parameter_names is None:
                if expected_peaks is None:
                    raise ValueError(
                        "Provide either 'parameter_names' or 'expected_peaks' together with 'fit_result'."
                    )
                parameter_names = _dho_parameter_names(expected_peaks)

            n_params = fit_result.shape[-1]
            if len(parameter_names) != n_params:
                raise ValueError(
                    f"'parameter_names' has {len(parameter_names)} entries but "
                    f"fit_result's last axis has {n_params}."
                )

            n_peaks = (n_params - 2) // 3
            wrapper.create_group("Treatment", parent_group="Brillouin", brillouin_type="Treatment")

            for i in range(n_peaks):
                wrapper.add_treated_data(
                    parent_group="Brillouin/Treatment",
                    name_group=f"Treat_{i}",
                    amplitude=np.asarray(fit_result[..., 3 * i]),
                    shift=np.asarray(fit_result[..., 3 * i + 1]),
                    # brillouinpy's LineWidth is the HWHM; HDF5_BLS uses the FWHM.
                    linewidth=np.asarray(2.0 * fit_result[..., 3 * i + 2]),
                )

            for j, name in enumerate(parameter_names[3 * n_peaks:]):
                wrapper.add_other(
                    np.asarray(fit_result[..., 3 * n_peaks + j]),
                    parent_group="Brillouin/Treatment",
                    name=name,
                )

        wrapper.save_as_hdf5(filepath, overwrite=overwrite)
        saved = True
    finally:
        if saved:
            wrapper.close()
        else:
            # We bailed out before save_as_hdf5 (e.g. a ValueError on a bad
            # fit_result). A plain wrapper.close() then raises
            # WrapperError_Save and masks the real error - drop the unsaved
            # temp file instead and let the original exception propagate.
            try:
                wrapper.close(delete_temp_file=True)
            except Exception:
                pass


def fit_to_tiff(
    fit_result: np.ndarray,
    output_directory: str,
    *,
    expected_peaks: Optional[int] = None,
    parameter_names: Optional[Sequence[str]] = None,
    prefix: str = "fit",
) -> list:
    """
    Export each fitted parameter map of a peak-fit result as a separate 32-bit
    float TIFF image, so students can load them in ImageJ/Fiji or any other tool.

    Parameters
    ----------
    fit_result : numpy.ndarray of shape (x, y, n_params)
        The array of fitted parameters as returned by ``FitStep.apply`` (e.g.
        ``brillouinpy.analysis.fit.DHO``/``Lorentzian``).
    output_directory : str
        Directory the TIFF files are written to. Created if it doesn't exist.
    expected_peaks : int, optional
        Number of fitted peaks in ``fit_result``. Required if ``fit_result`` is
        given and ``parameter_names`` isn't.
    parameter_names : list[str], optional
        Explicit names for the last axis of ``fit_result`` used as filenames,
        overriding the default DHO/Lorentzian naming derived from ``expected_peaks``.
    prefix : str, optional
        Filename prefix, by default "fit". Files are written as
        "{prefix}_{parameter_name}.tif".

    Returns
    -------
    list[str]
        The paths of the written TIFF files.
    """
    import os

    # np.asarray() would silently strip the mask off a masked-array input, making
    # the np.ma.filled(..., np.nan) below a no-op; np.ma.asarray() preserves it
    # (and is a no-op itself for a plain ndarray).
    fit_result = np.ma.asarray(fit_result)
    if fit_result.ndim != 3:
        raise ValueError(
            f"'fit_result' must have shape (x, y, n_params), got shape {fit_result.shape}."
        )

    if parameter_names is None:
        if expected_peaks is None:
            raise ValueError("Provide either 'parameter_names' or 'expected_peaks'.")
        parameter_names = _dho_parameter_names(expected_peaks)

    n_params = fit_result.shape[-1]
    if len(parameter_names) != n_params:
        raise ValueError(
            f"'parameter_names' has {len(parameter_names)} entries but "
            f"fit_result's last axis has {n_params}."
        )

    os.makedirs(output_directory, exist_ok=True)

    written_files = []
    for i, name in enumerate(parameter_names):
        parameter_map = np.ma.filled(fit_result[..., i], np.nan).astype(np.float32)
        filepath = os.path.join(output_directory, f"{prefix}_{name}.tif")
        tifffile.imwrite(filepath, parameter_map)
        written_files.append(filepath)

    return written_files


def _open_brim():
    try:
        import brimfile as brim
    except ImportError as exc:
        raise ImportError(
            "Reading/writing the brim format requires the 'brimfile' package "
            "(https://github.com/brillouin-imaging/brimfile). Install it with "
            "'pip install brimfile'."
        ) from exc
    return brim


def _spectral_data_to_zyx_psd(spectral_object: core.SpectralObject) -> np.ndarray:
    """Reshapes 'spectral_data' into the (z, y, x, spectral) layout brim requires."""
    data = np.ma.filled(spectral_object.spectral_data, np.nan)
    if data.ndim == 1:
        return data.reshape(1, 1, 1, -1)
    elif data.ndim == 3:
        return np.transpose(data, (1, 0, 2))[np.newaxis, ...]
    elif data.ndim == 4:
        return np.transpose(data, (2, 1, 0, 3))
    else:
        raise ValueError(
            "'to_brim' only supports Spectrum, SpectralImage or SpectralVolume "
            f"objects (spectral_data with 1, 3 or 4 dimensions); got {data.ndim} dimensions."
        )


def _zyx_psd_to_spectral_data(psd_zyx: np.ndarray) -> np.ndarray:
    """Inverse of '_spectral_data_to_zyx_psd'."""
    if psd_zyx.shape[0] == 1:
        return np.transpose(psd_zyx[0], (1, 0, 2))  # (y, x, spectral) -> (x, y, spectral)
    return np.transpose(psd_zyx, (2, 1, 0, 3))  # (z, y, x, spectral) -> (x, y, z, spectral)


def _spatial_map_to_zyx(array) -> np.ndarray:
    """Reshapes a fitted parameter's (x, y) or (x, y, z) spatial map into (z, y, x)."""
    array = np.ma.filled(array, np.nan) if np.ma.is_masked(array) else np.asarray(array)
    if array.ndim == 2:
        return np.transpose(array, (1, 0))[np.newaxis, ...]
    elif array.ndim == 3:
        return np.transpose(array, (2, 1, 0))
    else:
        raise ValueError(f"Expected a 2D (x, y) or 3D (x, y, z) spatial map, got shape {array.shape}.")


#: The five metadata categories brim knows (``brimfile.Metadata.Type`` members).
_BRIM_METADATA_CATEGORIES = ("Experiment", "Optics", "Brillouin", "Acquisition", "Spectrometer")

#: Best-effort mapping of common flat vendor / ``META.json`` metadata keys
#: (compared lower-cased) onto ``(brim category, attribute, units or None)``.
#: ``"Info"`` as the attribute routes the value into the free-form
#: ``Experiment.Info`` string rather than a schema field.
_FLAT_METADATA_MAP = {
    "date": ("Experiment", "Datetime", None),
    "datetime": ("Experiment", "Datetime", None),
    "timestamp": ("Experiment", "Timestamp", None),
    "sample": ("Experiment", "Sample", None),
    "sample_name": ("Experiment", "Sample", None),
    "temperature": ("Experiment", "Temperature", None),
    "operator": ("Experiment", "Info", None),
    "notes": ("Experiment", "Info", None),
    "note": ("Experiment", "Info", None),
    "info": ("Experiment", "Info", None),
    "comment": ("Experiment", "Info", None),
    "comments": ("Experiment", "Info", None),
    "description": ("Experiment", "Info", None),
    "wavelength": ("Optics", "Wavelength", None),
    "wavelength_nm": ("Optics", "Wavelength", "nm"),
    "laser_wavelength_nm": ("Optics", "Wavelength", "nm"),
    "laser_wavelenght_nm": ("Optics", "Wavelength", "nm"),  # sic - legacy misspelling
    "laser_power": ("Optics", "Power", None),
    "power_mw": ("Optics", "Power", "mW"),
    "laser_power_mw": ("Optics", "Power", "mW"),
    "laser": ("Optics", "Laser_model", None),
    "laser_model": ("Optics", "Laser_model", None),
    "objective": ("Optics", "Objective_model", None),
    "objective_model": ("Optics", "Objective_model", None),
    "scattering_angle": ("Brillouin", "Scattering_angle", None),
}


def _normalise_brim_metadata(metadata, *, drop_sample=False, drop_wavelength=False):
    """Coerce an arbitrary metadata dict into brim's ``{category: {attribute:
    value}}`` shape.

    Entries already keyed by a brim category (:data:`_BRIM_METADATA_CATEGORIES`)
    pass straight through. A flat vendor / ``META.json`` dict is best-effort
    mapped onto the brim schema via :data:`_FLAT_METADATA_MAP`; everything that
    doesn't map is bundled into ``Experiment.Info`` as a ``key: value; ...``
    string, so no metadata is silently dropped (just not given a schema field).

    ``drop_sample`` / ``drop_wavelength`` remove the mapped ``Experiment.Sample``
    / ``Optics.Wavelength`` so an explicit ``sample=`` / ``laser_wavelength_nm=``
    argument to :func:`to_brim` wins over whatever the dict carried.
    """
    if not metadata:
        return {}

    normalised = {category: {} for category in _BRIM_METADATA_CATEGORIES}
    info_parts = []

    for key, value in metadata.items():
        if key in _BRIM_METADATA_CATEGORIES and isinstance(value, dict):
            normalised[key].update(value)
            continue
        category, attribute, units = _FLAT_METADATA_MAP.get(
            str(key).strip().lower(), (None, None, None))
        if attribute == "Info":
            info_parts.append(f"{key}: {value}")
        elif category is not None:
            normalised[category][attribute] = (value, units) if units else value
        else:
            info_parts.append(f"{key}: {value}")

    if info_parts:
        existing = normalised["Experiment"].get("Info")
        existing = existing[0] if isinstance(existing, tuple) else existing
        normalised["Experiment"]["Info"] = "; ".join(
            str(part) for part in ([existing] if existing else []) + info_parts)

    if drop_sample:
        normalised["Experiment"].pop("Sample", None)
    if drop_wavelength:
        normalised["Optics"].pop("Wavelength", None)

    return {category: attrs for category, attrs in normalised.items() if attrs}


def to_brim(
    spectral_object: core.SpectralObject,
    filepath: str,
    *,
    sample: Optional[str] = None,
    laser_wavelength_nm: Optional[float] = None,
    acquisition_datetime=None,
    metadata: Optional[dict] = None,
    fit_result: Optional[np.ndarray] = None,
    expected_peaks: Optional[int] = None,
    fit_model: str = "DHO",
    x_step_um: Optional[float] = None,
    y_step_um: Optional[float] = None,
    z_step_um: Optional[float] = None,
    as_zip: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Export a spectral object (and, optionally, a peak-fit result) to the brim
    format (https://github.com/brillouin-imaging/Brillouin-standard-file), a
    Zarr-based format used by the brimfile
    (https://github.com/brillouin-imaging/brimfile) Python library and the
    napari/Fiji brim viewer plugins.

    Requires the ``brimfile`` package (``pip install brimfile``).

    Parameters
    ----------
    spectral_object : SpectralObject
        The (typically preprocessed) Brillouin data to export. Must be a
        :class:`~brillouinpy.Spectrum`, :class:`~brillouinpy.SpectralImage`
        or :class:`~brillouinpy.SpectralVolume` (i.e. ``spectral_data`` with 1,
        3 or 4 dimensions) - brim's PSD array is always 4D ``(z, y, x, spectral)``.
        Any ``channels`` on the object are not written (brim is a Brillouin-only
        format); use pickle ``save`` to keep them.
    filepath : str
        Destination path of the brim store. By convention, ``.brim.zarr`` for a
        directory store (the default), or ``.brim.zip`` when ``as_zip=True``.
    as_zip : bool, optional
        If ``True``, write a single ``.brim.zip`` archive file instead of a
        ``.brim.zarr`` directory (Zarr's usual, chunk-per-file layout). Both are
        valid brim files and both are readable by :func:`from_brim`, brimfile
        itself, and the brim viewer plugins/tools; a zip archive is easier to
        move/share as one file, at the cost of needing to be (partially)
        decompressed to read from. Default is ``False``.
    sample : str, optional
        Sample description, stored as the "Experiment.Sample" metadata field.
    laser_wavelength_nm : float, optional
        Excitation laser wavelength, in nm, stored as the "Optics.Wavelength"
        metadata field.
    acquisition_datetime : str or datetime.datetime/date, optional
        Value for the "Experiment.Datetime" metadata field (ISO 8601). If omitted,
        a Datetime is still written - taken from ``metadata`` if present there
        (e.g. a value round-tripped by :func:`from_brim`), otherwise the current
        local time. brillouinpy always writes this field because some viewers
        (BrimView) error out on a file whose Experiment metadata section is
        missing entirely.
    metadata : dict, optional
        Defaults to ``spectral_object.metadata`` if the object has one (e.g. when
        it came from :func:`from_brim`, or from a loader that attached the
        acquisition ``META.json``); pass a dict here to override that.

        Ideally already in brim form - ``{category: {attribute: value_or_(value,
        units)}}``, where ``category`` is a ``brimfile.Metadata.Type`` member
        (``"Experiment"``, ``"Optics"``, ``"Brillouin"``, ``"Acquisition"``,
        ``"Spectrometer"``) and each attribute matches the brim schema (call
        ``brimfile.metadata.print_schema()``). Values without units can be given
        directly; a value needing units as a ``(value, units)`` tuple.

        A **flat** dict (plain ``{key: value}``, e.g. a vendor ``META.json``) is
        accepted too: well-known keys (``Date``, ``Sample``, laser
        wavelength/power, ...) are mapped onto the brim schema and everything
        else is bundled into ``Experiment.Info`` as a ``key: value; ...`` string,
        so nothing is lost even though it isn't given a dedicated field.
    fit_result : numpy.ndarray of shape (..., n_params), optional
        The array of fitted parameters as returned by ``FitStep.apply`` (e.g.
        ``brillouinpy.analysis.fit.DHO``/``Lorentzian``). The last axis
        is expected to hold, per fitted peak, the triplet (I0, freqShift,
        LineWidth), followed by a single shared (Background, axis_shift) pair. Since
        brillouinpy's peak models fit one symmetric peak per mode, the same
        values are written for both the "AntiStokes" and "Stokes" sides.
    expected_peaks : int, optional
        Number of fitted peaks in ``fit_result``. Required if ``fit_result`` is given.
    fit_model : {"DHO", "Lorentzian", "Gaussian", "Voigt", "Custom", "Undefined"}, optional
        The fit model to record for ``fit_result``, matching
        ``brimfile.AnalysisResults.FitModel``. Default is ``"DHO"``.
    x_step_um, y_step_um, z_step_um : float, optional
        Real-world pixel spacing along x/y/z, in micrometers - brim's ``px_size_um``
        is always in micrometers, unlike :func:`to_hdf5_bls`'s configurable unit.
        Each axis not given here falls back to ``spectral_object.px_size_um`` if
        the object carries one (e.g. from :func:`from_brim`).
        If omitted, that axis' pixel size is left undefined.
    overwrite : bool, optional
        Overwrite ``filepath`` if it already exists, by default False.
    """
    import os
    import shutil

    brim = _open_brim()

    if overwrite and os.path.exists(filepath):
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)

    PSD = _spectral_data_to_zyx_psd(spectral_object)
    frequency = np.asarray(spectral_object.spectral_axis)

    # Carry over the bits that :func:`from_brim` attaches to the object, so a
    # from_brim -> (process) -> to_brim round-trip preserves them without the
    # caller having to thread them through by hand. Anything passed explicitly
    # still wins.
    stored_px_size = getattr(spectral_object, "px_size_um", None) or {}
    if x_step_um is None:
        x_step_um = stored_px_size.get("x")
    if y_step_um is None:
        y_step_um = stored_px_size.get("y")
    if z_step_um is None:
        z_step_um = stored_px_size.get("z")

    if metadata is None:
        metadata = getattr(spectral_object, "metadata", None)

    if z_step_um is None and PSD.shape[0] == 1:
        # A single z-slice (e.g. a plain SpectralImage) has no real z-extent, so
        # this is a harmless placeholder rather than a physical claim - unlike
        # x_step_um/y_step_um, which are left as None (and thus null in the file)
        # if not given, since those axes are essentially always physically
        # meaningful. Some brim readers (e.g. BrimView, as of writing) don't
        # handle a null pixel size gracefully, so avoid writing one where we can.
        z_step_um = 1.0

    store_type = brim.StoreType.ZIP if as_zip else brim.StoreType.ZARR
    f = brim.File.create(filepath, store_type=store_type)
    try:
        data_group = f.create_data_group(PSD, frequency, (z_step_um, y_step_um, x_step_um))

        # brimfile's own File.create()/create_data_group() never write the root
        # 'Subtype' attribute unless one of the brimfile.subtypes.* helpers (e.g.
        # single_point_VIPA.add_rawdata) is used. brimfile's own reader (File.subtype)
        # tolerates that absence and defaults to SubType.none, but at least one
        # downstream viewer (BrimView, as of brimfile 1.7.0) does not, and fails
        # with "Invalid subtype: None" on a file that never touched that API. Write
        # it explicitly, using brimfile's own (private but stable) helper for it,
        # so the file matches what a file that went through the "normal" (subtype-
        # aware) writing path would contain. Falls back to a no-op if that helper
        # ever disappears in a future brimfile release.
        try:
            from brimfile.subtypes.utils import _check_or_create_subtype
            _check_or_create_subtype(f._file, brim.subtypes.SubType.none)
        except (ImportError, AttributeError):
            pass

        Item = brim.Metadata.Item
        md = data_group.get_metadata()
        if sample is not None:
            md.add(brim.Metadata.Type.Experiment, {"Sample": Item(sample)})
        if laser_wavelength_nm is not None:
            md.add(brim.Metadata.Type.Optics, {"Wavelength": Item(laser_wavelength_nm, "nm")})

        normalised_metadata = _normalise_brim_metadata(
            metadata, drop_sample=sample is not None,
            drop_wavelength=laser_wavelength_nm is not None)
        for category_name, attributes in normalised_metadata.items():
            items = {
                key: value if isinstance(value, brim.Metadata.Item)
                else Item(*value) if isinstance(value, tuple)
                else Item(value)
                for key, value in attributes.items()
            }
            md.add(brim.Metadata.Type[category_name], items)

        # Always write an Experiment.Datetime. It is optional in the brim spec,
        # but at least one viewer (BrimView) assumes the Experiment metadata
        # section exists and crashes with "AttributeError: 'NoneType' object has
        # no attribute 'get'" on a file that has none. Prefer, in order: an
        # explicit 'acquisition_datetime', a value already carried in 'metadata'
        # (e.g. round-tripped by from_brim), otherwise the current local time.
        if "Datetime" not in normalised_metadata.get("Experiment", {}):
            import datetime as _dt

            value = acquisition_datetime
            if value is None:
                value = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
            elif isinstance(value, (_dt.datetime, _dt.date)):
                value = value.isoformat()
            md.add(brim.Metadata.Type.Experiment, {"Datetime": Item(str(value))})

        if fit_result is not None:
            fit_result = np.asarray(fit_result)
            if expected_peaks is None:
                raise ValueError("Provide 'expected_peaks' together with 'fit_result'.")

            background = _spatial_map_to_zyx(fit_result[..., 3 * expected_peaks])
            peak_data = [
                {
                    "amplitude": _spatial_map_to_zyx(fit_result[..., 3 * i]), "amplitude_units": "a.u.",
                    "shift": _spatial_map_to_zyx(fit_result[..., 3 * i + 1]), "shift_units": "GHz",
                    # brillouinpy's LineWidth is the HWHM; brim/HDF5_BLS use the FWHM.
                    "width": _spatial_map_to_zyx(2.0 * fit_result[..., 3 * i + 2]), "width_units": "GHz",
                    "offset": background, "offset_units": "a.u.",
                }
                for i in range(expected_peaks)
            ]
            peak_data = peak_data if expected_peaks > 1 else peak_data[0]

            # brillouinpy's peak models fit one symmetric (I0, freqShift, LineWidth)
            # triplet per mode, describing both the AntiStokes and Stokes peaks equally.
            data_group.create_analysis_results_group(
                peak_data, peak_data, fit_model=brim.AnalysisResults.FitModel[fit_model],
            )
    finally:
        f.close()


_MICROMETER_UNITS = {"um", "µm", "micron", "microns", "micrometer", "micrometre", "micrometers", "micrometres"}


def _read_px_size_um(data_group) -> dict:
    """
    Best-effort read of a data group's pixel size, as ``{"x", "y", "z"} -> float|None``
    in micrometers. Returns ``None`` for an axis whose size brim doesn't define (or
    defines in units we don't recognise as micrometers).
    """
    from numbers import Number

    result = {"x": None, "y": None, "z": None}
    try:
        # brimfile exposes this as a (z, y, x) tuple of Metadata.Item; it is a
        # "private but stable" attribute (populated on open, no public accessor
        # as of brimfile 1.7.0).
        px_z, px_y, px_x = data_group._spatial_map_px_size
    except (AttributeError, TypeError, ValueError):
        return result

    for axis, item in (("x", px_x), ("y", px_y), ("z", px_z)):
        value = getattr(item, "value", None)
        units = getattr(item, "units", None)
        if not isinstance(value, Number):
            continue
        # units=None with value==1 is brimfile's "undefined" placeholder.
        if units is None and value == 1:
            continue
        if units is None or str(units).lower() in _MICROMETER_UNITS:
            result[axis] = float(value)
    return result


def from_brim(filepath: str, *, index: int = 0) -> core.SpectralObject:
    """
    Read a measurement stored in the brim format
    (https://github.com/brillouin-imaging/Brillouin-standard-file) back into a
    brillouinpy spectral object, ready for further processing (preprocessing
    pipelines, fitting, ...).

    Requires the ``brimfile`` package (``pip install brimfile``).

    Parameters
    ----------
    filepath : str
        Path to the ``.brim.zarr`` store.
    index : int, optional
        Index of the data group to read, if the file contains more than one - use
        :func:`list_brim_measurements` to see the available options. Default is 0
        (the first/only one).

    Returns
    -------
    SpectralObject
        A :class:`~brillouinpy.Spectrum`/:class:`~brillouinpy.SpectralImage`/
        :class:`~brillouinpy.SpectralVolume` instance (chosen automatically
        based on the dimensionality of the stored data; a single spectrum stored in
        a 1x1-pixel data group comes back as a 1x1 ``SpectralImage``, since brim
        doesn't distinguish the two cases), with the data group's metadata attached
        as a ``metadata`` dict of the form ``{category: {attribute: (value, units)}}``
        and the pixel size attached as a ``px_size_um`` dict
        (``{"x"/"y"/"z": float_or_None}``, in micrometers). :func:`to_brim` reads
        both back off the object, so a ``from_brim`` -> process -> ``to_brim``
        round-trip carries them over automatically.
    """
    brim = _open_brim()

    f = brim.File(filepath)
    try:
        data_group = f.get_data(index)
        PSD, frequency, _psd_units, _frequency_units = data_group.get_PSD_as_spatial_map(broadcast_frequency=False)

        try:
            raw_metadata = data_group.get_metadata().all_to_dict()
        except ValueError:
            # Works around a bug in brimfile <= 1.7.0: reading the metadata of a
            # file that never had any metadata written to it tries to lazily
            # initialise it, which fails because the file is open read-only here.
            raw_metadata = {}

        metadata = {
            category: {
                attribute: (item.value, item.units) for attribute, item in attributes.items()
            }
            for category, attributes in raw_metadata.items()
            if attributes  # brimfile lists every known category; skip the empty ones
        }

        return core._create_data(
            _zyx_psd_to_spectral_data(np.asarray(PSD)), np.asarray(frequency),
            metadata=metadata, px_size_um=_read_px_size_um(data_group),
        )
    finally:
        f.close()


def list_brim_measurements(filepath: str) -> list:
    """
    List the data groups contained in a brim file.

    Useful to find the ``index`` to pass to :func:`from_brim` when a file contains
    more than one measurement.

    Parameters
    ----------
    filepath : str
        Path to the ``.brim.zarr`` store.

    Returns
    -------
    list[dict]
        The data groups found (each with ``'name'``, ``'index'`` and
        ``'custom_name'`` keys).
    """
    brim = _open_brim()
    f = brim.File(filepath)
    try:
        return f.list_data_groups(retrieve_custom_name=True)
    finally:
        f.close()
