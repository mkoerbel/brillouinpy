# -*- coding: utf-8 -*-
"""
Example 10 - Exporting results
================================

Shows how to export processed data and fit results to formats other tools can read:

- Per-parameter 32-bit TIFF images (:func:`brillouinpy.io.export.fit_to_tiff`),
  e.g. for opening in ImageJ/Fiji.
- The `HDF5_BLS <https://github.com/bio-brillouin/HDF5_BLS>`_ format
  (:func:`brillouinpy.io.to_hdf5_bls`/:func:`~brillouinpy.io.from_hdf5_bls`),
  for interoperability with other Brillouin analysis software. Requires the optional
  ``HDF5_BLS`` package (``pip install HDF5_BLS``).
- The `brim <https://github.com/brillouin-imaging/Brillouin-standard-file>`_ format
  (:func:`brillouinpy.io.to_brim`/:func:`~brillouinpy.io.from_brim`),
  a Zarr-based standard for Brillouin microscopy data that's also readable by the
  `napari <https://github.com/brillouin-imaging/brillouin-imaging-napari>`_ and
  `Fiji <https://github.com/brillouin-imaging/brillouin-imaging-fiji>`_ brim viewer
  plugins, and by `BrimView <https://github.com/brillouin-imaging/BrimView>`_
  (no installation needed). Requires the optional ``brimfile`` package
  (``pip install brimfile``).

Continues from ``03_preprocess_data.py`` and ``09_fit_spectra.py``: run those first
to get ``pp_data/preprocessed_image.pkl`` and ``pp_data/fitted_parameters.npy``,
otherwise this script falls back to synthetic data on its own.
"""
import os

import numpy as np

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    preprocessed_path = os.path.join('pp_data', 'preprocessed_image.pkl')
    fit_path = os.path.join('pp_data', 'fitted_parameters.npy')

    if os.path.exists(preprocessed_path):
        preprocessed_image = bp.SpectralImage.load(preprocessed_path)
    else:
        preprocessed_image = single_peak_image()

    if os.path.exists(fit_path):
        fitted_parameters = np.load(fit_path)
    else:
        fitted_parameters, _ = bp.analysis.fit.DHO(
            expected_peaks=1, p0=[0.005, 8.5, 1, 0, 0], bounds=None,
        ).apply(preprocessed_image)
        fitted_parameters = np.ma.filled(fitted_parameters, np.nan)

    os.makedirs('pp_data', exist_ok=True)

    # ------------------------------------------------------------------
    # Export the fitted parameter maps as individual TIFF images
    # ------------------------------------------------------------------
    tiff_files = bp.io.export.fit_to_tiff(
        fitted_parameters,
        output_directory=os.path.join('pp_data', 'tiff'),
        expected_peaks=1,   # must match the model used to produce 'fitted_parameters'
        prefix='dho_fit',
    )
    print('Wrote TIFF files:', tiff_files)

    # ------------------------------------------------------------------
    # Export the preprocessed data (+ fit) to the HDF5_BLS format
    # ------------------------------------------------------------------
    try:
        bp.io.to_hdf5_bls(
            preprocessed_image,
            os.path.join('pp_data', 'brillouin_data.h5'),
            sample='Example sample',
            fit_result=fitted_parameters,
            expected_peaks=1,
            x_step=0.5, y_step=0.5, spatial_unit='um',
            overwrite=True,
        )
        print('Wrote pp_data/brillouin_data.h5')

        # Reading it back returns a ready-to-use spectral object
        reloaded = bp.io.from_hdf5_bls(os.path.join('pp_data', 'brillouin_data.h5'))
        print(f"Reloaded from HDF5_BLS: shape {reloaded.shape}, metadata {reloaded.metadata}")

    except ImportError as exc:
        print(f"Skipping HDF5_BLS export ({exc}).")

    # ------------------------------------------------------------------
    # Export the preprocessed data (+ fit) to the brim format
    # ------------------------------------------------------------------
    try:
        bp.io.to_brim(
            preprocessed_image,
            os.path.join('pp_data', 'brillouin_data.brim.zarr'),
            sample='Example sample',
            laser_wavelength_nm=532.1,
            fit_result=fitted_parameters,
            expected_peaks=1,
            x_step_um=0.5, y_step_um=0.5,  # brim's pixel size is always in micrometers
            overwrite=True,
        )
        print('Wrote pp_data/brillouin_data.brim.zarr')

        # Reading it back returns a ready-to-use spectral object
        reloaded = bp.io.from_brim(os.path.join('pp_data', 'brillouin_data.brim.zarr'))
        print(f"Reloaded from brim: shape {reloaded.shape}, metadata {reloaded.metadata}")

    except ImportError as exc:
        print(f"Skipping brim export ({exc}).")
