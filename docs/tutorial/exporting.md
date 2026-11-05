# Exporting results

*Script: `examples/10_export_data.py`*

Once you have preprocessed data and/or a fit result, `brillouinpy.io.export`
gets it into formats other tools can read:

```python
# One 32-bit float TIFF per fitted parameter, e.g. for ImageJ/Fiji
bp.io.export.fit_to_tiff(fitted_parameters, output_directory='pp_data/tiff', expected_peaks=1)

# The HDF5_BLS format (https://github.com/bio-brillouin/HDF5_BLS), for
# interoperability with other Brillouin analysis software (requires the optional
# 'HDF5_BLS' package: pip install HDF5_BLS)
bp.io.to_hdf5_bls(
    preprocessed_image, 'pp_data/brillouin_data.h5',
    fit_result=fitted_parameters, expected_peaks=1, overwrite=True,
)
reloaded = bp.io.from_hdf5_bls('pp_data/brillouin_data.h5')

# The brim format (https://github.com/brillouin-imaging/Brillouin-standard-file),
# a Zarr-based standard also readable by the napari/Fiji brim viewer plugins and
# by BrimView, no installation needed (requires the optional 'brimfile' package:
# pip install brimfile)
bp.io.to_brim(
    preprocessed_image, 'pp_data/brillouin_data.brim.zarr',
    fit_result=fitted_parameters, expected_peaks=1,
    x_step_um=0.5, y_step_um=0.5, overwrite=True,
)
reloaded = bp.io.from_brim('pp_data/brillouin_data.brim.zarr')
```

If the object carries `metadata` / `px_size_um` (e.g. from `io.read_meta`, or from
`io.from_brim` on a round-trip), `to_brim` reads them off the object and writes
them back out, so you don't have to re-supply them.

Use the TIFF export when you just need a given parameter map as an image for a
figure or for further processing in ImageJ/Fiji (e.g. thresholding, ROI analysis).
Use the HDF5_BLS or brim export when you need to hand the data - spectra *and* fit
result together, with metadata - to someone using different Brillouin analysis
software, or to keep an interoperable long-term archive of a measurement rather
than a `brillouinpy`-specific pickle file. Between the two, brim is the
newer, more actively developed effort at a field-wide standard (with viewer
plugins for napari and Fiji, and the no-install BrimView web viewer), while
HDF5_BLS has its own, separate tooling ecosystem - which one to prefer depends on
what your collaborators already use.
