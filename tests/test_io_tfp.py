import json
import os

import numpy as np
import pytest

from conftest import write_dat_file
from brillouinpy import utils
from brillouinpy.io import tfp

# prepare_brillouin_data hardcodes Raman's spectral_dimension to 2048, regardless
# of what's actually in the file.
_RAMAN_LENGTH = 2048

NESTED_META = {
    "Brillouin": {
        "Mirror_spacing_mm": 6.0,
        "Scan_amplitude_nm": 480,
        "Scanning_cycles": 15,
        "Input_pinhole_um": "150",
        "Output_pinhole_um": "700",
    },
    "Laser": {"Model": "Cobolt Samba", "Power_mW": 1.0, "Wavelength_nm": 532.1},
    "General": {"Date": "2025-01-01T00:00:00", "Sample": "Test"},
}

FLAT_META_WITH_AMPLITUDE = {
    "FP1_spacing_mm": 6,
    "FP_scan_amplitude": 480,
    "Laser_wavelenght_nm": 532.1,
    "Sample": "Test",
}

FLAT_META_WITHOUT_AMPLITUDE = {
    "FP1_spacing_mm": 6,
    "Laser_wavelenght_nm": 532.1,
    "Sample": "Test",
}

UNRECOGNISED_META = {"Foo": "bar"}


def _write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_read_meta_finds_file_in_data_subfolder(tmp_path):
    (tmp_path / "data").mkdir()
    _write_meta(tmp_path / "data" / "META.json", NESTED_META)

    assert tfp.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_finds_file_directly_in_project_path(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    assert tfp.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_prefers_data_subfolder_over_project_root(tmp_path):
    (tmp_path / "data").mkdir()
    _write_meta(tmp_path / "data" / "META.json", NESTED_META)
    _write_meta(tmp_path / "META.json", FLAT_META_WITH_AMPLITUDE)

    assert tfp.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        tfp.read_meta(str(tmp_path))


@pytest.mark.parametrize(
    "meta, expected_mirror_spacing_m, expected_scan_amplitude_m, expected_wavelength_m",
    [
        (NESTED_META, 6e-3, 480e-9, 532.1e-9),
        (FLAT_META_WITH_AMPLITUDE, 6e-3, 480e-9, 532.1e-9),
    ],
)
def test_brillouin_scan_parameters_from_meta(
    meta, expected_mirror_spacing_m, expected_scan_amplitude_m, expected_wavelength_m
):
    mirror_spacing, scan_amplitude, laser_wavelength = tfp._brillouin_scan_parameters_from_meta(meta)

    assert mirror_spacing == pytest.approx(expected_mirror_spacing_m)
    assert scan_amplitude == pytest.approx(expected_scan_amplitude_m)
    assert laser_wavelength == pytest.approx(expected_wavelength_m)


def test_brillouin_scan_parameters_from_meta_flat_without_amplitude_raises():
    with pytest.raises(KeyError, match="FP_scan_amplitude"):
        tfp._brillouin_scan_parameters_from_meta(FLAT_META_WITHOUT_AMPLITUDE)


def test_brillouin_scan_parameters_from_meta_unrecognised_schema_raises():
    with pytest.raises(KeyError, match="Unrecognised META.json schema"):
        tfp._brillouin_scan_parameters_from_meta(UNRECOGNISED_META)


@pytest.mark.parametrize("no_of_channels", [255, 256, 512, 1024])
def test_brillouin_spectral_axis_channel_width_is_scan_range_over_n(no_of_channels):
    axis = utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=no_of_channels,
        laser_wavelength=532.1e-9,
    )
    half_range_GHz = utils.fsr(6e-3) * 480e-9 / 532.1e-9 * 1e-9

    assert axis.shape == (no_of_channels,)
    assert np.allclose(np.diff(axis), 2 * half_range_GHz / no_of_channels)
    assert np.allclose(axis, -axis[::-1])
    # Channel centres sit half a channel inside the scan limits.
    assert axis[-1] == pytest.approx(half_range_GHz * (no_of_channels - 1) / no_of_channels)


def test_brillouin_spectral_axis_from_meta_matches_manual_call(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    axis = tfp.brillouin_spectral_axis_from_meta(str(tmp_path), no_of_channels=200)
    expected = utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=200, laser_wavelength=532.1e-9
    )

    assert np.allclose(axis, expected)


def test_deprecated_utils_read_meta_still_works_and_warns(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    with pytest.deprecated_call():
        result = utils.read_meta(str(tmp_path))

    assert result == NESTED_META


def test_deprecated_utils_brillouin_spectral_axis_from_meta_still_works_and_warns(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    with pytest.deprecated_call():
        axis = utils.brillouin_spectral_axis_from_meta(str(tmp_path), no_of_channels=200)

    expected = tfp.brillouin_spectral_axis_from_meta(str(tmp_path), no_of_channels=200)
    assert np.allclose(axis, expected)


def _write_raman_csv(directory, x, y, z, t, spectrum):
    # prepare_brillouin_data's Raman/.csv path: np.loadtxt(f, delimiter=',',
    # skiprows=1), then spectrum = data[1] (second row after the header).
    arr = np.zeros((2, len(spectrum)))
    arr[1] = spectrum
    path = os.path.join(directory, f"Raman_{x}_{y}_{z}_{t}.csv")
    with open(path, "w") as f:
        f.write("header\n")
    with open(path, "a") as f:
        np.savetxt(f, arr, delimiter=",")
    return path


@pytest.mark.parametrize("spectral_data_type,prefix", [("Brillouin", "Bri"), ("Raman", "Raman")])
def test_extract_coordinates_matches(spectral_data_type, prefix):
    filename = f"/some/dir/{prefix}_1_2_3_4.ext"
    assert tfp.extract_coordinates(filename, spectral_data_type) == (1, 2, 3, 4)


def test_extract_coordinates_no_match_returns_none():
    assert tfp.extract_coordinates("/some/dir/not_matching.DAT", "Brillouin") is None
    # a Brillouin-named file doesn't match the Raman pattern either
    assert tfp.extract_coordinates("/some/dir/Bri_1_2_3_4.DAT", "Raman") is None


def test_import_dat_file_reads_data_after_header(tmp_path):
    path = write_dat_file(str(tmp_path), "Bri", 0, 0, 0, 0, [1.0, 2.0, 3.5])

    data, n = tfp.import_DAT_File(path)

    assert n == 3
    assert np.allclose(data, [1.0, 2.0, 3.5])


def test_prepare_brillouin_data_unknown_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Spectral data not supported"):
        tfp.prepare_brillouin_data(str(tmp_path), "Foo")


def test_prepare_brillouin_data_brillouin_grid_shape_and_masked_gap(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    values = {(0, 0): [1.0, 2.0, 3.0], (1, 0): [4.0, 5.0, 6.0], (0, 1): [7.0, 8.0, 9.0]}
    # (1, 1) is deliberately not written - a missing point in an otherwise full 2x2 grid.
    for (x, y), v in values.items():
        write_dat_file(str(data_dir), "Bri", x, y, 0, 0, v)

    with pytest.warns(UserWarning, match="do not match"):
        result = tfp.prepare_brillouin_data(str(tmp_path), "Brillouin")

    assert result.shape == (2, 2, 1, 1, 3)
    for (x, y), v in values.items():
        assert not np.ma.is_masked(result[x, y, 0, 0])
        assert np.allclose(result[x, y, 0, 0], v)
    assert np.ma.is_masked(result[1, 1, 0, 0])


def test_prepare_brillouin_data_supports_multiple_z_layers(tmp_path):
    # Unlike io.legacy's loaders, this one isn't restricted to a single z-layer.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_dat_file(str(data_dir), "Bri", 0, 0, 0, 0, [1.0, 2.0])
    write_dat_file(str(data_dir), "Bri", 0, 0, 1, 0, [3.0, 4.0])

    result = tfp.prepare_brillouin_data(str(tmp_path), "Brillouin")

    assert result.shape == (1, 1, 2, 1, 2)
    assert np.allclose(result[0, 0, 0, 0], [1.0, 2.0])
    assert np.allclose(result[0, 0, 1, 0], [3.0, 4.0])


def test_prepare_brillouin_data_raman_full_grid(tmp_path, recwarn):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    spectrum_00 = np.linspace(0, 1, _RAMAN_LENGTH)
    spectrum_10 = np.linspace(1, 2, _RAMAN_LENGTH)
    _write_raman_csv(str(data_dir), 0, 0, 0, 0, spectrum_00)
    _write_raman_csv(str(data_dir), 1, 0, 0, 0, spectrum_10)

    result = tfp.prepare_brillouin_data(str(tmp_path), "Raman")

    assert not any("do not match" in str(w.message) for w in recwarn.list)
    assert result.shape == (2, 1, 1, 1, _RAMAN_LENGTH)
    assert np.allclose(result[0, 0, 0, 0], spectrum_00)
    assert np.allclose(result[1, 0, 0, 0], spectrum_10)
