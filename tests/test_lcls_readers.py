"""Readers for LCLS XPP psana smalldata files.

The fixture is a shrunk copy of the real layout: a calibrated image stack in keV, its
image-grid mask, the run's photon energy and detector distance, piezo-stage channels, the
ipm2 monitor, and the X-ray and damage flags. Each rejected shot fails exactly one rule.
"""

from __future__ import annotations

from pathlib import Path
import math

import h5py
import numpy
import pytest

from ptychodus.api.diffraction import CropRegion
from ptychodus.api.plugins import PluginRegistry
from ptychodus.plugins.lcls_file_readers import (
    LCLSDiffractionFileReader,
    LCLSPositionFileReader,
    LCLSShotSelection,
)

PHOTON_ENERGY_KEV = 8.872
NUM_SHOTS = 10
HEIGHT = 6
WIDTH = 5

# shot: 0 kept; 1 X-rays off; 2 ipm2 not recorded; 3 detector not recorded; 4 no beam;
# 5 above the window; 6 below it; 7 stage not finite; 8 at the lower bound; 9 just inside
IPM2 = numpy.array([2500.0, 2500.0, 2500.0, 2500.0, -10.0, 3500.0, 1500.0, 2500.0, 2000.0, 2999.0])
KEPT = [0, 8, 9]
KEPT_IN_WIDE_WINDOW = [0, 5, 6, 8, 9]


def _photons() -> numpy.ndarray:
    rng = numpy.random.default_rng(0)
    return rng.integers(0, 6, size=(NUM_SHOTS, HEIGHT, WIDTH)).astype(numpy.float32)


def _mask() -> numpy.ndarray:
    mask = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.int64)
    mask[0, :] = 1
    mask[3, 2] = 1
    return mask


def _stage_um() -> numpy.ndarray:
    shots = numpy.arange(NUM_SHOTS, dtype=float)
    stage = numpy.stack([shots, 100.0 + shots, -2.0 * shots])
    stage[1, 7] = numpy.nan
    return stage


def _write(
    path: Path,
    *,
    with_mask: bool = True,
    with_distance: bool = True,
    with_energy: bool = True,
) -> Path:
    xray = numpy.ones(NUM_SHOTS, dtype=numpy.int64)
    xray[1] = 0
    ipm2_present = numpy.ones(NUM_SHOTS, dtype=numpy.int64)
    ipm2_present[2] = 0
    detector_present = numpy.ones(NUM_SHOTS, dtype=numpy.int64)
    detector_present[3] = 0

    with h5py.File(path, 'w') as h5_file:
        h5_file['/jungfrau1M/image_img'] = _photons() * numpy.float32(PHOTON_ENERGY_KEV)
        h5_file['/UserDataCfg/jungfrau1M/pixelsize'] = numpy.array([75e-6])

        if with_mask:
            h5_file['/UserDataCfg/jungfrau1M/image__image_mask_img'] = _mask()

        if with_energy:
            h5_file['/epics/lom_E'] = numpy.full(NUM_SHOTS, PHOTON_ENERGY_KEV)

        if with_distance:
            h5_file['/epicsUser/sample_det_dis'] = numpy.full(NUM_SHOTS, 4050.0)

        for channel, values in zip(('ch03', 'ch04', 'ch05'), _stage_um()):
            h5_file[f'/lmc/{channel}'] = values

        h5_file['/ipm2/sum'] = IPM2
        h5_file['/lightStatus/xray'] = xray
        h5_file['/damage/ipm2'] = ipm2_present
        h5_file['/damage/jungfrau1M'] = detector_present

    return path


def test_patterns_and_positions_cover_the_same_shots(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5')

    dataset = LCLSDiffractionFileReader().read(file_path)
    positions = LCLSPositionFileReader().read(file_path)

    assert len(dataset) == 1
    numpy.testing.assert_array_equal(dataset[0].get_indexes(), KEPT)
    numpy.testing.assert_array_equal(positions.get_indexes(), KEPT)
    assert dataset.get_metadata().num_patterns_per_array == [len(KEPT)]


def test_patterns_are_photons(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5')
    array = LCLSDiffractionFileReader().read(file_path)[0]

    patterns = array.get_patterns()

    assert patterns.dtype == numpy.float32
    numpy.testing.assert_allclose(patterns, _photons()[KEPT], rtol=1e-6, atol=1e-6)

    region = CropRegion(x_range=(1, 4), y_range=(2, 5))
    cropped = array.get_patterns(read_region=region)
    numpy.testing.assert_allclose(cropped, _photons()[KEPT, 2:5, 1:4], rtol=1e-6, atol=1e-6)


def test_bad_pixels_and_metadata(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5')
    dataset = LCLSDiffractionFileReader().read(file_path)
    metadata = dataset.get_metadata()

    numpy.testing.assert_array_equal(dataset.get_bad_pixels(), _mask() != 0)
    assert metadata.detector_extent.width_px == WIDTH
    assert metadata.detector_extent.height_px == HEIGHT
    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(75e-6)
    assert metadata.detector_pixel_geometry.height_m == pytest.approx(75e-6)
    assert metadata.detector_distance_m == pytest.approx(4.05)
    assert metadata.photon_energy_eV == pytest.approx(1000.0 * PHOTON_ENERGY_KEV)


def test_mask_and_distance_are_optional(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5', with_mask=False, with_distance=False)
    dataset = LCLSDiffractionFileReader().read(file_path)

    assert not dataset.get_bad_pixels().any()
    assert dataset.get_metadata().detector_distance_m is None


def test_photon_energy_is_required(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5', with_energy=False)

    with pytest.raises(ValueError, match='photon energy'):
        LCLSDiffractionFileReader().read(file_path)


@pytest.mark.parametrize('tomography_angle_deg', [180.0, 30.0])
def test_positions_carry_i0(tmp_path: Path, tomography_angle_deg: float) -> None:
    file_path = _write(tmp_path / 'run.h5')
    positions = LCLSPositionFileReader(tomography_angle_deg=tomography_angle_deg).read(file_path)

    stage_x_um, stage_y_um, stage_z_um = _stage_um()[:, KEPT]
    angle_rad = math.radians(tomography_angle_deg)
    expected_x_m = (math.cos(angle_rad) * stage_x_um + math.sin(angle_rad) * stage_y_um) * 1e-6

    numpy.testing.assert_allclose(positions.get_coordinates_x_m(), expected_x_m)
    numpy.testing.assert_allclose(positions.get_coordinates_y_m(), -stage_z_um * 1e-6)
    counts = positions.get_i0()
    assert counts is not None
    numpy.testing.assert_array_equal(counts, IPM2[KEPT])


def test_both_readers_follow_a_custom_window(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5')
    selection = LCLSShotSelection(i0_low_threshold=-math.inf, i0_high_threshold=math.inf)

    dataset = LCLSDiffractionFileReader(selection).read(file_path)
    positions = LCLSPositionFileReader(selection).read(file_path)

    # a non-positive I0 is no beam, whatever the window
    numpy.testing.assert_array_equal(dataset[0].get_indexes(), KEPT_IN_WIDE_WINDOW)
    numpy.testing.assert_array_equal(positions.get_indexes(), KEPT_IN_WIDE_WINDOW)


def test_no_surviving_shot_raises(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'run.h5')
    selection = LCLSShotSelection(i0_low_threshold=5000.0, i0_high_threshold=6000.0)

    with pytest.raises(ValueError, match='No shot passes'):
        LCLSPositionFileReader(selection).read(file_path)


def test_readers_are_registered() -> None:
    registry = PluginRegistry.load_plugins()

    diffraction = registry.diffraction_file_readers.find_plugin(LCLSPositionFileReader.SIMPLE_NAME)
    positions = registry.probe_position_file_readers.find_plugin(LCLSPositionFileReader.SIMPLE_NAME)

    assert diffraction is not None
    assert isinstance(diffraction.strategy, LCLSDiffractionFileReader)
    assert positions is not None
    assert isinstance(positions.strategy, LCLSPositionFileReader)
