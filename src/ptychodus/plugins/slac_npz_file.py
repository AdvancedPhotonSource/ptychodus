from pathlib import Path
from typing import Final, Sequence
import logging

import numpy

from ptychodus.api.constants import ELECTRON_VOLT_J, LIGHT_SPEED_M_PER_S, PLANCK_CONSTANT_J_PER_HZ
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, Product, ProductFileReader, ProductMetadata
from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition

logger = logging.getLogger(__name__)

DEFAULT_DETECTOR_DISTANCE_M = 0.75
DEFAULT_PROBE_ENERGY_EV = 8000.0
DEFAULT_DETECTOR_PIXEL_SIZE_M = 75e-6
PIXEL_COORDINATE_THRESHOLD_M = 1e-3


def _get_scalar(npz_file: numpy.lib.npyio.NpzFile, key: str, fallback: float) -> float:
    if key not in npz_file.files:
        return fallback

    try:
        return float(npz_file[key])
    except (TypeError, ValueError):
        logger.warning('Failed to parse %s from NPZ; using default %s', key, fallback)
        return fallback


def _probe_wavelength_m(energy_ev: float) -> float:
    if energy_ev <= 0.0:
        return 0.0

    hc_jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S
    return hc_jm / (energy_ev * ELECTRON_VOLT_J)


def _coords_in_pixels(scan_x: numpy.ndarray, scan_y: numpy.ndarray) -> bool:
    max_abs = max(float(numpy.max(numpy.abs(scan_x))), float(numpy.max(numpy.abs(scan_y))))
    return max_abs > PIXEL_COORDINATE_THRESHOLD_M


class SLACDiffractionFileReader(DiffractionFileReader):
    def read(self, file_path: Path) -> DiffractionDataset:
        with numpy.load(file_path) as npz_file:
            patterns = numpy.transpose(npz_file['diffraction'], [2, 0, 1])

        num_patterns, detector_height, detector_width = patterns.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=patterns.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path,
        )

        contents_tree = DiffractionDatasetLayoutNode.create_root()
        contents_tree.add_child(
            file_path.stem, type(patterns).__name__, f'{patterns.dtype}{patterns.shape}'
        )

        array = SimpleDiffractionArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            patterns=patterns,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array])


class SLACProductFileReader(ProductFileReader):
    def read(self, file_path: Path) -> Product:
        with numpy.load(file_path) as npz_file:
            scan_x = numpy.asarray(npz_file['xcoords_start'])
            scan_y = numpy.asarray(npz_file['ycoords_start'])
            probe_array = npz_file['probeGuess']
            object_array = npz_file['objectGuess']

            detector_distance_m = _get_scalar(
                npz_file, 'detector_distance_m', DEFAULT_DETECTOR_DISTANCE_M
            )
            probe_energy_ev = _get_scalar(npz_file, 'probe_energy_eV', DEFAULT_PROBE_ENERGY_EV)
            if probe_energy_ev == DEFAULT_PROBE_ENERGY_EV:
                probe_energy_ev = _get_scalar(npz_file, 'probe_energy_ev', DEFAULT_PROBE_ENERGY_EV)
            detector_pixel_size_m = _get_scalar(
                npz_file, 'detector_pixel_size_m', DEFAULT_DETECTOR_PIXEL_SIZE_M
            )

        scan_x_m = scan_x
        scan_y_m = scan_y

        if _coords_in_pixels(scan_x, scan_y):
            width_px = float(probe_array.shape[-1])
            detector_width_m = detector_pixel_size_m * width_px
            pixel_size_m = 0.0
            if detector_width_m > 0.0:
                pixel_size_m = _probe_wavelength_m(probe_energy_ev) * detector_distance_m
                pixel_size_m /= detector_width_m

            if pixel_size_m > 0.0:
                scan_x_m = scan_x * pixel_size_m
                scan_y_m = scan_y * pixel_size_m
            else:
                logger.warning('SLAC NPZ coordinate scaling skipped (invalid pixel size).')

        metadata = ProductMetadata(
            name=file_path.stem,
            comments='',
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_ev,
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        point_list: list[ProbePosition] = list()

        for idx, (x_m, y_m) in enumerate(zip(scan_x_m, scan_y_m)):
            point = ProbePosition(idx, x_m, y_m)
            point_list.append(point)

        loss: Sequence[LossValue] = list()  # not included in file

        return Product(
            metadata=metadata,
            probe_positions=ProbePositionSequence(point_list),
            probes=ProbeSequence(array=probe_array, opr_weights=None, pixel_geometry=None),
            object_=Object(array=object_array, pixel_geometry=None, center=None),
            losses=loss,
        )


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC_NPZ'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'  # noqa: N806

    registry.diffraction_file_readers.register_plugin(
        SLACDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        SLACProductFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
