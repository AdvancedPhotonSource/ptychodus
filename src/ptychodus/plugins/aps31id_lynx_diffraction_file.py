from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.constants import ONE_KILOELECTRONVOLT_EV
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    CropCenter,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)

_LEN_TO_M: dict[str, float] = {
    'm': 1.0,
    'meter': 1.0,
    'meters': 1.0,
    'cm': 1e-2,
    'centimeter': 1e-2,
    'centimeters': 1e-2,
    'mm': 1e-3,
    'millimeter': 1e-3,
    'millimeters': 1e-3,
    'um': 1e-6,
    'µm': 1e-6,
    'micron': 1e-6,
    'microns': 1e-6,
    'micrometer': 1e-6,
    'micrometers': 1e-6,
    'micrometre': 1e-6,
    'micrometres': 1e-6,
    'nm': 1e-9,
    'nanometer': 1e-9,
    'nanometers': 1e-9,
}

_E_TO_EV: dict[str, float] = {
    'ev': 1.0,
    'electronvolt': 1.0,
    'electronvolts': 1.0,
    'kev': 1e3,
    'kiloelectronvolt': 1e3,
    'kiloelectronvolts': 1e3,
    'mev': 1e6,
    'megaelectronvolt': 1e6,
    'megaelectronvolts': 1e6,
}


def _read_egu(h5_file: h5py.File, egu_path: str, fallback: str) -> str:
    if egu_path not in h5_file:
        logger.warning(f'{egu_path} not present; assuming {fallback!r}')
        return fallback
    raw = h5_file[egu_path][()]
    if isinstance(raw, numpy.ndarray):
        if raw.size == 0:
            return fallback
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')
    return str(raw).strip().lower()


def _to_meters(value: float, egu: str) -> float:
    try:
        return value * _LEN_TO_M[egu]
    except KeyError:
        logger.warning(f'Unrecognized length unit {egu!r}; assuming meters')
        return value


def _to_ev(value: float, egu: str) -> float:
    try:
        return value * _E_TO_EV[egu]
    except KeyError:
        logger.warning(f'Unrecognized energy unit {egu!r}; assuming eV')
        return value


class LYNXDiffractionFileReader(DiffractionFileReader):
    _LEGACY_DATA_PATH = '/entry/data/eiger_4'
    _AD_DATA_PATH = '/entry/data/data'

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

            if self._AD_DATA_PATH in h5_file:
                return self._read_area_detector(file_path, h5_file, contents_tree)
            if self._LEGACY_DATA_PATH in h5_file:
                return self._read_legacy(file_path, h5_file, contents_tree)

            raise KeyError(
                f'No LYNX diffraction data found in {file_path} '
                f'(expected {self._AD_DATA_PATH} or {self._LEGACY_DATA_PATH})'
            )

    def _read_legacy(
        self, file_path: Path, h5_file: h5py.File, contents_tree: DiffractionDatasetLayoutNode
    ) -> DiffractionDataset:
        data = h5_file[self._LEGACY_DATA_PATH]

        if not isinstance(data, h5py.Dataset):
            raise KeyError(f'{self._LEGACY_DATA_PATH} is not a Dataset!')

        num_patterns, detector_height, detector_width = data.shape

        crop_center: CropCenter | None = None
        detector_distance_m: float | None = None
        detector_pixel_geometry: PixelGeometry | None = None
        exposure_time_s: float | None = None
        probe_energy_eV: float | None = None  # noqa: N806

        try:
            center_x_px: int = data.attrs['Center_x_pixel'].item()
            center_y_px: int = data.attrs['Center_y_pixel'].item()
            detector_distance_m = data.attrs['Detector_distance_m'].item()
            exposure_time_s = data.attrs['Exposure_time'].item()
            photon_energy_keV = data.attrs['Photon_energy_kev'].item()  # noqa: N806
            pixel_size = data.attrs['Pixel_size'].item()
        except KeyError:
            pass
        else:
            crop_center = CropCenter(center_x_px, center_y_px)
            detector_pixel_geometry = PixelGeometry(pixel_size, pixel_size)
            probe_energy_eV = ONE_KILOELECTRONVOLT_EV * photon_energy_keV  # noqa: N806

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=data.dtype,
            detector_distance_m=detector_distance_m,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_pixel_geometry=detector_pixel_geometry,
            crop_center=crop_center,
            probe_energy_eV=probe_energy_eV,
            exposure_time_s=exposure_time_s,
            file_path=file_path,
        )
        array = H5DiffractionPatternArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            file_path=file_path,
            data_path=self._LEGACY_DATA_PATH,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, [array])

    def _read_area_detector(
        self, file_path: Path, h5_file: h5py.File, contents_tree: DiffractionDatasetLayoutNode
    ) -> DiffractionDataset:
        data = h5_file[self._AD_DATA_PATH]

        if not isinstance(data, h5py.Dataset):
            raise KeyError(f'{self._AD_DATA_PATH} is not a Dataset!')

        num_patterns, detector_height, detector_width = data.shape

        crop_center: CropCenter | None = None
        detector_distance_m: float | None = None
        detector_pixel_geometry: PixelGeometry | None = None
        exposure_time_s: float | None = None
        probe_energy_eV: float | None = None  # noqa: N806

        try:
            distance_raw = float(h5_file['/entry/instrument/detector/detector_distance'][0])
            x_pixel_raw = float(h5_file['/entry/instrument/detector/x_pixel_size'][0])
            y_pixel_raw = float(h5_file['/entry/instrument/detector/y_pixel_size'][0])
            energy_raw = float(h5_file['/entry/instrument/monochromator/energy'][0])
            exposure_raw = float(h5_file['/entry/instrument/detector/count_time'][0])
            beam_center_x = float(h5_file['/entry/instrument/detector/beam_center_x'][0])
            beam_center_y = float(h5_file['/entry/instrument/detector/beam_center_y'][0])
        except KeyError:
            pass
        else:
            distance_egu = _read_egu(h5_file, '/entry/metadata_units/Distance_EGU', 'mm')
            energy_egu = _read_egu(h5_file, '/entry/metadata_units/MonoEnergy_EGU', 'kev')
            pixel_egu = _read_egu(h5_file, '/entry/metadata_units/PixelSize_EGU', 'um')

            detector_distance_m = _to_meters(distance_raw, distance_egu)
            detector_pixel_geometry = PixelGeometry(
                width_m=_to_meters(x_pixel_raw, pixel_egu),
                height_m=_to_meters(y_pixel_raw, pixel_egu),
            )
            probe_energy_eV = _to_ev(energy_raw, energy_egu)  # noqa: N806
            exposure_time_s = exposure_raw
            crop_center = CropCenter(int(round(beam_center_x)), int(round(beam_center_y)))

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=data.dtype,
            detector_distance_m=detector_distance_m,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_pixel_geometry=detector_pixel_geometry,
            crop_center=crop_center,
            probe_energy_eV=probe_energy_eV,
            exposure_time_s=exposure_time_s,
            file_path=file_path,
        )
        array = H5DiffractionPatternArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            file_path=file_path,
            data_path=self._AD_DATA_PATH,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, [array])


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        LYNXDiffractionFileReader(),
        simple_name='APS_LYNX',
        display_name='APS 31-ID-E LYNX Files (*.h5 *.hdf5)',
    )
