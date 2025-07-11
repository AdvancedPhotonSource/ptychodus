from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    CropCenter,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class LYNXDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/entry/data/eiger_4'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            data = h5_file[self._data_path]

            if not isinstance(data, h5py.Dataset):
                raise KeyError(f'{self._data_path} is not a Dataset!')

            num_patterns, detector_height, detector_width = data.shape

            crop_center: CropCenter | None = None

            # FIXME count_cutoff: int | None = None
            detector_distance_m: float | None = None
            # FIXME exposure_period: float | None = None
            # FIXME exposure_time: float | None = None
            probe_energy_eV: float | None = None  # noqa: N806
            detector_pixel_geometry: PixelGeometry | None = None
            # FIXME silicon: float | None = None
            # FIXME tau: float | None = None
            # FIXME image_nr_high: int | None = None
            # FIXME image_nr_low: int | None = None

            try:
                center_x_px: int = data.attrs['Center_x_pixel'].item()
                center_y_px: int = data.attrs['Center_y_pixel'].item()
                # FIXME count_cutoff = data.attrs['Count_cutoff'].item()
                detector_distance_m = data.attrs['Detector_distance_m'].item()
                # FIXME exposure period/time to metadata
                # FIXME exposure_period = data.attrs['Exposure_period'].item()
                # FIXME exposure_time = data.attrs['Exposure_time'].item()
                photon_energy_keV = data.attrs['Photon_energy_kev'].item()  # noqa: N806
                pixel_size = data.attrs['Pixel_size'].item()
                # FIXME silicon = data.attrs['Silicon'].item()
                # FIXME tau = data.attrs['Tau'].item()
                # FIXME image_nr_high = data.attrs['image_nr_high'].item()
                # FIXME image_nr_low = data.attrs['image_nr_low'].item()
            except KeyError:
                pass
            else:
                detector_pixel_geometry = PixelGeometry(pixel_size, pixel_size)
                crop_center = CropCenter(center_x_px, center_y_px)
                probe_energy_eV = 1000.0 * photon_energy_keV  # noqa: N806

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data.dtype,
                detector_distance_m=detector_distance_m,
                detector_extent=ImageExtent(detector_width, detector_height),
                detector_pixel_geometry=detector_pixel_geometry,
                # FIXME detector_bit_depth: int | None = None
                crop_center=crop_center,
                # FIXME probe_photon_count: int | None = None
                probe_energy_eV=probe_energy_eV,
                # FIXME tomography_angle_deg: float | None = None
                file_path=file_path,
            )
            contents_tree = self._tree_builder.build(h5_file)
            array = H5DiffractionPatternArray(
                label=file_path.stem,
                indexes=numpy.arange(num_patterns),
                file_path=file_path,
                data_path=self._data_path,
            )
            return SimpleDiffractionDataset(metadata, contents_tree, [array])


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        LYNXDiffractionFileReader(),
        simple_name='APS_LYNX',
        display_name='APS 31-ID-E LYNX Files (*.h5 *.hdf5)',
    )
