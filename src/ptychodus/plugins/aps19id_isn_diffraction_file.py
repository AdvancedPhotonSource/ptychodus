from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class ISNDiffractionFileReader(DiffractionFileReader):
    ONE_MILLIMETER_M: Final[float] = 1.0e-3

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)
            array_list: list[DiffractionArray] = []
            ptycho = h5_file['PTYCHO']

            if isinstance(ptycho, h5py.Group):
                for name in sorted(ptycho):
                    h5_item = ptycho.get(name, getlink=True)

                    if isinstance(h5_item, h5py.ExternalLink):
                        # FIXME BEGIN
                        offset = len(array_list) * metadata.num_patterns_per_array  # FIXME
                        data_path = '/entry/data/data'  # TODO str(h5_item.path)
                        array = H5DiffractionPatternArray(
                            label=name,
                            indexes=numpy.arange(metadata.num_patterns_per_array) + offset,
                            file_path=file_path.parent / h5_item.filename,
                            data_path=data_path,
                        )
                        # FIXME END
                        array_list.append(array)
                    else:
                        logger.debug(f'Skipping "{name}": not an external link.')
            else:
                raise KeyError('PTYCHO is not a group!')

            # FIXME num_patterns, detector_height, detector_width = ptycho.shape  # FIXME

            configs = h5_file['configs']
            detector_pixel_geometry: PixelGeometry | None = None
            probe_energy_eV: float | None = None  # noqa: N806

            if isinstance(configs, h5py.Group):
                num_images = int(configs['num_images'][()])  # FIXME
                num_exposures_per_image = int(configs['num_exposures_per_img'][()])  # FIXME

                detector_distance_m = configs['det_dist_mm'][()] * self.ONE_MILLIMETER_M
                detector_pixel_geometry = PixelGeometry(
                    configs['pix_size_x'][()], configs['pix_size_y'][()]
                )
                probe_energy_eV = configs['photon_energy_eV'][()]  # noqa: N806
            else:
                raise KeyError('configs is not a group!')

            metadata = DiffractionMetadata(
                num_patterns_per_array=num_patterns,  # FIXME num patterns -> list
                num_patterns_total=num_patterns,
                pattern_dtype=numpy.dtype('u4'),  # FIXME from data
                detector_distance_m=detector_distance_m,
                detector_extent=ImageExtent(detector_width, detector_height),
                detector_pixel_geometry=detector_pixel_geometry,
                probe_energy_eV=probe_energy_eV,
                file_path=file_path,
            )

            dataset = SimpleDiffractionDataset(metadata, contents_tree, array_list)

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ISNDiffractionFileReader(),
        simple_name='APS_ISN',
        display_name='APS 19-ID In-Situ Nanoprobe Files (*.h5 *.hdf5)',
    )
