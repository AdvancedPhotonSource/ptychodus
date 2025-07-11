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
        array_list: list[DiffractionArray] = []

        with h5py.File(file_path, 'r') as h5_file:
            configs = h5_file['configs']

            if isinstance(configs, h5py.Group):
                num_images = int(configs['num_images'][()])
                num_exposures_per_image = int(configs['num_exposures_per_img'][()])

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_exposures_per_image] * num_images,
                    pattern_dtype=numpy.dtype('u4'),  # TODO from data
                    detector_distance_m=float(configs['det_dist_mm'][()]) * self.ONE_MILLIMETER_M,
                    detector_extent=ImageExtent(
                        int(configs['det_size_x'][()]), int(configs['det_size_y'][()])
                    ),
                    detector_pixel_geometry=PixelGeometry(
                        float(configs['pix_size_x'][()]), float(configs['pix_size_y'][()])
                    ),
                    probe_energy_eV=float(configs['photon_energy_eV'][()]),  # noqa: N806
                    file_path=file_path,
                )
            else:
                raise KeyError('configs is not a group!')

            contents_tree = self._tree_builder.build(h5_file)
            ptycho = h5_file['PTYCHO']

            if isinstance(ptycho, h5py.Group):
                offset = 0

                for name in sorted(ptycho):
                    h5_item = ptycho.get(name, getlink=True)

                    if isinstance(h5_item, h5py.ExternalLink):
                        data_path = '/entry/data/data'  # TODO str(h5_item.path)
                        array = H5DiffractionPatternArray(
                            label=name,
                            indexes=numpy.arange(num_exposures_per_image) + offset,
                            file_path=file_path.parent / h5_item.filename,
                            data_path=data_path,
                        )
                        array_list.append(array)
                        offset += num_exposures_per_image
                    else:
                        logger.debug(f'Skipping "{name}": not an external link.')
            else:
                raise KeyError('PTYCHO is not a group!')

            return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ISNDiffractionFileReader(),
        simple_name='APS_ISN',
        display_name='APS 19-ID In-Situ Nanoprobe Files (*.h5 *.hdf5)',
    )
