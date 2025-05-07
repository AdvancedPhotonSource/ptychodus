from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
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

        try:
            with h5py.File(file_path, 'r') as h5_file:
                metadata = DiffractionMetadata.create_null(file_path)
                contents_tree = self._tree_builder.build(h5_file)
                array_list: list[DiffractionPatternArray] = []
                dataset = SimpleDiffractionDataset(metadata, contents_tree, array_list)

                try:
                    configs = h5_file['configs']
                    num_patterns_per_array = int(configs['num_images'][()])
                    num_patterns_total = 50 * num_patterns_per_array  # TODO generalize
                    detector_distance_mm = float(configs['det_dist_mm'])
                    detector_width = int(configs['det_size_x'][()])
                    detector_height = int(configs['det_size_y'][()])
                    pixel_width_m = float(configs['pix_size_x'][()])
                    pixel_height_m = float(configs['pix_size_y'][()])
                    probe_energy_eV = float(configs['photon_energy_eV'][()])  # noqa: N806
                except KeyError:
                    logger.warning('Unable to find metadata.')
                    return dataset

                metadata = DiffractionMetadata(
                    num_patterns_per_array=num_patterns_per_array,
                    num_patterns_total=num_patterns_total,
                    pattern_dtype=numpy.dtype('u4'),
                    detector_distance_m=self.ONE_MILLIMETER_M * detector_distance_mm,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    detector_pixel_geometry=PixelGeometry(pixel_width_m, pixel_height_m),
                    probe_energy_eV=probe_energy_eV,
                    file_path=file_path,
                )

                try:
                    ptycho = h5_file['PTYCHO']
                except KeyError:
                    logger.warning('Unable to find data.')
                    return dataset

                for name, h5_item in sorted(ptycho.items()):
                    h5_item = ptycho.get(name, getlink=True)

                    if isinstance(h5_item, h5py.ExternalLink):
                        offset = len(array_list) * metadata.num_patterns_per_array
                        data_path = '/entry/data/data'  # TODO str(h5_item.path)
                        array = H5DiffractionPatternArray(
                            label=name,
                            indexes=numpy.arange(num_patterns_per_array) + offset,
                            file_path=file_path.parent / h5_item.filename,
                            data_path=data_path,
                        )
                        array_list.append(array)
                    else:
                        logger.debug(f'Skipping "{name}": not an external link.')

                dataset = SimpleDiffractionDataset(metadata, contents_tree, array_list)
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ISNDiffractionFileReader(),
        simple_name='APS_ISN',
        display_name='APS ISN Files (*.h5 *.hdf5)',
    )
