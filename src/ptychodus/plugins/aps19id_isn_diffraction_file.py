from pathlib import Path
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
    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        array_list: list[DiffractionArray] = []

        with h5py.File(file_path, 'r') as h5_file:
            ptycho = h5_file['PTYCHO']
            configs = h5_file['configs']
            num_patterns_per_array: list[int] = []

            if isinstance(ptycho, h5py.Group):
                data_path = '/entry/data/data'
                offset = 0

                for name in sorted(ptycho):
                    h5_item = ptycho.get(name, getlink=True)

                    if isinstance(h5_item, h5py.ExternalLink):
                        data_file_path = file_path.parent / h5_item.filename

                        with h5py.File(data_file_path) as h5_data_file:
                            h5_data = h5_data_file[data_path]

                            if isinstance(h5_data, h5py.Dataset):
                                num_patterns, _, _ = h5_data.shape
                                num_patterns_per_array.append(num_patterns)
                            else:
                                raise ValueError(
                                    f'Expected {data_file_path}:{data_path} to be a dataset.'
                                )

                        array = H5DiffractionPatternArray(
                            label=name,
                            indexes=numpy.arange(num_patterns) + offset,
                            file_path=data_file_path,
                            data_path=data_path,
                        )
                        array_list.append(array)
                        offset += num_patterns
                    else:
                        logger.debug(f'Skipping "{name}": not an external link.')
            else:
                raise KeyError('PTYCHO is not a group!')

            if isinstance(configs, h5py.Group):
                detector_distance_m = float(
                    configs['det_dist_mm'][()]
                )  # file units are m; typo in dataset name
                detector_width_px = int(configs['det_size_x'][()])
                detector_height_px = int(configs['det_size_y'][()])
                pixel_width_m = float(configs['pix_size_x'][()])
                pixel_height_m = float(configs['pix_size_y'][()])
                probe_energy_eV = float(configs['photon_energy_eV'][()])  # noqa: N806

                metadata = DiffractionMetadata(
                    num_patterns_per_array=num_patterns_per_array,
                    pattern_dtype=numpy.dtype(numpy.uint32),
                    detector_distance_m=detector_distance_m,
                    detector_extent=ImageExtent(detector_width_px, detector_height_px),
                    detector_pixel_geometry=PixelGeometry(pixel_width_m, pixel_height_m),
                    probe_energy_eV=probe_energy_eV,
                    file_path=file_path,
                )
            else:
                raise KeyError('configs is not a group!')

            contents_tree = self._tree_builder.build(h5_file)

            return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ISNDiffractionFileReader(),
        simple_name='APS_ISN',
        display_name='APS 19-ID In-Situ Nanoprobe Files (*.h5 *.hdf5)',
    )
