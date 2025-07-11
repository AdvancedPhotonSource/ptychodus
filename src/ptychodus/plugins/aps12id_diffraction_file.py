from collections import defaultdict
from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

from .h5_diffraction_file import H5DiffractionPatternArray

logger = logging.getLogger(__name__)


class APS12IDDiffractionFileReader(DiffractionFileReader):
    DATA_PATH: Final[str] = '/entry/data/data'

    def read(self, file_path: Path) -> DiffractionDataset:
        stem_parts = file_path.stem.split('_')
        stem_prefix = '_'.join(stem_parts[:-2])
        logger.debug(f'{stem_prefix=}')
        scan_num = int(stem_parts[-3])
        logger.debug(f'{scan_num=}')

        lines: set[int] = set()
        points: set[int] = set()
        points_per_line: dict[int, set[int]] = defaultdict(set[int])
        file_dict: dict[tuple[int, int], Path] = dict()

        for p in file_path.parent.glob(f'{stem_prefix}_*{file_path.suffix}'):
            stem_parts = p.stem.split('_')
            line = int(stem_parts[-2])
            point = int(stem_parts[-1])

            lines.add(line)
            points.add(point)
            points_per_line[line].add(point)
            file_dict[line, point] = p

        logger.debug(f'{lines=}')
        lines_min = min(lines)
        lines_max = max(lines)
        lines_num = lines_max - lines_min + 1
        logger.debug(f'{points=}')
        points_min = min(points)
        points_max = max(points)
        points_num = points_max - points_min + 1

        for line, line_points in points_per_line.items():
            missing_points = points - line_points

            if missing_points:
                logger.warning(f'Line {line} is missing points {missing_points}')

        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Line', 'Point'])
        array_list: list[DiffractionArray] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                h5_data = h5_file[self.DATA_PATH]
            except KeyError:
                logger.warning(f'File "{file_path}" is not an APS 12-ID data file.')
                return SimpleDiffractionDataset.create_null(file_path)
            else:
                if not isinstance(h5_data, h5py.Dataset):
                    logger.warning(
                        f'Data path "{self.DATA_PATH}" in "{file_path}" is not a dataset.'
                    )
                    return SimpleDiffractionDataset.create_null(file_path)

                detector_height, detector_width = h5_data.shape

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[1] * lines_num * points_num,
                    pattern_dtype=h5_data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    file_path=file_path,
                )

        for (line, point), fp in file_dict.items():
            index = (point - points_min) + (line - lines_min) * points_num
            indexes = numpy.array([index])
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, self.DATA_PATH)
            contents_tree.create_child([array.get_label(), 'HDF5', str(line), str(point)])
            array_list.append(array)

        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDDiffractionFileReader(),
        simple_name='APS_PtychoSAXS',
        display_name='APS 12-ID PtychoSAXS Files (*.h5 *.hdf5)',
    )
