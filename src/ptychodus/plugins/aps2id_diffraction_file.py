from collections.abc import Mapping
from pathlib import Path
import logging
import re

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

from .h5_diffraction_file import H5DiffractionPatternArray

logger = logging.getLogger(__name__)


class APS2IDDiffractionFileReader(DiffractionFileReader):
    def _getFileSeries(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        file_pathDict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', file_path.stem)
        longest_digits = max(digits, key=len)
        filePattern = file_path.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in file_path.parent.iterdir():
            z = re.match(filePattern, fp.name)

            if z:
                index = int(z.group(1))
                file_pathDict[index] = fp

        return file_pathDict, filePattern

    def read(self, file_path: Path) -> DiffractionDataset:
        file_pathMapping, filePattern = self._getFileSeries(file_path)
        data_path = '/entry/data/data'

        with h5py.File(file_path, 'r') as h5File:
            try:
                h5data = h5File[data_path]
            except KeyError:
                logger.warning(f'File {file_path} is not an APS 2-ID data file.')
                return SimpleDiffractionDataset.create_null(file_path)
            else:
                numberOfPatternsPerArray, detectorHeight, detectorWidth = h5data.shape
                metadata = DiffractionMetadata(
                    num_patterns_per_array=numberOfPatternsPerArray,
                    num_patterns_total=numberOfPatternsPerArray * len(file_pathMapping),
                    pattern_dtype=h5data.dtype,
                    detector_extent=ImageExtent(detectorWidth, detectorHeight),
                    file_path=file_path.parent / filePattern,
                )

        contentsTree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, fp in sorted(file_pathMapping.items()):
            indexes = numpy.arange(numberOfPatternsPerArray) + idx * numberOfPatternsPerArray
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, data_path)
            contentsTree.create_child([array.get_label(), 'HDF5', str(idx)])
            arrayList.append(array)

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2ID',
        display_name='APS 2-ID Files (*.h5 *.hdf5)',
    )
