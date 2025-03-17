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

from .h5DiffractionFile import H5DiffractionPatternArray

logger = logging.getLogger(__name__)


class APS2IDDiffractionFileReader(DiffractionFileReader):
    def _getFileSeries(self, filePath: Path) -> tuple[Mapping[int, Path], str]:
        filePathDict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', filePath.stem)
        longest_digits = max(digits, key=len)
        filePattern = filePath.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in filePath.parent.iterdir():
            z = re.match(filePattern, fp.name)

            if z:
                index = int(z.group(1))
                filePathDict[index] = fp

        return filePathDict, filePattern

    def read(self, filePath: Path) -> DiffractionDataset:
        filePathMapping, filePattern = self._getFileSeries(filePath)
        dataPath = '/entry/data/data'

        with h5py.File(filePath, 'r') as h5File:
            try:
                h5data = h5File[dataPath]
            except KeyError:
                logger.warning(f'File {filePath} is not an APS 2-ID data file.')
                return SimpleDiffractionDataset.create_null(filePath)
            else:
                numberOfPatternsPerArray, detectorHeight, detectorWidth = h5data.shape
                metadata = DiffractionMetadata(
                    num_patterns_per_array=numberOfPatternsPerArray,
                    num_patterns_total=numberOfPatternsPerArray * len(filePathMapping),
                    pattern_dtype=h5data.dtype,
                    detector_extent=ImageExtent(detectorWidth, detectorHeight),
                    file_path=filePath.parent / filePattern,
                )

        contentsTree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, fp in sorted(filePathMapping.items()):
            indexes = numpy.arange(numberOfPatternsPerArray) + idx * numberOfPatternsPerArray
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, dataPath)
            contentsTree.create_child([array.get_label(), 'HDF5', str(idx)])
            arrayList.append(array)

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2ID',
        display_name='APS 2-ID Diffraction Files (*.h5 *.hdf5)',
    )
