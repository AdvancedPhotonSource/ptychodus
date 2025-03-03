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
                return SimpleDiffractionDataset.createNullInstance(filePath)
            else:
                numberOfPatternsPerArray, detectorHeight, detectorWidth = h5data.shape
                metadata = DiffractionMetadata(
                    numberOfPatternsPerArray=numberOfPatternsPerArray,
                    numberOfPatternsTotal=numberOfPatternsPerArray * len(filePathMapping),
                    patternDataType=h5data.dtype,
                    detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                    filePath=filePath.parent / filePattern,
                )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, fp in sorted(filePathMapping.items()):
            indexes = numpy.arange(numberOfPatternsPerArray) + idx * numberOfPatternsPerArray
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, dataPath)
            contentsTree.createChild([array.getLabel(), 'HDF5', str(idx)])
            arrayList.append(array)

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        APS2IDDiffractionFileReader(),
        simpleName='APS_2ID',
        displayName='APS 2-ID Diffraction Files (*.h5 *.hdf5)',
    )
