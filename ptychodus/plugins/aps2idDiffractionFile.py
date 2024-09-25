from collections.abc import Mapping
from pathlib import Path
import logging
import re

import h5py

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

        digits = re.findall(r"\d+", filePath.stem)
        longest_digits = max(digits, key=len)
        filePattern = filePath.name.replace(longest_digits, f"(\\d{{{len(longest_digits)}}})")

        for fp in filePath.parent.iterdir():
            z = re.match(filePattern, fp.name)

            if z:
                index = int(z.group(1))
                filePathDict[index] = fp

        return filePathDict, filePattern

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)
        dataPath = "/entry/data/data"

        filePathMapping, filePattern = self._getFileSeries(filePath)
        contentsTree = SimpleTreeNode.createRoot(["Name", "Type", "Details"])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, fp in sorted(filePathMapping.items()):
            array = H5DiffractionPatternArray(fp.stem, idx, fp, dataPath)
            contentsTree.createChild([array.getLabel(), "HDF5", str(idx)])
            arrayList.append(array)

        try:
            with h5py.File(filePath, "r") as h5File:
                try:
                    h5data = h5File[dataPath]
                except KeyError:
                    logger.warning(f"File {filePath} is not an APS 2-ID data file.")
                else:
                    numberOfPatternsPerArray, detectorHeight, detectorWidth = h5data.shape
                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatternsPerArray,
                        numberOfPatternsTotal=numberOfPatternsPerArray * len(arrayList),
                        patternDataType=h5data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath.parent / filePattern,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        APS2IDDiffractionFileReader(),
        simpleName="APS_2ID",
        displayName="APS 2-ID Diffraction Files (*.h5 *.hdf5)",
    )
