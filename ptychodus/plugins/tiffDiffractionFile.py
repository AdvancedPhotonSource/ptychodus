from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                                DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                                DiffractionMetadata, SimpleDiffractionArray,
                                SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'TIFF'

    @property
    def fileFilter(self) -> str:
        return 'Tagged Image File Format Files (*.tif *.tiff)'

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(filePath, 0, 0, 0)
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionArray] = list()

        if filePath:
            digits = re.findall(r'\d+', filePath.stem)
            longest_digits = max(digits, key=len)
            pattern = filePath.name.replace(longest_digits, f'\\d{{{len(longest_digits)}}}')
            numberOfImagesTotal = 0

            for fp in filePath.parent.iterdir():
                if re.search(pattern, fp.name):
                    with TiffFile(fp) as tiff:
                        numberOfImagesTotal += len(tiff.pages)

                        itemName = fp.stem
                        itemType = 'TIFF'
                        itemDetails = str(tiff.epics_metadata)
                        contentsTree.createChild([itemName, itemType, itemDetails])

                    index = 0  # FIXME from digits
                    dataOffset = 0  # FIXME see numberOfImagesTotal
                    data = tiff.asarray()

                    if data.ndim == 2:
                        data = data[numpy.newaxis, :, :]

                    array = SimpleDiffractionArray(fp.stem, index, dataOffset, data)
                    arrayList.append(array)

            with TiffFile(filePath) as tiff:  # FIXME not needed
                data = tiff.asarray()
                imageWidth = data.shape[-1]
                imageHeight = data.shape[-2]

            metadata = DiffractionMetadata(
                filePath=filePath.parent / pattern,
                imageWidth=imageWidth,
                imageHeight=imageHeight,
                numberOfImagesPerArray=0,  # FIXME
                numberOfImagesTotal=numberOfImagesTotal,
            )

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(TiffDiffractionFileReader())


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiffFile = reader.read(filePath)
    print(tiffFile)
