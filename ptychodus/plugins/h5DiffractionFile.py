from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (DiffractionPatternArrayType, DiffractionDataset,
                                    DiffractionFileReader, DiffractionFileWriter,
                                    DiffractionMetadata, DiffractionPatternArray,
                                    DiffractionPatternState, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class H5DiffractionPatternArray(DiffractionPatternArray):

    def __init__(self, label: str, index: int, filePath: Path, dataPath: str) -> None:
        super().__init__()
        self._label = label
        self._index = index
        self._state = DiffractionPatternState.UNKNOWN
        self._filePath = filePath
        self._dataPath = dataPath

    def getLabel(self) -> str:
        return self._label

    def getIndex(self) -> int:
        return self._index

    def getState(self) -> DiffractionPatternState:
        return self._state

    def getData(self) -> DiffractionPatternArrayType:
        self._state = DiffractionPatternState.MISSING

        with h5py.File(self._filePath, 'r') as h5File:
            try:
                item = h5File[self._dataPath]
            except KeyError:
                raise ValueError(f'Symlink {self._filePath}:{self._dataPath} is broken!')
            else:
                if isinstance(item, h5py.Dataset):
                    self._state = DiffractionPatternState.FOUND
                else:
                    raise ValueError(
                        f'Symlink {self._filePath}:{self._dataPath} is not a dataset!')

            data = item[()]

        return data


class H5DiffractionFileTreeBuilder:

    def _addAttributes(self, treeNode: SimpleTreeNode,
                       attributeManager: h5py.AttributeManager) -> None:
        for name, value in attributeManager.items():
            if isinstance(value, str):
                itemDetails = f'STRING = "{value}"'
            elif isinstance(value, h5py.Empty):
                logger.debug(f'Skipping empty attribute {name}.')
            else:
                stringInfo = h5py.check_string_dtype(value.dtype)
                itemDetails = f'STRING = "{value.decode(stringInfo.encoding)}"' if stringInfo \
                        else f'SCALAR {value.dtype} = {value}'

            treeNode.createChild([str(name), 'Attribute', itemDetails])

    def createRootNode(self) -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

    def build(self, h5File: h5py.File) -> SimpleTreeNode:
        rootNode = self.createRootNode()
        unvisited = [(rootNode, h5File)]

        while unvisited:
            parentItem, h5Group = unvisited.pop()

            for itemName in h5Group:
                itemType = 'Unknown'
                itemDetails = ''
                h5Item = h5Group.get(itemName, getlink=True)

                treeNode = parentItem.createChild(list())

                if isinstance(h5Item, h5py.HardLink):
                    itemType = 'Hard Link'
                    h5Item = h5Group.get(itemName, getlink=False)

                    if isinstance(h5Item, h5py.Group):
                        itemType = 'Group'
                        self._addAttributes(treeNode, h5Item.attrs)
                        unvisited.append((treeNode, h5Item))
                    elif isinstance(h5Item, h5py.Dataset):
                        itemType = 'Dataset'
                        self._addAttributes(treeNode, h5Item.attrs)
                        spaceId = h5Item.id.get_space()

                        if spaceId.get_simple_extent_type() == h5py.h5s.SCALAR:
                            value = h5Item[()]

                            if isinstance(value, bytes):
                                itemDetails = value.decode()
                            else:
                                stringInfo = h5py.check_string_dtype(value.dtype)
                                itemDetails = f'STRING = "{value.decode(stringInfo.encoding)}"' \
                                        if stringInfo else f'SCALAR {value.dtype} = {value}'
                        else:
                            itemDetails = f'{h5Item.shape} {h5Item.dtype}'
                elif isinstance(h5Item, h5py.SoftLink):
                    itemType = 'Soft Link'
                    itemDetails = f'{h5Item.path}'
                elif isinstance(h5Item, h5py.ExternalLink):
                    itemType = 'External Link'
                    itemDetails = f'{h5Item.filename}/{h5Item.path}'
                else:
                    logger.debug(f'Unknown item "{itemName}"')

                treeNode.itemData = [itemName, itemType, itemDetails]

        return rootNode


class H5DiffractionFileReader(DiffractionFileReader):

    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.createNullInstance(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.warning('Unable to find data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath,
                    )

                array = H5DiffractionPatternArray(
                    label=filePath.stem,
                    index=0,
                    filePath=filePath,
                    dataPath=self._dataPath,
                )

                dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')

        return dataset


class H5DiffractionFileWriter(DiffractionFileWriter):

    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        data = numpy.concatenate([array.getData() for array in dataset])

        with h5py.File(filePath, 'w') as h5File:
            h5File.create_dataset(self._dataPath, data=data, compression='gzip')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        H5DiffractionFileReader(dataPath='/entry/data/data'),
        simpleName='HDF5',
        displayName='Hierarchical Data Format 5 Files (*.h5 *.hdf5)',
    )
    registry.diffractionFileReaders.registerPlugin(
        H5DiffractionFileReader(dataPath='/entry/measurement/Eiger/data'),
        simpleName='NanoMax',
        displayName='NanoMax Diffraction Files (*.h5 *.hdf5)',
    )
    registry.diffractionFileReaders.registerPlugin(
        H5DiffractionFileReader(dataPath='/dp'),
        simpleName='PtychoShelves',
        displayName='PtychoShelves Diffraction Files (*.h5 *.hdf5)',
    )
    registry.diffractionFileWriters.registerPlugin(
        H5DiffractionFileWriter(dataPath='/dp'),
        simpleName='PtychoShelves',
        displayName='PtychoShelves Diffraction Files (*.h5 *.hdf5)',
    )
