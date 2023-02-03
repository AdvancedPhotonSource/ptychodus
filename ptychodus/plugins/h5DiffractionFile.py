from pathlib import Path
import logging

try:
    # NOTE must import hdf5plugin before h5py
    import hdf5plugin
except ModuleNotFoundError:
    pass

import h5py
import numpy

from ptychodus.api.data import (DiffractionPatternData, DiffractionDataset, DiffractionFileReader,
                                DiffractionMetadata, DiffractionPatternArray,
                                DiffractionPatternState, SimpleDiffractionDataset)
from ptychodus.api.observer import Observable
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class H5DiffractionPatternArray(DiffractionPatternArray):

    def __init__(self, data: DiffractionPatternData, label: str) -> None:
        super().__init__()
        self._data = data
        self._label = label

    def getLabel(self) -> str:
        return self._label

    def getIndex(self) -> int:
        return 0

    def getState(self) -> DiffractionPatternState:
        return DiffractionPatternState.FOUND

    def getData(self) -> DiffractionPatternData:
        return self._data


class H5DiffractionFileTreeBuilder:

    def _addAttributes(self, treeNode: SimpleTreeNode,
                       attributeManager: h5py.AttributeManager) -> None:
        for name, value in attributeManager.items():
            stringInfo = h5py.check_string_dtype(value.dtype)
            valueStr = None

            if stringInfo:
                valueStr = f'STRING = "{value.decode(stringInfo.encoding)}"'
            elif numpy.ndim(value) == 0:
                valueStr = f'SCALAR {value.dtype} = {value}'
            elif isinstance(value, numpy.ndarray):
                valueStr = f'{value.shape} {value.dtype}'
            else:
                logger.debug(f'UNKNOWN: {value} {type(value)}')
                valueStr = 'UNKNOWN'

            treeNode.createChild([str(name), 'Attribute', valueStr])

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
                            stringInfo = h5py.check_string_dtype(value.dtype)

                            if stringInfo:
                                valueStr = f'STRING = "{value.decode(stringInfo.encoding)}"'
                            elif numpy.ndim(value) == 0:
                                valueStr = f'SCALAR {value.dtype} = {value}'
                            else:
                                logger.debug(f'UNKNOWN: {value} {type(value)}')
                                valueStr = 'UNKNOWN'
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

    def __init__(self, simpleName: str, fileFilter: str, dataPath: str) -> None:
        self._simpleName = simpleName
        self._fileFilter = fileFilter
        self._dataPath = dataPath
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    @property
    def simpleName(self) -> str:
        return self._simpleName

    @property
    def fileFilter(self) -> str:
        return self._fileFilter

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(0, 0, numpy.dtype(numpy.ubyte), filePath=filePath)
        contentsTree = self._treeBuilder.createRootNode()
        arrayList: list[DiffractionPatternArray] = list()

        if filePath:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath][()]
                except KeyError:
                    logger.debug('Unable to find data.')
                except OSError:
                    logger.debug('Unable to read found data.')
                else:
                    array = H5DiffractionPatternArray(data, self._dataPath)
                    arrayList.append(array)

                    metadata = DiffractionMetadata(
                        filePath=filePath,
                        numberOfPatternsPerArray=data.shape[0],
                        numberOfPatternsTotal=data.shape[0],
                        patternDataType=data.dtype,
                    )

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(
        H5DiffractionFileReader(
            simpleName='HDF5',
            fileFilter='Hierarchical Data Format 5 Files (*.h5 *.hdf5)',
            dataPath='/entry/data/data',
        ))
    registry.registerPlugin(
        H5DiffractionFileReader(
            simpleName='LYNX',
            fileFilter='LYNX Diffraction Data Files (*.h5 *.hdf5)',
            dataPath='/entry/data/eiger_4',
        ))
