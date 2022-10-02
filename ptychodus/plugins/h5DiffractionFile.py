from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternArray, SimpleDiffractionDataset)
from ptychodus.api.observer import Observable
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class H5DiffractionFileTreeBuilder:

    def _addAttributes(self, treeNode: SimpleTreeNode,
                       attributeManager: h5py.AttributeManager) -> None:
        for name, value in attributeManager.items():
            stringInfo = h5py.check_string_dtype(value.dtype)
            valueStr = None

            if stringInfo:
                valueStr = f'STRING = "{value.decode(stringInfo.encoding)}"'
            elif numpy.isscalar(value):
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
                            elif numpy.isscalar(value):
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

    def __init__(self) -> None:
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    @property
    def simpleName(self) -> str:
        return 'HDF5'

    @property
    def fileFilter(self) -> str:
        return 'Hierarchical Data Format 5 Files (*.h5 *.hdf5)'

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(filePath, 0, 0, numpy.dtype(numpy.ubyte))
        contentsTree = self._treeBuilder.createRootNode()
        arrayList: list[DiffractionPatternArray] = list()

        if filePath:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(H5DiffractionFileReader())
