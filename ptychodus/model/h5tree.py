from pathlib import Path
import logging

import h5py
import numpy

from .observer import Observable
from .tree import SimpleTreeNode
from .data_file import DataFileReader


logger = logging.getLogger(__name__)


class H5FileTreeReader(DataFileReader,Observable):
    FILE_FILTER = 'Hierarchical Data Format 5 Files (*.h5)'

    @staticmethod
    def createRootNode() -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

    def __init__(self) -> None:
        super().__init__()
        self._rootNode = H5FileTreeReader.createRootNode()

    def getTree(self) -> SimpleTreeNode:
        return self._rootNode

    @staticmethod
    def _addAttributes(treeNode: SimpleTreeNode, attributeManager: h5py.AttributeManager) -> None:
        for name, value in attributeManager.items():
            valueStr = None

            if isinstance(value, numpy.bytes_):
                valueStr = 'STRING = "' + value.decode('utf-8') + '"'
            elif numpy.isscalar(value):
                valueStr = f'SCALAR {value.dtype} = {value}'
            elif isinstance(value, numpy.ndarray):
                valueStr = f'{value.shape} {value.dtype}'
            else:
                logger.debug(f'UNKNOWN: {value} {type(value)}')
                valueStr = 'UNKNOWN'

            treeNode.createChild([str(name), 'Attribute', valueStr])

    def read(self, rootGroup: h5py.Group) -> None:
        self._rootNode = H5FileTreeReader.createRootNode()
        unvisited = [ (self._rootNode, rootGroup) ]

        while unvisited:
            parentItem, h5Group = unvisited.pop()

            for itemName in h5Group:
                itemType = 'Unknown'
                itemDetails = ''
                h5Item = h5Group.get(itemName, getlink=True)

                treeNode = parentItem.createChild(None)

                if isinstance(h5Item, h5py.HardLink):
                    itemType = 'Hard Link'
                    h5Item = h5Group.get(itemName, getlink=False)

                    if isinstance(h5Item, h5py.Group):
                        itemType = 'Group'
                        H5FileTreeReader._addAttributes(treeNode, h5Item.attrs)
                        unvisited.append( (treeNode, h5Item) )
                    elif isinstance(h5Item, h5py.Dataset):
                        itemType = 'Dataset'
                        H5FileTreeReader._addAttributes(treeNode, h5Item.attrs)
                        spaceId = h5Item.id.get_space()

                        if spaceId.get_simple_extent_type() == h5py.h5s.SCALAR:
                            value = h5Item[()]

                            if isinstance(value, numpy.bytes_):
                                valueStr = 'STRING = "' + value.decode('utf-8') + '"'
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

        self.notifyObservers()

