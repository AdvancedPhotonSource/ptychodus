from pathlib import Path
import logging

import h5py
import numpy

from .observer import Observable
from .tree import SimpleTreeNode
from .data import DataFileReader

logger = logging.getLogger(__name__)


class H5FileReader(DataFileReader, Observable):
    @staticmethod
    def createRootNode() -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

    def __init__(self, simpleName: str = 'HDF5', fileFilter: str = 'Hierarchical Data Format 5 Files (*.h5)') -> None:
        super().__init__()
        self._rootNode = H5FileReader.createRootNode()
        self._simpleName = simpleName
        self._fileFilter = fileFilter

    def getTree(self) -> SimpleTreeNode:
        return self._rootNode

    @property
    def simpleName(self) -> str:
        return self._simpleName

    @property
    def fileFilter(self) -> str:
        return self._fileFilter

    @staticmethod
    def _addAttributes(treeNode: SimpleTreeNode, attributeManager: h5py.AttributeManager) -> None:
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

    def read(self, filePath: Path) -> None:
        if not filePath:
            return

        with h5py.File(filePath, 'r') as h5File:
            self._rootNode = H5FileReader.createRootNode()
            unvisited = [(self._rootNode, h5File)]

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
                            H5FileReader._addAttributes(treeNode, h5Item.attrs)
                            unvisited.append((treeNode, h5Item))
                        elif isinstance(h5Item, h5py.Dataset):
                            itemType = 'Dataset'
                            H5FileReader._addAttributes(treeNode, h5Item.attrs)
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

            self.readRootGroup(h5File)

        self.notifyObservers()

    def readRootGroup(self, rootGroup: h5py.Group) -> None:
        pass
