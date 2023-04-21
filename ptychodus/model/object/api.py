from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

from ...api.object import ObjectArrayType
from ..probe import Apparatus
from .factory import ObjectRepositoryItemFactory
from .repository import ObjectRepository
from .selected import SelectedObject

logger = logging.getLogger(__name__)


class ObjectAPI:

    def __init__(self, apparatus: Apparatus, factory: ObjectRepositoryItemFactory,
                 repository: ObjectRepository, object_: SelectedObject) -> None:
        self._apparatus = apparatus
        self._factory = factory
        self._repository = repository
        self._object = object_

    def insertItemIntoRepositoryFromFile(self,
                                         filePath: Path,
                                         *,
                                         simpleFileType: str = '',
                                         displayFileType: str = '') -> Optional[str]:
        itemName: Optional[str] = None
        item = self._factory.openItemFromFile(filePath,
                                              simpleFileType=simpleFileType,
                                              displayFileType=displayFileType)

        if item is None:
            logger.error(f'Unable to open object from \"{filePath}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def insertItemIntoRepositoryFromArray(self,
                                          nameHint: str,
                                          array: ObjectArrayType,
                                          *,
                                          filePath: Optional[Path] = None,
                                          simpleFileType: str = '',
                                          displayFileType: str = '') -> Optional[str]:
        item = self._factory.createItemFromArray(nameHint,
                                                 array,
                                                 filePath=filePath,
                                                 simpleFileType=simpleFileType,
                                                 displayFileType=displayFileType)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializer(self, initializerName: str) -> Optional[str]:
        itemName: Optional[str] = None
        item = self._factory.createItem(initializerName)

        if item is None:
            logger.error(f'Unknown object initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def selectItem(self, itemName: str) -> None:
        self._object.selectItem(itemName)

    def initializeAndActivateItem(self, initializerName: str) -> None:
        itemName = self.insertItemIntoRepositoryFromInitializer(initializerName)

        if itemName is None:
            logger.error('Failed to initialize \"{initializerName}\"!')
        else:
            self.selectItem(itemName)

    def getSelectedObjectArray(self) -> Optional[ObjectArrayType]:
        selectedItem = self._object.getSelectedItem()
        return None if selectedItem is None else selectedItem.getArray()

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeYInMeters()
