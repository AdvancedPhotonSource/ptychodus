from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

from ...api.object import ObjectArrayType
from ..probe import Apparatus
from .active import ActiveObject
from .itemFactory import ObjectRepositoryItemFactory
from .itemRepository import ObjectRepository
from .simple import ObjectFileInfo

logger = logging.getLogger(__name__)


class ObjectAPI:

    def __init__(self, apparatus: Apparatus, factory: ObjectRepositoryItemFactory,
                 repository: ObjectRepository, object_: ActiveObject) -> None:
        self._apparatus = apparatus
        self._factory = factory
        self._repository = repository
        self._object = object_

    def insertObjectIntoRepositoryFromFile(self, filePath: Path, fileFilter: str) -> Optional[str]:
        itemName: Optional[str] = None
        object_ = self._factory.openObject(filePath, fileFilter)

        if object_ is None:
            logger.error(f'Unable to open object from \"{filePath}\"!')
        else:
            itemName = self._repository.insertItem(object_)

        return itemName

    def insertObjectIntoRepositoryFromInitializer(self, initializerName: str) -> Optional[str]:
        itemName: Optional[str] = None
        object_ = self._factory.createItem(initializerName)

        if object_ is None:
            logger.error(f'Unknown object initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(object_)

        return itemName

    def insertObjectIntoRepository(self, name: str, array: ObjectArrayType,
                                   fileInfo: Optional[ObjectFileInfo]) -> Optional[str]:
        item = self._factory.createItemFromArray(name, array, fileInfo)
        return self._repository.insertItem(item)

    def setActiveObject(self, itemName: str) -> None:
        self._object.setActiveObject(itemName)

    def initializeAndActivateObject(self, name: str) -> None:
        itemName = self.insertObjectIntoRepositoryFromInitializer(name)

        if itemName is None:
            logger.error('Failed to initialize \"{name}\"!')
        else:
            self.setActiveObject(itemName)

    def getActiveObjectArray(self) -> ObjectArrayType:
        return self._object.getArray()

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeYInMeters()
