from pathlib import Path
from typing import Optional
import logging

from ...api.object import ObjectArrayType, ObjectInterpolator
from .factory import ObjectRepositoryItemFactory
from .interpolator import ObjectInterpolatorFactory
from .repository import ObjectRepository
from .selected import SelectedObject
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class ObjectAPI:

    def __init__(self, factory: ObjectRepositoryItemFactory, repository: ObjectRepository,
                 object_: SelectedObject, sizer: ObjectSizer,
                 interpolatorFactory: ObjectInterpolatorFactory) -> None:
        self._factory = factory
        self._repository = repository
        self._object = object_
        self._sizer = sizer
        self._interpolatorFactory = interpolatorFactory

    def insertItemIntoRepositoryFromFile(self,
                                         filePath: Path,
                                         *,
                                         simpleFileType: str = '',
                                         displayFileType: str = '') -> Optional[str]:
        item = self._factory.openItemFromFile(filePath,
                                              simpleFileType=simpleFileType,
                                              displayFileType=displayFileType)

        if item is None:
            logger.error(f'Unable to open object from \"{filePath}\"!')

        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromArray(self,
                                          name: str,
                                          array: ObjectArrayType,
                                          *,
                                          filePath: Optional[Path] = None,
                                          simpleFileType: str = '',
                                          displayFileType: str = '',
                                          replaceItem: bool = False,
                                          selectItem: bool = False) -> Optional[str]:
        item = self._factory.createItemFromArray(name,
                                                 array,
                                                 filePath=filePath,
                                                 simpleFileType=simpleFileType,
                                                 displayFileType=displayFileType)
        itemName = self._repository.insertItem(item, name=name if replaceItem else None)

        if itemName is None:
            logger.error(f'Failed to insert object array \"{name}\"!')
        elif selectItem:
            self._object.selectItem(itemName)

        return itemName

    def insertItemIntoRepositoryFromInitializerSimpleName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromSimpleName(name)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializerDisplayName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromDisplayName(name)
        return self._repository.insertItem(item)

    def selectNewItemFromInitializerSimpleName(self, name: str) -> None:  # TODO improve name
        itemName = self.insertItemIntoRepositoryFromInitializerSimpleName(name)

        if itemName is None:
            logger.error('Refusing to select null item!')
        else:
            self._object.selectItem(itemName)

    def getSelectedObjectArray(self) -> ObjectArrayType:
        selectedItem = self._object.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No object is selected!')

        return selectedItem.getArray()

    def getSelectedObjectInterpolator(self) -> ObjectInterpolator:
        return self._interpolatorFactory.createInterpolator(
            objectArray=self.getSelectedObjectArray(),
            objectCentroid=self._sizer.getCentroidInMeters(),
        )
