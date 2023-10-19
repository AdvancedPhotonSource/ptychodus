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
                                         fileType: str = '') -> Optional[str]:
        item = self._factory.openItemFromFile(filePath, fileType)

        if item is None:
            logger.error(f'Unable to open object from \"{filePath}\"!')

        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromArray(self,
                                          name: str,
                                          array: ObjectArrayType,
                                          *,
                                          filePath: Optional[Path] = None,
                                          fileType: str = '',
                                          replaceItem: bool = False,
                                          selectItem: bool = False) -> Optional[str]:
        item = self._factory.createItemFromArray(name, array, filePath=filePath, fileType=fileType)
        itemName = self._repository.insertItem(item, name=name if replaceItem else None)

        if itemName is None:
            logger.error(f'Failed to insert object array \"{name}\"!')
        elif selectItem:
            self._object.selectItem(itemName)

        return itemName

    def insertItemIntoRepositoryFromInitializerName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromInitializerName(name)
        return self._repository.insertItem(item)

    def selectNewItemFromInitializerSimpleName(self, name: str) -> None:  # TODO improve name
        itemName = self.insertItemIntoRepositoryFromInitializerName(name)

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
            objectCentroid=self._sizer.getMidpointInMeters(),
        )

    def getSelectedThinObjectInterpolator(self) -> ObjectInterpolator:  # TODO remove when able
        return self._interpolatorFactory.createInterpolator(
            objectArray=self.getSelectedObjectArray()[0, ...],
            objectCentroid=self._sizer.getMidpointInMeters(),
        )
