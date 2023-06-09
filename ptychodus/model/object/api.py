from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

from ...api.object import ObjectArrayType, ObjectPoint
from ...api.scan import ScanPoint
from .factory import ObjectRepositoryItemFactory
from .interpolator import ObjectInterpolator
from .repository import ObjectRepository
from .selected import SelectedObject
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class ObjectAPI:

    def __init__(self, factory: ObjectRepositoryItemFactory, repository: ObjectRepository,
                 object_: SelectedObject, sizer: ObjectSizer,
                 interpolator: ObjectInterpolator) -> None:
        self._factory = factory
        self._repository = repository
        self._object = object_
        self._sizer = sizer
        self._interpolator = interpolator

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

    def insertItemIntoRepositoryFromInitializerSimpleName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromSimpleName(name)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializerDisplayName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromDisplayName(name)
        return self._repository.insertItem(item)

    def selectItem(self, itemName: str) -> None:
        self._object.selectItem(itemName)

    def selectNewItemFromInitializerSimpleName(self, name: str) -> None:  # TODO improve name
        itemName = self.insertItemIntoRepositoryFromInitializerSimpleName(name)

        if itemName is None:
            logger.error('Refusing to select null item!')
        else:
            self.selectItem(itemName)

    def getSelectedObjectArray(self) -> ObjectArrayType:
        selectedItem = self._object.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No object is selected!')

        return selectedItem.getArray()

    def getPixelSizeXInMeters(self) -> Decimal:  # FIXME remove
        return self._sizer.getPixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:  # FIXME remove
        return self._sizer.getPixelSizeYInMeters()

    def mapScanPointToObjectPoint(self, point: ScanPoint) -> ObjectPoint:
        '''returns the object position in pixels'''
        return self._interpolator.mapScanPointToObjectPoint(point)

    def mapObjectPointToScanPoint(self, point: ObjectPoint) -> ScanPoint:
        '''returns the scan position in meters'''
        return self._interpolator.mapObjectPointToScanPoint(point)

    def getObjectPatch(self, point: ScanPoint) -> ObjectArrayType:
        return self._interpolator.getPatch(point)
