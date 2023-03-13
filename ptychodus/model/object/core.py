from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import Scan
from ...api.settings import SettingsRegistry
from ..probe import Apparatus, ProbeSizer
from ..statefulCore import StateDataType, StatefulCore
from .active import ActiveObject
from .api import ObjectAPI
from .itemFactory import ObjectRepositoryItemFactory
from .itemRepository import ObjectRepository, ObjectRepositoryItem
from .settings import ObjectSettings
from .simple import ObjectFileInfo, SimpleObjectRepositoryItem
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectRepositoryItemPresenter:
    name: str
    initializer: str
    dataType: str
    extentInPixels: ImageExtent
    sizeInBytes: int


class ObjectRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ObjectRepository) -> None:
        super().__init__()
        self._repository = repository
        self._nameList: list[str] = list()

    @classmethod
    def createInstance(cls, repository: ObjectRepository) -> ObjectRepositoryPresenter:
        presenter = cls(repository)
        presenter._updateNameList()
        repository.addObserver(presenter)
        return presenter

    def __getitem__(self, index: int) -> ObjectRepositoryItemPresenter:
        name = self._nameList[index]
        item = self._repository[name]
        return ObjectRepositoryItemPresenter(
            name=name,
            initializer=item.initializer,
            dataType=item.getDataType(),
            extentInPixels=item.getExtent(),
            sizeInBytes=item.getSizeInBytes(),
        )

    def __len__(self) -> int:
        return len(self._nameList)

    def canRemoveObject(self, name: str) -> bool:
        return self._repository.canRemoveItem(name)

    def removeObject(self, name: str) -> None:
        self._repository.removeItem(name)

    def _updateNameList(self) -> None:
        self._nameList = list(self._repository.keys())
        self._nameList.sort()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateNameList()


class ObjectPresenter(Observable, Observer):

    def __init__(self, sizer: ObjectSizer, itemFactory: ObjectRepositoryItemFactory,
                 repository: ObjectRepository, object_: ActiveObject, objectAPI: ObjectAPI,
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        super().__init__()
        self._sizer = sizer
        self._itemFactory = itemFactory
        self._repository = repository
        self._object = object_
        self._objectAPI = objectAPI
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, sizer: ObjectSizer, itemFactory: ObjectRepositoryItemFactory,
                       repository: ObjectRepository, object_: ActiveObject, objectAPI: ObjectAPI,
                       fileWriterChooser: PluginChooser[ObjectFileWriter]) -> ObjectPresenter:
        presenter = cls(sizer, itemFactory, repository, object_, objectAPI, fileWriterChooser)
        sizer.addObserver(presenter)
        object_.addObserver(presenter)
        return presenter

    def isActiveObjectValid(self) -> bool:
        actualExtent = self._object.getExtent()
        expectedExtent = self._sizer.getObjectExtent()
        widthIsBigEnough = (actualExtent.width >= expectedExtent.width)
        heightIsBigEnough = (actualExtent.height >= expectedExtent.height)
        hasComplexDataType = numpy.iscomplexobj(self._object.getArray())
        return (widthIsBigEnough and heightIsBigEnough and hasComplexDataType)

    def getActiveObject(self) -> str:
        return self._object.name

    def canActivateObject(self, name: str) -> bool:
        return self._object.canActivateObject(name)

    def setActiveObject(self, name: str) -> None:
        self._objectAPI.setActiveObject(name)

    def getItemNameList(self) -> list[str]:
        return self._itemFactory.getItemNameList()

    def initializeObject(self, name: str) -> Optional[str]:
        return self._objectAPI.insertObjectIntoRepositoryFromInitializer(name)

    def getOpenFileFilterList(self) -> list[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openObject(self, filePath: Path, fileFilter: str) -> None:
        self._objectAPI.insertObjectIntoRepositoryFromFile(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveObject(self, name: str, filePath: Path, fileFilter: str) -> None:
        try:
            initializer = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._object.getArray())

        if isinstance(initializer, SimpleObjectRepositoryItem):
            if initializer.getFileInfo() is None:
                fileInfo = ObjectFileInfo(fileType, filePath)
                initializer.setFileInfo(fileInfo)

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._objectAPI.getPixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._objectAPI.getPixelSizeYInMeters()

    def getObject(self) -> ObjectArrayType:
        return self._object.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()


class ObjectCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 apparatus: Apparatus, scan: Scan, probeSizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self._settings = ObjectSettings.createInstance(settingsRegistry)
        self.sizer = ObjectSizer.createInstance(apparatus, scan, probeSizer)
        self._factory = ObjectRepositoryItemFactory(rng, self._settings, self.sizer,
                                                    fileReaderChooser)
        self._repository = ObjectRepository()
        self._object = ActiveObject.createInstance(self._settings, self._factory, self._repository,
                                                   settingsRegistry)
        self.objectAPI = ObjectAPI(apparatus, self._factory, self._repository, self._object)
        self.repositoryPresenter = ObjectRepositoryPresenter.createInstance(self._repository)
        self.presenter = ObjectPresenter.createInstance(self.sizer, self._factory,
                                                        self._repository, self._object,
                                                        self.objectAPI, fileWriterChooser)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        state: StateDataType = {
            'object': self.objectAPI.getActiveObjectArray(),
        }
        return state

    def setStateData(self, state: StateDataType) -> None:
        try:
            array = state['object']
        except KeyError:
            logger.debug('Failed to restore object array state!')
            return

        name = 'Restart'
        itemName = self.objectAPI.insertObjectIntoRepository(name, array,
                                                             ObjectFileInfo.createNull())

        if itemName is None:
            logger.error('Failed to initialize \"{name}\"!')
        else:
            self.objectAPI.setActiveObject(itemName)
