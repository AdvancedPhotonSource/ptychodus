from __future__ import annotations
from collections.abc import Iterator
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


class ObjectRepositoryItemPresenter(Observable, Observer):

    def __init__(self, name: str, item: ObjectRepositoryItem) -> None:
        super().__init__()
        self._name = name
        self._item = item

    @classmethod
    def createInstance(cls, name: str,
                       item: ObjectRepositoryItem) -> ObjectRepositoryItemPresenter:
        presenter = cls(name, item)
        item.addObserver(presenter)
        return presenter

    @property
    def name(self) -> str:
        return self._name

    @property
    def initializer(self) -> str:
        return self._item.initializer

    @property
    def canActivate(self) -> bool:
        return self._item.canActivate

    def getDataType(self) -> str:
        return self._item.getDataType()

    def getExtentInPixels(self) -> ImageExtent:
        return self._item.getExtentInPixels()

    def getSizeInBytes(self) -> int:
        return self._item.getSizeInBytes()

    def getArray(self) -> ObjectArrayType:
        return self._item.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self.notifyObservers()


class ObjectRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ObjectRepository, itemFactory: ObjectRepositoryItemFactory,
                 objectAPI: ObjectAPI, fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._itemFactory = itemFactory
        self._objectAPI = objectAPI
        self._fileWriterChooser = fileWriterChooser
        self._itemPresenterList: list[ObjectRepositoryItemPresenter] = list()

    @classmethod
    def createInstance(
            cls, repository: ObjectRepository, itemFactory: ObjectRepositoryItemFactory,
            objectAPI: ObjectAPI,
            fileWriterChooser: PluginChooser[ObjectFileWriter]) -> ObjectRepositoryPresenter:
        presenter = cls(repository, itemFactory, objectAPI, fileWriterChooser)
        presenter._updateItemPresenterList()
        repository.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[ObjectRepositoryItemPresenter]:
        return iter(self._itemPresenterList)

    def __getitem__(self, index: int) -> ObjectRepositoryItemPresenter:
        return self._itemPresenterList[index]

    def __len__(self) -> int:
        return len(self._itemPresenterList)

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

    def getObjectArray(self, name: str) -> ObjectArrayType:  # FIXME remove when able
        return self._repository[name].getArray()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveObject(self, name: str, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, item.getArray())

        # FIXME setFileInfo
        if isinstance(item, SimpleObjectRepositoryItem):
            if item.getFileInfo() is None:
                fileInfo = ObjectFileInfo(fileType, filePath)
                item.setFileInfo(fileInfo)

    def canRemoveObject(self, name: str) -> bool:
        return self._repository.canRemoveItem(name)

    def removeObject(self, name: str) -> None:
        self._repository.removeItem(name)

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._objectAPI.getPixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._objectAPI.getPixelSizeYInMeters()

    def _updateItemPresenterList(self) -> None:
        itemPresenterList: list[ObjectRepositoryItemPresenter] = list()
        itemPresenterNames: set[str] = set()

        for itemPresenter in self._itemPresenterList:
            if itemPresenter.name in self._repository:
                itemPresenterList.append(itemPresenter)
                itemPresenterNames.add(itemPresenter.name)

        for name, item in self._repository.items():
            if name not in itemPresenterNames:
                itemPresenter = ObjectRepositoryItemPresenter.createInstance(name, item)
                itemPresenterList.append(itemPresenter)

        itemPresenterList.sort(key=lambda item: item.name)
        self._itemPresenterList = itemPresenterList
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateItemPresenterList()


class ObjectPresenter(Observable, Observer):

    def __init__(self, sizer: ObjectSizer, object_: ActiveObject, objectAPI: ObjectAPI) -> None:
        super().__init__()
        self._sizer = sizer
        self._object = object_
        self._objectAPI = objectAPI

    @classmethod
    def createInstance(cls, sizer: ObjectSizer, object_: ActiveObject,
                       objectAPI: ObjectAPI) -> ObjectPresenter:
        presenter = cls(sizer, object_, objectAPI)
        sizer.addObserver(presenter)
        object_.addObserver(presenter)
        return presenter

    def isActiveObjectValid(self) -> bool:
        actualExtent = self._object.getExtentInPixels()
        expectedExtent = self._sizer.getObjectExtent()
        widthIsBigEnough = (actualExtent.width >= expectedExtent.width)
        heightIsBigEnough = (actualExtent.height >= expectedExtent.height)
        hasComplexDataType = numpy.iscomplexobj(self._object.getArray())
        return (widthIsBigEnough and heightIsBigEnough and hasComplexDataType)

    def setActiveObject(self, name: str) -> None:
        self._objectAPI.setActiveObject(name)

    def getActiveObject(self) -> str:
        return self._object.name

    def getObjectArray(self) -> ObjectArrayType:  # FIXME remove
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
        self.repositoryPresenter = ObjectRepositoryPresenter.createInstance(
            self._repository, self._factory, self.objectAPI, fileWriterChooser)
        self.presenter = ObjectPresenter.createInstance(self.sizer, self._object, self.objectAPI)

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
