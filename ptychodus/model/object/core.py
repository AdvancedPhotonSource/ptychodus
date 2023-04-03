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
from ...api.settings import SettingsRegistry
from ..probe import Apparatus, ProbeSizer
from ..scan import ScanSizer
from ..statefulCore import StateDataType, StatefulCore
from .api import ObjectAPI
from .factory import ObjectRepositoryItemFactory
from .repository import ObjectRepository, ObjectRepositoryItem
from .selected import ObjectRepositoryItemSettingsDelegate, SelectedObject
from .settings import ObjectSettings
from .simple import ObjectFileInfo, SimpleObjectRepositoryItem
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectRepositoryItemPresenter:
    name: str
    item: ObjectRepositoryItem


class ObjectRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ObjectRepository, itemFactory: ObjectRepositoryItemFactory,
                 objectAPI: ObjectAPI, fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._itemFactory = itemFactory
        self._objectAPI = objectAPI
        self._fileWriterChooser = fileWriterChooser
        self._itemNameList: list[str] = list()

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
        for name, item in self._repository.items():
            yield ObjectRepositoryItemPresenter(name, item)

    def __getitem__(self, index: int) -> ObjectRepositoryItemPresenter:
        itemName = self._itemNameList[index]
        return ObjectRepositoryItemPresenter(itemName, self._repository[itemName])

    def __len__(self) -> int:
        return len(self._repository)

    def getInitializerNameList(self) -> list[str]:
        return self._itemFactory.getInitializerNameList()

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
            item = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, item.getArray())

        # TODO test this
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
        self._itemNameList = [name for name in self._repository]
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateItemPresenterList()


class ObjectPresenter(Observable, Observer):

    def __init__(self, sizer: ObjectSizer, object_: SelectedObject, objectAPI: ObjectAPI) -> None:
        super().__init__()
        self._sizer = sizer
        self._object = object_
        self._objectAPI = objectAPI

    @classmethod
    def createInstance(cls, sizer: ObjectSizer, object_: SelectedObject,
                       objectAPI: ObjectAPI) -> ObjectPresenter:
        presenter = cls(sizer, object_, objectAPI)
        sizer.addObserver(presenter)
        object_.addObserver(presenter)
        return presenter

    def isSelectedObjectValid(self) -> bool:
        actualExtent = self._object.getSelectedItem().getExtentInPixels()
        expectedExtent = self._sizer.getObjectExtent()
        widthIsBigEnough = (actualExtent.width >= expectedExtent.width)
        heightIsBigEnough = (actualExtent.height >= expectedExtent.height)
        hasComplexDataType = numpy.iscomplexobj(self._object.getSelectedItem().getArray())
        return (widthIsBigEnough and heightIsBigEnough and hasComplexDataType)

    def selectObject(self, name: str) -> None:
        self._objectAPI.selectObject(name)

    def getSelectedObject(self) -> str:
        return self._object.getSelectedName()

    def getSelectedObjectArray(self) -> ObjectArrayType:
        return self._object.getSelectedItem().getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()


class ObjectCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 apparatus: Apparatus, scanSizer: ScanSizer, probeSizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self._settings = ObjectSettings.createInstance(settingsRegistry)
        self.sizer = ObjectSizer.createInstance(self._settings, apparatus, scanSizer, probeSizer)
        self._factory = ObjectRepositoryItemFactory(rng, self._settings, self.sizer,
                                                    fileReaderChooser)
        self._repository = ObjectRepository()
        self._itemSettingsDelegate = ObjectRepositoryItemSettingsDelegate(
            self._settings, self._factory, self._repository)
        self._object = SelectedObject.createInstance(self._repository, self._itemSettingsDelegate,
                                                     settingsRegistry)
        self.objectAPI = ObjectAPI(apparatus, self._factory, self._repository, self._object)
        self.repositoryPresenter = ObjectRepositoryPresenter.createInstance(
            self._repository, self._factory, self.objectAPI, fileWriterChooser)
        self.presenter = ObjectPresenter.createInstance(self.sizer, self._object, self.objectAPI)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        state: StateDataType = {
            'object': self.objectAPI.getSelectedObjectArray(),
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
            self.objectAPI.selectObject(itemName)
