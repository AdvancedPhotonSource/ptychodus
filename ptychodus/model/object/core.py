from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.object import (ObjectArrayType, ObjectFileReader, ObjectFileWriter,
                           ObjectPhaseCenteringStrategy)
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from ...api.state import ObjectStateData, StatefulCore
from ..probe import Apparatus, ProbeSizer
from ..scan import ScanSizer
from .api import ObjectAPI
from .factory import ObjectRepositoryItemFactory
from .interpolator import ObjectInterpolatorFactory
from .repository import ObjectRepository, ObjectRepositoryItem
from .selected import ObjectRepositoryItemSettingsDelegate, SelectedObject
from .settings import ObjectSettings
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

    @classmethod
    def createInstance(
            cls, repository: ObjectRepository, itemFactory: ObjectRepositoryItemFactory,
            objectAPI: ObjectAPI,
            fileWriterChooser: PluginChooser[ObjectFileWriter]) -> ObjectRepositoryPresenter:
        presenter = cls(repository, itemFactory, objectAPI, fileWriterChooser)
        repository.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[ObjectRepositoryItemPresenter]:
        for name, item in self._repository.items():
            yield ObjectRepositoryItemPresenter(name, item)

    def __getitem__(self, index: int) -> ObjectRepositoryItemPresenter:
        nameItemTuple = self._repository.getNameItemTupleByIndex(index)
        return ObjectRepositoryItemPresenter(*nameItemTuple)

    def __len__(self) -> int:
        return len(self._repository)

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._itemFactory.getInitializerDisplayNameList()

    def initializeObject(self, displayName: str) -> Optional[str]:
        return self._objectAPI.insertItemIntoRepositoryFromInitializerDisplayName(displayName)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openObject(self, filePath: Path, fileFilter: str) -> None:
        self._objectAPI.insertItemIntoRepositoryFromFile(filePath, displayFileType=fileFilter)

    def getSaveFileFilterList(self) -> Sequence[str]:
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

        if item.getInitializer() is None:
            initializer = self._itemFactory.createFileInitializer(filePath,
                                                                  simpleFileType=fileType)

            if initializer is not None:
                item.setInitializer(initializer)

    def removeObject(self, name: str) -> None:
        self._repository.removeItem(name)

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()


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
        selectedObject = self._object.getSelectedItem()

        if selectedObject is None:
            return False

        actualExtent = selectedObject.getExtentInPixels()
        expectedExtent = self._sizer.getObjectExtent()
        widthIsBigEnough = (actualExtent.width >= expectedExtent.width)
        heightIsBigEnough = (actualExtent.height >= expectedExtent.height)
        hasComplexDataType = numpy.iscomplexobj(selectedObject.getArray())
        return (widthIsBigEnough and heightIsBigEnough and hasComplexDataType)

    def selectObject(self, name: str) -> None:
        self._object.selectItem(name)

    def getSelectedObject(self) -> str:
        return self._object.getSelectedName()

    def getSelectedObjectArray(self) -> Optional[ObjectArrayType]:
        array: Optional[ObjectArrayType] = None

        try:
            array = self._objectAPI.getSelectedObjectArray()
        except ValueError as err:
            logger.debug(err)

        return array

    def getSelectableNames(self) -> Sequence[str]:
        return self._object.getSelectableNames()

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()


class ObjectCore(StatefulCore[ObjectStateData]):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 apparatus: Apparatus, scanSizer: ScanSizer, probeSizer: ProbeSizer,
                 phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self._settings = ObjectSettings.createInstance(settingsRegistry)
        self.sizer = ObjectSizer.createInstance(self._settings, apparatus, scanSizer, probeSizer)
        self._itemFactory = ObjectRepositoryItemFactory(rng, self._settings, self.sizer,
                                                        fileReaderChooser)
        self._repository = ObjectRepository()
        self._itemSettingsDelegate = ObjectRepositoryItemSettingsDelegate(
            self._settings, self._itemFactory, self._repository)
        self._object = SelectedObject.createInstance(self._repository, self._itemSettingsDelegate,
                                                     settingsRegistry)
        self._interpolatorFactory = ObjectInterpolatorFactory.createInstance(
            self._settings, self.sizer, phaseCenteringStrategyChooser, settingsRegistry)
        self.objectAPI = ObjectAPI(self._itemFactory, self._repository, self._object, self.sizer,
                                   self._interpolatorFactory)
        self.repositoryPresenter = ObjectRepositoryPresenter.createInstance(
            self._repository, self._itemFactory, self.objectAPI, fileWriterChooser)
        self.presenter = ObjectPresenter.createInstance(self.sizer, self._object, self.objectAPI)

    def getStateData(self) -> ObjectStateData:
        return ObjectStateData(
            array=self.objectAPI.getSelectedObjectArray(),  # FIXME try/except
        )

    def setStateData(self, stateData: ObjectStateData, stateFilePath: Path) -> None:
        self.objectAPI.insertItemIntoRepositoryFromArray(name='Restart',
                                                         array=stateData.array,
                                                         filePath=stateFilePath,
                                                         simpleFileType=stateFilePath.suffix,
                                                         selectItem=True)
