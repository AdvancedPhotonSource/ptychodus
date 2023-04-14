from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.object import ObjectArrayType, ObjectFileReader
from ...api.plugins import PluginChooser
from .file import FromFileObjectInitializer
from .random import RandomObjectInitializer
from .repository import ObjectInitializer, ObjectRepositoryItem
from .settings import ObjectSettings
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings, sizer: ObjectSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._initializers: Mapping[str, Callable[[], Optional[ObjectRepositoryItem]]] = {
            FromFileObjectInitializer.NAME: self.createItemFromFile,
            RandomObjectInitializer.NAME: self.createRandomItem,
        }

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def createFileInitializer(self,
                              filePath: Path,
                              *,
                              simpleFileType: str = '',
                              displayFileType: str = '') -> Optional[FromFileObjectInitializer]:
        if simpleFileType:
            self._fileReaderChooser.setFromSimpleName(simpleFileType)
        elif displayFileType:
            self._fileReaderChooser.setFromDisplayName(displayFileType)
        else:
            logger.error('Refusing to create file initializer without file type!')
            return None

        fileReader = self._fileReaderChooser.getCurrentStrategy()
        return FromFileObjectInitializer(filePath, fileReader)

    def openItemFromFile(self,
                         filePath: Path,
                         *,
                         simpleFileType: str = '',
                         displayFileType: str = '') -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath,
                                                     simpleFileType=simpleFileType,
                                                     displayFileType=displayFileType)

            if initializer is None:
                logger.error('Refusing to create item without initializer!')
            else:
                item = ObjectRepositoryItem(filePath.stem)
                item.setInitializer(initializer)
        else:
            logger.debug(f'Refusing to create item with invalid file path \"{filePath}\"')

        return item

    def createItemFromArray(self,
                            nameHint: str,
                            array: ObjectArrayType,
                            *,
                            filePath: Optional[Path] = None,
                            simpleFileType: str = '',
                            displayFileType: str = '') -> ObjectRepositoryItem:
        item = ObjectRepositoryItem.createFromArray(nameHint, array)

        if filePath is not None:
            if filePath.is_file():
                initializer = self.createFileInitializer(filePath,
                                                         simpleFileType=simpleFileType,
                                                         displayFileType=displayFileType)

                if initializer is None:
                    logger.error('Refusing to add null initializer!')
                else:
                    item.setInitializer(initializer)
            else:
                logger.debug(f'Refusing to add initializer with invalid file path \"{filePath}\"')

        return item

    def createRandomItem(self) -> ObjectRepositoryItem:
        initializer = RandomObjectInitializer(self._rng, self._sizer)
        item = ObjectRepositoryItem(initializer.displayName)
        item.setInitializer(initializer)
        return item

    def createItemFromFile(self) -> Optional[ObjectRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, simpleFileType=fileType)

    def getInitializerNameList(self) -> list[str]:
        return [initializerName for initializerName in self._initializers]

    def createItem(self, initializerName: str) -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        try:
            itemFactory = self._initializers[initializerName]
        except KeyError:
            logger.error(f'Unknown object initializer \"{initializerName}\"!')
        else:
            item = itemFactory()

        return item
