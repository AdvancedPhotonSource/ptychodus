from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.object import ObjectArrayType, ObjectFileReader
from ...api.plugins import PluginChooser
from .itemRepository import ObjectRepositoryItem
from .random import RandomObjectRepositoryItem
from .settings import ObjectSettings
from .simple import ObjectFileInfo, SimpleObjectRepositoryItem
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings, sizer: ObjectSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader]) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._variants: Mapping[str, Callable[[], ObjectRepositoryItem]] = {
            RandomObjectRepositoryItem.NAME.casefold(): self.createRandomItem,
        }

    def createRandomItem(self) -> RandomObjectRepositoryItem:
        return RandomObjectRepositoryItem(self._rng, self._sizer)

    def createItemFromArray(self, name: str, array: ObjectArrayType,
                            fileInfo: Optional[ObjectFileInfo]) -> SimpleObjectRepositoryItem:
        return SimpleObjectRepositoryItem(name, array, fileInfo)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _readObject(self, filePath: Path) -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        if filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileInfo = ObjectFileInfo(fileType, filePath)
            reader = self._fileReaderChooser.getCurrentStrategy()

            try:
                array = reader.read(filePath)
            except Exception:
                logger.exception(f'Failed to read \"{filePath}\"')
            else:
                item = self.createItemFromArray(filePath.stem, array, fileInfo)
        else:
            logger.debug(f'Refusing to read invalid file path \"{filePath}\"')

        return item

    def openObject(self, filePath: Path, fileFilter: str) -> Optional[ObjectRepositoryItem]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readObject(filePath)

    def getItemNameList(self) -> list[str]:
        return [name.title() for name in self._variants]

    def createItem(self, initializerName: str) -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        if initializerName.casefold() == 'fromfile':
            fileInfo = ObjectFileInfo.createFromSettings(self._settings)
            self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
            item = self._readObject(fileInfo.filePath)
        else:
            try:
                itemFactory = self._variants[initializerName.casefold()]
            except KeyError:
                logger.error(f'Unknown object initializer \"{initializerName}\"!')
            else:
                item = itemFactory()

        return item
