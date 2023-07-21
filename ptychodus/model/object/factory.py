from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, TypeAlias
import logging

import numpy

from ...api.object import ObjectArrayType, ObjectFileReader
from ...api.plugins import PluginChooser
from .file import FromFileObjectInitializer
from .random import RandomObjectInitializer
from .repository import ObjectRepositoryItem
from .settings import ObjectSettings
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)

InitializerFactory: TypeAlias = Callable[[], Optional[ObjectRepositoryItem]]


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings, sizer: ObjectSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._initializers = PluginChooser[InitializerFactory]()
        self._initializers.registerPlugin(
            self.createItemFromFile,
            simpleName=FromFileObjectInitializer.SIMPLE_NAME,
            displayName=FromFileObjectInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createRandomItem,
            simpleName=RandomObjectInitializer.SIMPLE_NAME,
            displayName=RandomObjectInitializer.DISPLAY_NAME,
        )

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createFileInitializer(self, filePath: Path,
                              fileType: str) -> Optional[FromFileObjectInitializer]:
        self._fileReaderChooser.setCurrentPluginByName(fileType)
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileObjectInitializer(filePath,
                                         self._fileReaderChooser.currentPlugin.simpleName,
                                         fileReader)

    def openItemFromFile(self, filePath: Path, fileType: str) -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath, fileType)

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
                            fileType: str = '') -> ObjectRepositoryItem:
        item = ObjectRepositoryItem(nameHint, array)

        if filePath is not None:
            if filePath.is_file():
                initializer = self.createFileInitializer(filePath, fileType)

                if initializer is None:
                    logger.error('Refusing to add null initializer!')
                else:
                    item.setInitializer(initializer)
            else:
                logger.debug(f'Refusing to add initializer with invalid file path \"{filePath}\"')

        return item

    def createItemFromFile(self) -> Optional[ObjectRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, fileType)

    def createRandomItem(self) -> ObjectRepositoryItem:
        initializer = RandomObjectInitializer(self._rng, self._sizer)
        item = ObjectRepositoryItem(initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._initializers.getDisplayNameList()

    def createItemFromInitializerName(self, name: str) -> Optional[ObjectRepositoryItem]:
        item: Optional[ObjectRepositoryItem] = None

        try:
            plugin = self._initializers[name]
        except KeyError:
            logger.error(f'Unknown object initializer \"{name}\"!')
        else:
            item = plugin.strategy()

        return item
