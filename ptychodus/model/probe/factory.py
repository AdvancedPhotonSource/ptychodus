from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, TypeAlias
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.probe import ProbeArrayType, ProbeFileReader
from .apparatus import Apparatus
from .disk import DiskProbeInitializer
from .file import FromFileProbeInitializer
from .fzp import FresnelZonePlateProbeInitializer
from .repository import ProbeRepositoryItem
from .settings import ProbeSettings
from .sizer import ProbeSizer
from .superGaussian import SuperGaussianProbeInitializer

logger = logging.getLogger(__name__)

InitializerFactory: TypeAlias = Callable[[], Optional[ProbeRepositoryItem]]


class ProbeRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings, apparatus: Apparatus,
                 sizer: ProbeSizer, fileReaderChooser: PluginChooser[ProbeFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._apparatus = apparatus
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._initializers = PluginChooser[InitializerFactory]()
        self._initializers.registerPlugin(
            self.createItemFromFile,
            simpleName=FromFileProbeInitializer.SIMPLE_NAME,
            displayName=FromFileProbeInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createDiskItem,
            simpleName=DiskProbeInitializer.SIMPLE_NAME,
            displayName=DiskProbeInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createFZPItem,
            simpleName=FresnelZonePlateProbeInitializer.SIMPLE_NAME,
            displayName=FresnelZonePlateProbeInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createSuperGaussianItem,
            simpleName=SuperGaussianProbeInitializer.SIMPLE_NAME,
            displayName=SuperGaussianProbeInitializer.DISPLAY_NAME,
        )

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createFileInitializer(self, filePath: Path,
                              fileType: str) -> Optional[FromFileProbeInitializer]:
        self._fileReaderChooser.setCurrentPluginByName(fileType)
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileProbeInitializer(filePath, self._fileReaderChooser.currentPlugin.simpleName,
                                        fileReader)

    def openItemFromFile(self, filePath: Path, fileType: str) -> Optional[ProbeRepositoryItem]:
        item: Optional[ProbeRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath, fileType)

            if initializer is None:
                logger.error('Refusing to create item without initializer!')
            else:
                item = ProbeRepositoryItem(self._rng, filePath.stem)
                item.setInitializer(initializer)
        else:
            logger.debug(f'Refusing to create item with invalid file path \"{filePath}\"')

        return item

    def createItemFromArray(self,
                            nameHint: str,
                            array: ProbeArrayType,
                            *,
                            filePath: Optional[Path] = None,
                            fileType: str = '') -> ProbeRepositoryItem:
        item = ProbeRepositoryItem(self._rng, nameHint, array)

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

    def createItemFromFile(self) -> Optional[ProbeRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, fileType)

    def createDiskItem(self) -> ProbeRepositoryItem:
        initializer = DiskProbeInitializer(self._sizer, self._apparatus)
        item = ProbeRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createFZPItem(self) -> ProbeRepositoryItem:
        initializer = FresnelZonePlateProbeInitializer(self._sizer, self._apparatus)
        item = ProbeRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createSuperGaussianItem(self) -> ProbeRepositoryItem:
        initializer = SuperGaussianProbeInitializer(self._sizer, self._apparatus)
        item = ProbeRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._initializers.getDisplayNameList()

    def createItemFromInitializerName(self, name: str) -> Optional[ProbeRepositoryItem]:
        item: Optional[ProbeRepositoryItem] = None

        try:
            plugin = self._initializers[name]
        except KeyError:
            logger.error(f'Unknown probe initializer \"{name}\"!')
        else:
            item = plugin.strategy()

        return item
