from collections.abc import Callable, Mapping, Sequence
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
        self._initializersBySimpleName: Mapping[str, InitializerFactory] = {
            FromFileProbeInitializer.SIMPLE_NAME: self.createItemFromFile,
            DiskProbeInitializer.SIMPLE_NAME: self.createDiskItem,
            FresnelZonePlateProbeInitializer.SIMPLE_NAME: self.createFZPItem,
            SuperGaussianProbeInitializer.SIMPLE_NAME: self.createSuperGaussianItem,
        }
        self._initializersByDisplayName: Mapping[str, InitializerFactory] = {
            FromFileProbeInitializer.DISPLAY_NAME: self.createItemFromFile,
            DiskProbeInitializer.DISPLAY_NAME: self.createDiskItem,
            FresnelZonePlateProbeInitializer.DISPLAY_NAME: self.createFZPItem,
            SuperGaussianProbeInitializer.DISPLAY_NAME: self.createSuperGaussianItem,
        }

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def createFileInitializer(self,
                              filePath: Path,
                              *,
                              simpleFileType: str = '',
                              displayFileType: str = '') -> Optional[FromFileProbeInitializer]:
        if simpleFileType:
            self._fileReaderChooser.setFromSimpleName(simpleFileType)
        elif displayFileType:
            self._fileReaderChooser.setFromDisplayName(displayFileType)
        else:
            logger.error('Refusing to create file initializer without file type!')
            return None

        fileReader = self._fileReaderChooser.getCurrentStrategy()
        return FromFileProbeInitializer(filePath, fileReader)

    def openItemFromFile(self,
                         filePath: Path,
                         *,
                         simpleFileType: str = '',
                         displayFileType: str = '') -> Optional[ProbeRepositoryItem]:
        item: Optional[ProbeRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath,
                                                     simpleFileType=simpleFileType,
                                                     displayFileType=displayFileType)

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
                            simpleFileType: str = '',
                            displayFileType: str = '') -> ProbeRepositoryItem:
        item = ProbeRepositoryItem(self._rng, nameHint, array)

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

    def createItemFromFile(self) -> Optional[ProbeRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, simpleFileType=fileType)

    def createDiskItem(self) -> ProbeRepositoryItem:
        initializer = DiskProbeInitializer(self._sizer)
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
        return [initializerName for initializerName in self._initializersByDisplayName]

    def createItemFromSimpleName(self, name: str) -> Optional[ProbeRepositoryItem]:
        item: Optional[ProbeRepositoryItem] = None

        try:
            itemFactory = self._initializersBySimpleName[name]
        except KeyError:
            logger.error(f'Unknown probe initializer \"{name}\"!')
        else:
            item = itemFactory()

        return item

    def createItemFromDisplayName(self, name: str) -> Optional[ProbeRepositoryItem]:
        item: Optional[ProbeRepositoryItem] = None

        try:
            itemFactory = self._initializersByDisplayName[name]
        except KeyError:
            logger.error(f'Unknown probe initializer \"{name}\"!')
        else:
            item = itemFactory()

        return item
