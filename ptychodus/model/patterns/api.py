from collections.abc import Sequence
from pathlib import Path
import logging

from ...api.patterns import (DiffractionFileReader, DiffractionMetadata, DiffractionPatternArray,
                             SimpleDiffractionDataset)
from ...api.plugins import PluginChooser
from ...api.tree import SimpleTreeNode
from .builder import ActiveDiffractionDatasetBuilder

logger = logging.getLogger(__name__)


class DiffractionDataAPI:

    def __init__(self, builder: ActiveDiffractionDatasetBuilder,
                 fileReaderChooser: PluginChooser[DiffractionFileReader]) -> None:
        super().__init__()
        self._builder = builder
        self._fileReaderChooser = fileReaderChooser

    def initializeStreaming(self, metadata: DiffractionMetadata) -> None:
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()
        dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)
        self._builder.switchTo(dataset)

    def startAssemblingDiffractionPatterns(self) -> None:
        self._builder.start()

    def assemble(self, array: DiffractionPatternArray) -> None:
        self._builder.insertArray(array)

    def getAssemblyQueueSize(self) -> int:
        return self._builder.getAssemblyQueueSize()

    def stopAssemblingDiffractionPatterns(self, finishAssembling: bool) -> None:
        self._builder.stop(finishAssembling)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def loadDiffractionDataset(self,
                               filePath: Path,
                               fileType: str,
                               *,
                               assemble: bool = True) -> str | None:
        self._fileReaderChooser.setCurrentPluginByName(fileType)

        if filePath.is_file():
            fileReader = self._fileReaderChooser.currentPlugin.strategy
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

            try:
                dataset = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read \"{filePath}\"') from exc
            else:
                self._builder.switchTo(dataset)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')
            return None

        if assemble:
            self._builder.start()
            self._builder.stop(finishAssembling=True)

        return self._fileReaderChooser.currentPlugin.simpleName
