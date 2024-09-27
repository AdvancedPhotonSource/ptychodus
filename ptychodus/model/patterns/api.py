from collections.abc import Sequence
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    CropCenter,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    DiffractionPatternArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.tree import SimpleTreeNode

from .active import ActiveDiffractionDataset
from .builder import ActiveDiffractionDatasetBuilder
from .settings import PatternSettings

logger = logging.getLogger(__name__)


class PatternsAPI:
    def __init__(
        self,
        settings: PatternSettings,
        builder: ActiveDiffractionDatasetBuilder,
        dataset: ActiveDiffractionDataset,
        fileReaderChooser: PluginChooser[DiffractionFileReader],
        fileWriterChooser: PluginChooser[DiffractionFileWriter],
    ) -> None:
        super().__init__()
        self._settings = settings
        self._builder = builder
        self._dataset = dataset
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    def initializeStreaming(self, metadata: DiffractionMetadata) -> None:
        contentsTree = SimpleTreeNode.createRoot(["Name", "Type", "Details"])
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

    def openPatterns(
        self,
        filePath: Path,
        *,
        fileType: str | None = None,
        cropCenter: CropCenter | None = None,
        cropExtent: ImageExtent | None = None,
        assemble: bool = True,
    ) -> str | None:
        if cropCenter is not None:
            self._settings.cropCenterXInPixels.setValue(cropCenter.positionXInPixels)
            self._settings.cropCenterYInPixels.setValue(cropCenter.positionYInPixels)

        if cropExtent is not None:
            self._settings.cropWidthInPixels.setValue(cropExtent.widthInPixels)
            self._settings.cropHeightInPixels.setValue(cropExtent.heightInPixels)

        fileType_ = self._settings.fileType.getValue() if fileType is None else fileType
        self._fileReaderChooser.setCurrentPluginByName(fileType_)

        if filePath.is_file():
            fileReader = self._fileReaderChooser.currentPlugin.strategy
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading "{filePath}" as "{fileType}"')

            try:
                dataset = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                self._builder.switchTo(dataset)
        else:
            logger.warning(f"Refusing to read invalid file path {filePath}")
            return None

        if assemble:
            self._builder.start()
            self._builder.stop(finishAssembling=True)

        return self._fileReaderChooser.currentPlugin.simpleName

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def savePatterns(self, filePath: Path, fileType: str) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileType)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, self._dataset)

    def importProcessedPatterns(self, filePath: Path) -> None:
        if filePath.is_file():
            logger.debug(f'Reading processed patterns from "{filePath}"')

            try:
                contents = numpy.load(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc

            self._builder.stop(finishAssembling=False)
            self._dataset.setAssembledData(contents["patterns"], contents["indexes"])
            self._builder.start()
            self._builder.stop(finishAssembling=True)
        else:
            logger.warning(f"Refusing to read invalid file path {filePath}")

    def exportProcessedPatterns(self, filePath: Path) -> None:
        contents: dict[str, Any] = {
            "indexes": numpy.array(self._dataset.getAssembledIndexes()),
            "patterns": numpy.array(self._dataset.getAssembledData()),
        }
        logger.debug(f'Writing processed patterns to "{filePath}"')
        numpy.savez(filePath, **contents)
