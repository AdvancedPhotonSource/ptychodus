from collections.abc import Sequence
from pathlib import Path
import logging


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

from .dataset import AssembledDiffractionDataset
from .settings import DetectorSettings, PatternSettings

logger = logging.getLogger(__name__)


class PatternsStreamingContext:
    def __init__(self, dataset: AssembledDiffractionDataset, metadata: DiffractionMetadata) -> None:
        self._dataset = dataset
        self._metadata = metadata
        self._trigger_counts: list[int] = []  # FIXME use this

    def start(self) -> None:
        contents_tree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        stream_dataset = SimpleDiffractionDataset(self._metadata, contents_tree, [])
        self._dataset.reload(stream_dataset, start_assembling=True, finish_assembling=False)

    def append_array(self, array: DiffractionPatternArray, trigger_counts: Sequence[int]) -> None:
        self._dataset.append_array(array)
        self._trigger_counts.extend(trigger_counts)

    def get_queue_size(self) -> int:
        return self._dataset.queue_size

    def stop(self) -> None:
        self._dataset.stop(finish_assembling=True)


class PatternsAPI:
    def __init__(
        self,
        patternSettings: PatternSettings,
        detectorSettings: DetectorSettings,
        dataset: AssembledDiffractionDataset,
        fileReaderChooser: PluginChooser[DiffractionFileReader],
        fileWriterChooser: PluginChooser[DiffractionFileWriter],
    ) -> None:
        super().__init__()
        self._patternSettings = patternSettings
        self._detectorSettings = detectorSettings
        self._dataset = dataset
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    def createStreamingContext(self, metadata: DiffractionMetadata) -> PatternsStreamingContext:
        return PatternsStreamingContext(self._dataset, metadata)

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
        detectorExtent: ImageExtent | None = None,
        assemble: bool = True,
    ) -> int:
        if cropCenter is not None:
            self._patternSettings.cropCenterXInPixels.setValue(cropCenter.positionXInPixels)
            self._patternSettings.cropCenterYInPixels.setValue(cropCenter.positionYInPixels)

        if cropExtent is not None:
            self._patternSettings.cropWidthInPixels.setValue(cropExtent.widthInPixels)
            self._patternSettings.cropHeightInPixels.setValue(cropExtent.heightInPixels)

        if detectorExtent is not None:
            self._detectorSettings.widthInPixels.setValue(detectorExtent.widthInPixels)
            self._detectorSettings.heightInPixels.setValue(detectorExtent.heightInPixels)

        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(
                self._patternSettings.fileType.getValue() if fileType is None else fileType
            )
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading "{filePath}" as "{fileType}"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                dataset = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                self._dataset.reload(dataset, start_assembling=True, finish_assembling=True)
                return 0
        else:
            logger.warning(f'Refusing to read invalid file path {filePath}')

        return -1

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

    def importAssembledPatterns(self, filePath: Path) -> None:
        self._dataset.import_assembled_patterns(filePath)

    def exportAssembledPatterns(self, filePath: Path) -> None:
        self._dataset.export_assembled_patterns(filePath)
