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

    def start(self) -> None:
        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        stream_dataset = SimpleDiffractionDataset(self._metadata, contents_tree, [])
        self._dataset.reload(stream_dataset)
        self._dataset.start_loading()

    def append_array(self, array: DiffractionPatternArray) -> None:
        self._dataset.append_array(array)

    def get_queue_size(self) -> int:
        return self._dataset.queue_size

    def stop(self) -> None:
        self._dataset.finish_loading(block=True)
        self._dataset.assemble_patterns()


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

    def getFileReaderChooser(self) -> PluginChooser[DiffractionFileReader]:
        return self._fileReaderChooser

    def open_patterns(
        self,
        filePath: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
        detectorExtent: ImageExtent | None = None,
    ) -> int:
        if crop_center is not None:
            self._patternSettings.cropCenterXInPixels.set_value(crop_center.position_x_px)
            self._patternSettings.cropCenterYInPixels.set_value(crop_center.position_y_px)

        if crop_extent is not None:
            self._patternSettings.cropWidthInPixels.set_value(crop_extent.width_px)
            self._patternSettings.cropHeightInPixels.set_value(crop_extent.height_px)

        if detectorExtent is not None:
            self._detectorSettings.widthInPixels.set_value(detectorExtent.width_px)
            self._detectorSettings.heightInPixels.set_value(detectorExtent.height_px)

        if filePath.is_file():
            if file_type is not None:
                self._fileReaderChooser.set_current_plugin(file_type)

            file_type = self._fileReaderChooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{filePath}" as "{file_type}"')
            fileReader = self._fileReaderChooser.get_current_plugin().strategy

            try:
                dataset = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                self._dataset.reload(dataset)
                return 0
        else:
            logger.warning(f'Refusing to read invalid file path {filePath}')

        return -1

    def startAssemblingDiffractionPatterns(self) -> None:
        self._dataset.start_loading()

    def finishAssemblingDiffractionPatterns(self, *, block: bool) -> None:
        self._dataset.finish_loading(block=block)

        if block:
            self._dataset.assemble_patterns()

    def closePatterns(self) -> None:
        self._dataset.clear()

    def getFileWriterChooser(self) -> PluginChooser[DiffractionFileWriter]:
        return self._fileWriterChooser

    def savePatterns(self, filePath: Path, fileType: str) -> None:
        self._fileWriterChooser.set_current_plugin(fileType)
        fileType = self._fileWriterChooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        writer = self._fileWriterChooser.get_current_plugin().strategy
        writer.write(filePath, self._dataset)

    def import_assembled_patterns(self, filePath: Path) -> None:
        self._dataset.import_assembled_patterns(filePath)

    def export_assembled_patterns(self, filePath: Path) -> None:
        self._dataset.export_assembled_patterns(filePath)
