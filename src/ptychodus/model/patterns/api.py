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
        pattern_settings: PatternSettings,
        detector_settings: DetectorSettings,
        dataset: AssembledDiffractionDataset,
        file_reader_chooser: PluginChooser[DiffractionFileReader],
        file_writer_chooser: PluginChooser[DiffractionFileWriter],
    ) -> None:
        super().__init__()
        self._pattern_settings = pattern_settings
        self._detector_settings = detector_settings
        self._dataset = dataset
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser

    def create_streaming_context(self, metadata: DiffractionMetadata) -> PatternsStreamingContext:
        return PatternsStreamingContext(self._dataset, metadata)

    def get_file_reader_chooser(self) -> PluginChooser[DiffractionFileReader]:
        return self._file_reader_chooser

    def open_patterns(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
        detector_extent: ImageExtent | None = None,
    ) -> int:
        if crop_center is not None:
            self._pattern_settings.crop_center_x_px.set_value(crop_center.position_x_px)
            self._pattern_settings.crop_center_y_px.set_value(crop_center.position_y_px)

        if crop_extent is not None:
            self._pattern_settings.crop_width_px.set_value(crop_extent.width_px)
            self._pattern_settings.crop_height_px.set_value(crop_extent.height_px)

        if detector_extent is not None:
            self._detector_settings.width_px.set_value(detector_extent.width_px)
            self._detector_settings.height_px.set_value(detector_extent.height_px)

        if file_path.is_file():
            if file_type is not None:
                self._file_reader_chooser.set_current_plugin(file_type)

            file_type = self._file_reader_chooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{file_path}" as "{file_type}"')
            file_reader = self._file_reader_chooser.get_current_plugin().strategy

            try:
                dataset = file_reader.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                self._dataset.reload(dataset)
                return 0
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

        return -1

    def start_assembling_diffraction_patterns(self) -> None:
        self._dataset.start_loading()

    def finish_assembling_diffraction_patterns(self, *, block: bool) -> None:
        self._dataset.finish_loading(block=block)

        if block:
            self._dataset.assemble_patterns()

    def close_patterns(self) -> None:
        self._dataset.clear()

    def get_file_writer_chooser(self) -> PluginChooser[DiffractionFileWriter]:
        return self._file_writer_chooser

    def save_patterns(self, file_path: Path, file_type: str) -> None:
        self._file_writer_chooser.set_current_plugin(file_type)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        writer = self._file_writer_chooser.get_current_plugin().strategy
        writer.write(file_path, self._dataset)

    def import_assembled_patterns(self, file_path: Path) -> None:
        self._dataset.import_assembled_patterns(file_path)

    def export_assembled_patterns(self, file_path: Path) -> None:
        self._dataset.export_assembled_patterns(file_path)
