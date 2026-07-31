from pathlib import Path
import logging


from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    BadPixelsFileReader,
    CropCenter,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import AssembledDiffractionData
from ptychodus.api.tree import SimpleTreeNode

from .dataset import AssembledDiffractionDataset
from .repository import DiffractionDatasetRepository
from .settings import DiffractionSettings

logger = logging.getLogger(__name__)


class PatternsStreamingContext:
    def __init__(self, dataset: AssembledDiffractionDataset, metadata: DiffractionMetadata) -> None:
        self._dataset = dataset
        self._metadata = metadata

    def start(self) -> None:
        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        stream_dataset = SimpleDiffractionDataset(self._metadata, contents_tree, [])
        self._dataset.reload(stream_dataset)
        self._dataset.load_all_arrays(process_patterns=True, block=True)

    def append_array(self, array: DiffractionArray) -> None:
        self._dataset.append_array(array, process_patterns=True)

    def stop(self) -> None:
        pass


class DiffractionAPI:
    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        repository: DiffractionDatasetRepository,
        bad_pixels_file_reader_chooser: PluginChooser[BadPixelsFileReader],
        file_reader_chooser: PluginChooser[DiffractionFileReader],
        file_writer_chooser: PluginChooser[DiffractionFileWriter],
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings
        self._repository = repository
        self._bad_pixels_file_reader_chooser = bad_pixels_file_reader_chooser
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser

    def get_repository(self) -> DiffractionDatasetRepository:
        return self._repository

    def create_streaming_context(self, metadata: DiffractionMetadata) -> PatternsStreamingContext:
        dataset = self._repository.create_dataset('stream')
        self._repository.insert_dataset(dataset)
        return PatternsStreamingContext(dataset, metadata)

    def get_bad_pixels_file_reader_chooser(self) -> PluginChooser[BadPixelsFileReader]:
        return self._bad_pixels_file_reader_chooser

    def get_file_reader_chooser(self) -> PluginChooser[DiffractionFileReader]:
        return self._file_reader_chooser

    def open_patterns(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
        bad_pixels_file_path: Path | None = None,
        bad_pixels_file_type: str | None = None,
        process_patterns: bool = True,
        block: bool = False,
    ) -> int:
        if not file_path.is_file():
            logger.warning(f'Refusing to read invalid file path {file_path}')
            return -1

        if crop_center is not None:
            self._diffraction_settings.crop_center_x_px.set_value(crop_center.position_x_px)
            self._diffraction_settings.crop_center_y_px.set_value(crop_center.position_y_px)

        if crop_extent is not None:
            self._diffraction_settings.crop_width_px.set_value(crop_extent.width_px)
            self._diffraction_settings.crop_height_px.set_value(crop_extent.height_px)

        if file_type is not None:
            self._file_reader_chooser.set_current_plugin(file_type)

        plugin = self._file_reader_chooser.get_current_plugin()
        logger.debug(f'Reading "{file_path}" as "{plugin.simple_name}"')

        try:
            source_dataset = plugin.strategy.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        dataset = self._repository.create_dataset(file_path.stem)
        dataset_index = self._repository.insert_dataset(dataset)
        dataset.reload(source_dataset)

        if bad_pixels_file_path is not None:
            if bad_pixels_file_path.is_file():
                if bad_pixels_file_type is not None:
                    self._bad_pixels_file_reader_chooser.set_current_plugin(bad_pixels_file_type)
                bad_pixels_plugin = self._bad_pixels_file_reader_chooser.get_current_plugin()
                logger.debug(
                    f'Reading "{bad_pixels_file_path}" as "{bad_pixels_plugin.simple_name}"'
                )
                try:
                    bad_pixels = bad_pixels_plugin.strategy.read(bad_pixels_file_path)
                except Exception:
                    logger.warning(f'Failed to load bad pixels from "{bad_pixels_file_path}"')
                else:
                    try:
                        dataset.set_bad_pixels(bad_pixels)
                    except ValueError as exc:
                        logger.warning(f'Ignoring bad pixels from "{bad_pixels_file_path}": {exc}')
            else:
                logger.warning(
                    f'Refusing to read invalid bad pixels file path {bad_pixels_file_path}'
                )

        if block:
            dataset.load_all_arrays(process_patterns=process_patterns, block=True)

        return dataset_index

    def load_all_arrays(
        self, *, dataset_index: int, process_patterns: bool = True, block: bool = False
    ) -> None:
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot load arrays: no dataset at index {dataset_index}')
        else:
            dataset.load_all_arrays(process_patterns=process_patterns, block=block)

    def close_patterns(self, dataset_index: int) -> None:
        self._repository.remove_dataset(dataset_index)

    def close_all_patterns(self) -> None:
        self._repository.clear()

    def get_file_writer_chooser(self) -> PluginChooser[DiffractionFileWriter]:
        return self._file_writer_chooser

    def save_patterns(self, file_path: Path, file_type: str, *, dataset_index: int) -> None:
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot save patterns: no dataset at index {dataset_index}')
            return

        self._file_writer_chooser.set_current_plugin(file_type)
        plugin = self._file_writer_chooser.get_current_plugin()
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, dataset)

    def get_assembled_data(self, dataset_index: int) -> AssembledDiffractionData:
        return self._repository[dataset_index].get_assembled_data()

    def import_assembled_patterns(self, file_path: Path) -> int:
        if not file_path.is_file():
            logger.warning(f'Refusing to read invalid file path {file_path}')
            return -1

        dataset = self._repository.create_dataset(file_path.stem)
        dataset_index = self._repository.insert_dataset(dataset)
        dataset.import_assembled_patterns(file_path)
        return dataset_index

    def export_assembled_patterns(self, file_path: Path, *, dataset_index: int) -> None:
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot export patterns: no dataset at index {dataset_index}')
            return
        dataset.export_assembled_patterns(file_path)
