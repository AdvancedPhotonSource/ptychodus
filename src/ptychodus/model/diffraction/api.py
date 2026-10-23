from pathlib import Path
import logging

import numpy

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import (
    BadPixels,
    BadPixelsFileReader,
    DetectorBackground,
    CropRegion,
    DiffractionArray,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    FlatField,
    PixelMapFileReader,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginChooser

from ..file_types import FileTypeChooser
from ..parameters import PluginChooserParameter
from .dataset import AssembledDiffractionDataset
from .repository import DiffractionDatasetRepository
from .settings import DiffractionSettings

logger = logging.getLogger(__name__)


class PatternsStreamingContext:
    def __init__(self, dataset: AssembledDiffractionDataset, metadata: DiffractionMetadata) -> None:
        self._dataset = dataset
        self._metadata = metadata

    def start(self) -> None:
        contents_tree = DiffractionDatasetLayoutNode.create_root()
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
        bad_pixels_file_reader_parameter: PluginChooserParameter[BadPixelsFileReader],
        file_reader_parameter: PluginChooserParameter[DiffractionFileReader],
        background_file_reader_chooser: PluginChooser[PixelMapFileReader],
        flat_field_file_reader_chooser: PluginChooser[PixelMapFileReader],
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings
        self._repository = repository
        self._bad_pixels_file_reader_chooser = bad_pixels_file_reader_chooser
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self.file_writers = FileTypeChooser(file_writer_chooser)
        self._bad_pixels_file_reader_parameter = bad_pixels_file_reader_parameter
        self._file_reader_parameter = file_reader_parameter
        self._background_file_reader_chooser = background_file_reader_chooser
        self._flat_field_file_reader_chooser = flat_field_file_reader_chooser

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

    def get_bad_pixels_file_reader_parameter(self) -> PluginChooserParameter[BadPixelsFileReader]:
        return self._bad_pixels_file_reader_parameter

    def get_file_reader_parameter(self) -> PluginChooserParameter[DiffractionFileReader]:
        return self._file_reader_parameter

    def open_patterns(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_region: CropRegion | None = None,
        bad_pixels_file_path: Path | None = None,
        bad_pixels_file_type: str | None = None,
        background_file_path: Path | None = None,
        background_file_type: str | None = None,
        flat_field_file_path: Path | None = None,
        flat_field_file_type: str | None = None,
        process_patterns: bool = True,
        block: bool = False,
    ) -> int:
        if not file_path.is_file():
            logger.warning(f'Refusing to read invalid file path {file_path}')
            return -1

        if crop_region is not None:
            self._diffraction_settings.beam_center_x_px.set_value(crop_region.center_x_px)
            self._diffraction_settings.beam_center_y_px.set_value(crop_region.center_y_px)
            self._diffraction_settings.crop_width_px.set_value(crop_region.width_px)
            self._diffraction_settings.crop_height_px.set_value(crop_region.height_px)

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
            self._apply_bad_pixels_from_file(dataset, bad_pixels_file_path, bad_pixels_file_type)

        if background_file_path is not None:
            dataset.set_background(self.load_background(background_file_path, background_file_type))

        if flat_field_file_path is not None:
            dataset.set_flat_field(self.load_flat_field(flat_field_file_path, flat_field_file_type))

        if block:
            dataset.load_all_arrays(process_patterns=process_patterns, block=True)

        return dataset_index

    def load_bad_pixels(self, file_path: Path, file_type: str | None = None) -> BadPixels:
        """Load a bad-pixel mask from disk without applying it to any dataset.

        The wizard's Summary panel needs the mask before Finish so it can pass
        it to `summarize_dataset` (whose inpainting keeps the mean pattern
        smooth for `estimate_beam_center`) without mutating a dataset that
        hasn't been loaded yet.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f'Invalid bad pixels file path "{file_path}"')

        if file_type is not None:
            self._bad_pixels_file_reader_chooser.set_current_plugin(file_type)
        bad_pixels_plugin = self._bad_pixels_file_reader_chooser.get_current_plugin()
        logger.debug(f'Reading "{file_path}" as "{bad_pixels_plugin.simple_name}"')
        try:
            return bad_pixels_plugin.strategy.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to load bad pixels from "{file_path}"') from exc

    def _apply_bad_pixels_from_file(
        self,
        dataset: AssembledDiffractionDataset,
        file_path: Path,
        file_type: str | None,
    ) -> None:
        dataset.set_bad_pixels(self.load_bad_pixels(file_path, file_type))

    def apply_bad_pixels(
        self,
        dataset_index: int,
        file_path: Path,
        file_type: str | None = None,
    ) -> None:
        """Load a bad-pixel mask from disk and apply it to an already-open dataset."""
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot apply bad pixels: no dataset at index {dataset_index}')
            return
        self._apply_bad_pixels_from_file(dataset, file_path, file_type)

    def _read_pixel_map(
        self,
        description: str,
        chooser: PluginChooser[PixelMapFileReader],
        file_path: Path,
        file_type: str | None,
    ) -> numpy.ndarray:
        if not file_path.is_file():
            raise FileNotFoundError(f'Invalid {description} file path "{file_path}"')

        if file_type is not None:
            chooser.set_current_plugin(file_type)

        plugin = chooser.get_current_plugin()
        logger.debug(f'Reading "{file_path}" as "{plugin.simple_name}"')

        try:
            return plugin.strategy.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to load {description} from "{file_path}"') from exc

    def load_background(self, file_path: Path, file_type: str | None = None) -> DetectorBackground:
        """Load a measured detector background from disk without applying it."""
        return self._read_pixel_map(
            'background', self._background_file_reader_chooser, file_path, file_type
        )

    def load_flat_field(self, file_path: Path, file_type: str | None = None) -> FlatField:
        """Load a flat-field response from disk without applying it."""
        return self._read_pixel_map(
            'flat field', self._flat_field_file_reader_chooser, file_path, file_type
        )

    def apply_background(
        self, dataset_index: int, file_path: Path, file_type: str | None = None
    ) -> None:
        """Load a background from disk and give it to an already-open dataset.

        It replaces any background the dataset's source carried.
        """
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot apply background: no dataset at index {dataset_index}')
            return

        dataset.set_background(self.load_background(file_path, file_type))

    def apply_flat_field(
        self, dataset_index: int, file_path: Path, file_type: str | None = None
    ) -> None:
        """Load a flat field from disk and give it to an already-open dataset.

        It replaces any flat field the dataset's source carried.
        """
        try:
            dataset = self._repository[dataset_index]
        except IndexError:
            logger.warning(f'Cannot apply flat field: no dataset at index {dataset_index}')
            return

        dataset.set_flat_field(self.load_flat_field(file_path, file_type))

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

        plugin = self.file_writers.select(file_type)
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
