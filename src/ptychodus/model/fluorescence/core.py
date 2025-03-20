from __future__ import annotations
from collections.abc import Iterator
from pathlib import Path
import logging


from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancingAlgorithm,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductRepository, ProductRepositoryItem
from ..visualization import VisualizationEngine
from .settings import FluorescenceSettings
from .two_step import TwoStepFluorescenceEnhancingAlgorithm
from .vspi import VSPIFluorescenceEnhancingAlgorithm

logger = logging.getLogger(__name__)


class FluorescenceEnhancer(Observable, Observer):
    def __init__(
        self,
        settings: FluorescenceSettings,
        product_repository: ProductRepository,
        two_step_enhancing_algorithm: TwoStepFluorescenceEnhancingAlgorithm,
        vspi_enhancing_algorithm: VSPIFluorescenceEnhancingAlgorithm,
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
        reinit_observable: Observable,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._product_repository = product_repository
        self.two_step_enhancing_algorithm = two_step_enhancing_algorithm
        self.vspi_enhancing_algorithm = vspi_enhancing_algorithm
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._reinit_observable = reinit_observable

        self._algorithm_chooser = PluginChooser[FluorescenceEnhancingAlgorithm]()
        self._algorithm_chooser.register_plugin(
            two_step_enhancing_algorithm,
            simple_name=TwoStepFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            display_name=TwoStepFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._algorithm_chooser.register_plugin(
            vspi_enhancing_algorithm,
            simple_name=VSPIFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            display_name=VSPIFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._algorithm_chooser.synchronize_with_parameter(settings.algorithm)
        self._algorithm_chooser.add_observer(self)

        self._product_index = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        file_reader_chooser.synchronize_with_parameter(settings.file_type)
        file_writer_chooser.set_current_plugin(settings.file_type.get_value())
        reinit_observable.add_observer(self)

    @property
    def _product(self) -> ProductRepositoryItem:
        return self._product_repository[self._product_index]

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._enhanced = None
            self.notify_observers()

    def get_product_name(self) -> str:
        return self._product.get_name()

    def get_pixel_geometry(self) -> PixelGeometry:
        return self._product.get_geometry().get_object_plane_pixel_geometry()

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def open_measured_dataset(self, file_path: Path, file_filter: str) -> None:
        if file_path.is_file():
            self._file_reader_chooser.set_current_plugin(file_filter)
            file_type = self._file_reader_chooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{file_path}" as "{file_type}"')
            file_reader = self._file_reader_chooser.get_current_plugin().strategy

            try:
                measured = file_reader.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                self._measured = measured
                self._enhanced = None

                self._settings.file_path.set_value(file_path)

                self.notify_observers()
        else:
            logger.warning(f'Refusing to load dataset from invalid file path "{file_path}"')

    def get_num_channels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def get_measured_element_map(self, channel_index: int) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channel_index]

    def algorithms(self) -> Iterator[str]:
        for plugin in self._algorithm_chooser:
            yield plugin.display_name

    def get_algorithm(self) -> str:
        return self._algorithm_chooser.get_current_plugin().display_name

    def set_algorithm(self, name: str) -> None:
        self._algorithm_chooser.set_current_plugin(name)

    def enhance_fluorescence(self) -> None:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')
        else:
            algorithm = self._algorithm_chooser.get_current_plugin().strategy
            product = self._product.get_product()
            self._enhanced = algorithm.enhance(self._measured, product)
            self.notify_observers()

    def get_enhanced_element_map(self, channel_index: int) -> ElementMap:
        if self._enhanced is None:
            return self.get_measured_element_map(channel_index)

        return self._enhanced.element_maps[channel_index]

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_enhanced_dataset(self, file_path: Path, file_filter: str) -> None:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        self._file_writer_chooser.set_current_plugin(file_filter)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        writer = self._file_writer_chooser.get_current_plugin().strategy
        writer.write(file_path, self._enhanced)

    def _open_fluorescence_file_from_settings(self) -> None:
        self.open_measured_dataset(
            self._settings.file_path.get_value(), self._settings.file_type.get_value()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm_chooser:
            self.notify_observers()
        elif observable is self._reinit_observable:
            self._open_fluorescence_file_from_settings()


class FluorescenceCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        product_repository: ProductRepository,
        upscaling_strategy_chooser: PluginChooser[UpscalingStrategy],
        deconvolution_strategy_chooser: PluginChooser[DeconvolutionStrategy],
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self._settings = FluorescenceSettings(settings_registry)
        self._two_step_enhancing_algorithm = TwoStepFluorescenceEnhancingAlgorithm(
            self._settings, upscaling_strategy_chooser, deconvolution_strategy_chooser
        )
        self._vspi_enhancing_algorithm = VSPIFluorescenceEnhancingAlgorithm(self._settings)

        self.enhancer = FluorescenceEnhancer(
            self._settings,
            product_repository,
            self._two_step_enhancing_algorithm,
            self._vspi_enhancing_algorithm,
            file_reader_chooser,
            file_writer_chooser,
            settings_registry,
        )
        self.visualization_engine = VisualizationEngine(is_complex=False)

    def enhance_fluorescence(
        self, product_index: int, input_file_path: Path, output_file_path: Path
    ) -> int:
        file_type = 'XRF-Maps'

        try:
            self.enhancer.set_product(product_index)
            self.enhancer.open_measured_dataset(input_file_path, file_type)
            self.enhancer.enhance_fluorescence()
            self.enhancer.save_enhanced_dataset(output_file_path, file_type)
        except Exception as exc:
            logger.exception(exc)
            return -1

        return 0
