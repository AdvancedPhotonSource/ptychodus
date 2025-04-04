from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import overload
import logging
import sys

try:
    # NOTE must import hdf5plugin before h5py
    import hdf5plugin  # noqa
except ModuleNotFoundError:
    pass

import h5py
import numpy

from ptychodus.api.patterns import DiffractionMetadata, DiffractionPatternArray
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.workflow import WorkflowAPI

from .agent import AgentCore
from .analysis import AnalysisCore
from .automation import AutomationCore
from .fluorescence import FluorescenceCore
from .memory import MemoryPresenter
from .metadata import MetadataPresenter
from .patterns import PatternsCore, PatternsStreamingContext
from .product import PositionsStreamingContext, ProductCore
from .ptychi import PtyChiReconstructorLibrary
from .ptychonn import PtychoNNReconstructorLibrary
from .ptychopinn import PtychoPINNReconstructorLibrary
from .reconstructor import ReconstructorCore
from .tike import TikeReconstructorLibrary
from .visualization import VisualizationEngine
from .workflow import WorkflowCore

logger = logging.getLogger(__name__)


def configure_logger(is_developer_mode_enabled: bool) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
        encoding='utf-8',
        level=logging.DEBUG if is_developer_mode_enabled else logging.INFO,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info(f'Ptychodus {version("ptychodus")}')
    logger.info(f'NumPy {version("numpy")}')
    logger.info(f'Matplotlib {version("matplotlib")}')
    logger.info(f'HDF5Plugin {version("hdf5plugin")}')
    logger.info(f'H5Py {version("h5py")}')
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


class PtychodusStreamingContext:
    def __init__(
        self,
        positions_context: PositionsStreamingContext,
        patterns_context: PatternsStreamingContext,
    ) -> None:
        self._positions_context = positions_context
        self._patterns_context = patterns_context

    def start(self) -> None:
        self._positions_context.start()
        self._patterns_context.start()

    def append_positions_x(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_context.append_positions_x(values_m, trigger_counts)

    def append_positions_y(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_context.append_positions_y(values_m, trigger_counts)

    def append_array(self, array: DiffractionPatternArray) -> None:
        self._patterns_context.append_array(array)

    def get_queue_size(self) -> int:
        return self._patterns_context.get_queue_size()

    def stop(self) -> None:
        self._patterns_context.stop()
        self._positions_context.stop()


class ModelCore:
    def __init__(
        self, settings_file: Path | None = None, *, is_developer_mode_enabled: bool = False
    ) -> None:
        configure_logger(is_developer_mode_enabled)
        self.rng = numpy.random.default_rng()
        self._plugin_registry = PluginRegistry.load_plugins()

        self.memory_presenter = MemoryPresenter()
        self.settings_registry = SettingsRegistry()

        self.patterns = PatternsCore(
            self.settings_registry,
            self._plugin_registry.diffraction_file_readers,
            self._plugin_registry.diffraction_file_writers,
            self.settings_registry,
        )
        self.product = ProductCore(
            self.rng,
            self.settings_registry,
            self.patterns.pattern_sizer,
            self.patterns.dataset,
            self._plugin_registry.position_file_readers,
            self._plugin_registry.position_file_writers,
            self._plugin_registry.fresnel_zone_plates,
            self._plugin_registry.probe_file_readers,
            self._plugin_registry.probe_file_writers,
            self._plugin_registry.object_file_readers,
            self._plugin_registry.object_file_writers,
            self._plugin_registry.product_file_readers,
            self._plugin_registry.product_file_writers,
            self.settings_registry,
        )
        self.metadata_presenter = MetadataPresenter(
            self.patterns.detector_settings,
            self.patterns.pattern_settings,
            self.patterns.dataset,
            self.product.settings,
        )

        self.pattern_visualization_engine = VisualizationEngine(is_complex=False)
        self.probe_visualization_engine = VisualizationEngine(is_complex=True)
        self.object_visualization_engine = VisualizationEngine(is_complex=True)

        self.ptychi_reconstructor_library = PtyChiReconstructorLibrary(
            self.settings_registry, self.patterns.pattern_sizer, is_developer_mode_enabled
        )
        self.tike_reconstructor_library = TikeReconstructorLibrary.create_instance(
            self.settings_registry, is_developer_mode_enabled
        )
        self.ptychonn_reconstructor_library = PtychoNNReconstructorLibrary.create_instance(
            self.settings_registry, is_developer_mode_enabled
        )
        self.ptychopinn_reconstructor_library = PtychoPINNReconstructorLibrary(
            self.settings_registry, is_developer_mode_enabled
        )
        self.reconstructor = ReconstructorCore(
            self.settings_registry,
            self.patterns.dataset,
            self.product.product_api,
            [
                self.ptychi_reconstructor_library,
                self.tike_reconstructor_library,
                self.ptychonn_reconstructor_library,
                self.ptychopinn_reconstructor_library,
            ],
        )
        self.fluorescence_core = FluorescenceCore(
            self.settings_registry,
            self.product.product_repository,
            self._plugin_registry.upscaling_strategies,
            self._plugin_registry.deconvolution_strategies,
            self._plugin_registry.fluorescence_file_readers,
            self._plugin_registry.fluorescence_file_writers,
        )
        self.analysis = AnalysisCore(
            self.settings_registry,
            self.reconstructor.data_matcher,
            self.product.product_repository,
            self.product.object_repository,
        )
        self.workflow = WorkflowCore(
            self.settings_registry,
            self.patterns.patterns_api,
            self.product.product_api,
            self.product.scan_api,
            self.product.probe_api,
            self.product.object_api,
            self.reconstructor.reconstructor_api,
        )
        self.automation = AutomationCore(
            self.settings_registry,
            self.workflow.workflow_api,
            self._plugin_registry.file_based_workflows,
        )
        self.agent = AgentCore(self.settings_registry)

        if settings_file:
            self.settings_registry.open_settings(settings_file)

    def __enter__(self) -> ModelCore:
        self.patterns.start()
        self.reconstructor.start()
        self.workflow.start()
        self.automation.start()
        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None: ...

    @overload
    def __exit__(
        self,
        exception_type: type[BaseException],
        exception_value: BaseException,
        traceback: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.automation.stop()
        self.workflow.stop()
        self.reconstructor.stop()
        self.patterns.stop()

    def create_streaming_context(self, metadata: DiffractionMetadata) -> PtychodusStreamingContext:
        return PtychodusStreamingContext(
            self.product.scan_api.create_streaming_context(),
            self.patterns.patterns_api.create_streaming_context(metadata),
        )

    def refresh_active_dataset(self) -> None:
        self.patterns.dataset.assemble_patterns()

    def batch_mode_execute(
        self,
        action: str,
        input_path: Path,
        output_path: Path,
        *,
        product_file_type: str = 'NPZ',
        fluorescence_input_file_path: Path | None = None,
        fluorescence_output_file_path: Path | None = None,
    ) -> int:
        # TODO add enum for actions; implement using workflow API
        if action.lower() == 'train':
            output = self.reconstructor.reconstructor_api.train(input_path)
            self.reconstructor.reconstructor_api.save_model(output_path)
            return output.result

        input_product_index = self.product.product_api.open_product(
            input_path, file_type=product_file_type
        )

        if input_product_index < 0:
            logger.error(f'Failed to open product "{input_path}"!')
            return -1

        if action.lower() == 'reconstruct':
            logger.info('Reconstructing...')
            output_product_index = self.reconstructor.reconstructor_api.reconstruct(
                input_product_index
            )
            self.reconstructor.reconstructor_api.process_results(block=True)
            logger.info('Reconstruction complete.')

            self.product.product_api.save_product(
                output_product_index, output_path, file_type=product_file_type
            )

            if (
                fluorescence_input_file_path is not None
                and fluorescence_output_file_path is not None
            ):
                self.fluorescence_core.enhance_fluorescence(
                    output_product_index,
                    fluorescence_input_file_path,
                    fluorescence_output_file_path,
                )
        elif action.lower() == 'prepare_training_data':
            self.reconstructor.reconstructor_api.export_training_data(
                output_path, input_product_index
            )
        else:
            logger.error(f'Unknown batch mode action "{action}"!')
            return -1

        return 0

    @property
    def workflow_api(self) -> WorkflowAPI:
        return self.workflow.workflow_api
