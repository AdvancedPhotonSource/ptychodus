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

from ptychodus.api.diffraction import DiffractionMetadata, DiffractionArray
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.workflow import WorkflowAPI

from .agent import AgentCore
from .analysis import AnalysisCore
from .automation import AutomationCore
from .diffraction import DiffractionCore, PatternsStreamingContext
from .fluorescence import FluorescenceCore
from .globus import GlobusCore
from .memory import MemoryPresenter
from .metadata import MetadataPresenter
from .product import PositionsStreamingContext, ProductCore
from .ptychi import PtyChiReconstructorLibrary
from .ptychonn import PtychoNNReconstructorLibrary
from .ptychopinn import PtychoPINNReconstructorLibrary
from .reconstructor import ReconstructorCore
from .task_manager import TaskManager
from .visualization import VisualizationEngine
from .workflow import ConcreteWorkflowAPI

logger = logging.getLogger(__name__)


def configure_logger(*, log_level: int) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
        encoding='utf-8',
        level=log_level,
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

    def append_array(self, array: DiffractionArray) -> None:
        self._patterns_context.append_array(array)

    def get_queue_size(self) -> int:
        return 0  # TODO get_queue_size

    def stop(self) -> None:
        self._patterns_context.stop()
        self._positions_context.stop()


class ModelCore:
    def __init__(
        self, settings_file: Path | None = None, *, log_level: int = logging.WARNING
    ) -> None:
        configure_logger(log_level=log_level)

        self.rng = numpy.random.default_rng()
        self.plugin_registry = PluginRegistry.load_plugins()
        self._task_manager = TaskManager()

        self.memory_presenter = MemoryPresenter()
        self.settings_registry = SettingsRegistry()

        self.diffraction_core = DiffractionCore(
            self._task_manager,
            self.settings_registry,
            self.plugin_registry.bad_pixels_file_readers,
            self.plugin_registry.diffraction_file_readers,
            self.plugin_registry.diffraction_file_writers,
            self.settings_registry,
        )
        self.product_core = ProductCore(
            self.rng,
            self.settings_registry,
            self.diffraction_core.pattern_sizer,
            self.diffraction_core.dataset,
            self.plugin_registry.probe_position_file_readers,
            self.plugin_registry.probe_position_file_writers,
            self.plugin_registry.fresnel_zone_plates,
            self.plugin_registry.probe_file_readers,
            self.plugin_registry.probe_file_writers,
            self.plugin_registry.object_file_readers,
            self.plugin_registry.object_file_writers,
            self.plugin_registry.product_file_readers,
            self.plugin_registry.product_file_writers,
            self.settings_registry,
        )
        self.metadata_presenter = MetadataPresenter(
            self.diffraction_core.detector_settings,
            self.diffraction_core.diffraction_settings,
            self.diffraction_core.dataset,
            self.product_core.settings,
        )

        self.pattern_visualization_engine = VisualizationEngine(is_complex=False)
        self.probe_visualization_engine = VisualizationEngine(is_complex=True)
        self.object_visualization_engine = VisualizationEngine(is_complex=True)

        self.ptychi_reconstructor_library = PtyChiReconstructorLibrary(
            self.settings_registry,
            self.diffraction_core.pattern_sizer,
            self.is_developer_mode_enabled,
        )
        self.ptychonn_reconstructor_library = PtychoNNReconstructorLibrary.create_instance(
            self.settings_registry, self.is_developer_mode_enabled
        )
        self.ptychopinn_reconstructor_library = PtychoPINNReconstructorLibrary(
            self.settings_registry, self.is_developer_mode_enabled
        )
        self.reconstructor_core = ReconstructorCore(
            self._task_manager,
            self.settings_registry,
            self.diffraction_core.dataset,
            self.product_core.product_api,
            [
                self.ptychi_reconstructor_library,
                self.ptychonn_reconstructor_library,
                self.ptychopinn_reconstructor_library,
            ],
        )
        self.fluorescence_core = FluorescenceCore(
            self.settings_registry,
            self.product_core.product_repository,
            self.plugin_registry.upscaling_strategies,
            self.plugin_registry.deconvolution_strategies,
            self.plugin_registry.fluorescence_file_readers,
            self.plugin_registry.fluorescence_file_writers,
        )
        self.analysis_core = AnalysisCore(
            self.settings_registry,
            self.reconstructor_core.data_matcher,
            self.product_core.product_repository,
            self.product_core.object_repository,
        )
        self.globus_core = GlobusCore(
            self.settings_registry,
            self.diffraction_core.diffraction_api,
            self.product_core.product_api,
        )
        self.workflow_api: WorkflowAPI = ConcreteWorkflowAPI(
            self.settings_registry,
            self.diffraction_core.diffraction_api,
            self.product_core.product_api,
            self.product_core.probe_positions_api,
            self.product_core.probe_api,
            self.product_core.object_api,
            self.reconstructor_core.reconstructor_api,
            self.globus_core.executor,
        )
        self.automation_core = AutomationCore(
            self.settings_registry,
            self.workflow_api,
            self.plugin_registry.file_based_workflows,
        )
        self.agent_core = AgentCore(self.settings_registry)

        if settings_file:
            self.settings_registry.open_settings(settings_file)

    def __enter__(self) -> ModelCore:
        self._task_manager.start()
        self.globus_core.start()
        self.automation_core.start()
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
        self.automation_core.stop()
        self.globus_core.stop()
        self._task_manager.stop(await_finish=False)

    def create_streaming_context(self, metadata: DiffractionMetadata) -> PtychodusStreamingContext:
        return PtychodusStreamingContext(
            self.product_core.probe_positions_api.create_streaming_context(),
            self.diffraction_core.diffraction_api.create_streaming_context(metadata),
        )

    def run_tasks(self) -> None:
        self._task_manager.run_foreground_tasks()
        self.reconstructor_core.notify_observers_if_progress_changed()

    def batch_mode_execute(
        self,
        action: str,
        input_path: Path,
        output_path: Path,
        *,
        fluorescence_input_file_path: Path | None = None,
        fluorescence_output_file_path: Path | None = None,
    ) -> int:
        # TODO add enum for actions
        if action.lower() == 'train':
            output = self.workflow_api.train_reconstructor(input_path, output_path)
            return output.result

        if action.lower() == 'reconstruct':
            input_product_api = self.workflow_api.open_product(input_path)
            output_product_api = input_product_api.reconstruct_local(block=True)
            output_product_api.save_product(output_path)

            if (
                fluorescence_input_file_path is not None
                and fluorescence_output_file_path is not None
            ):
                # TODO implement using workflow API
                self.fluorescence_core.enhance_fluorescence(
                    output_product_api.get_product_index(),
                    fluorescence_input_file_path,
                    fluorescence_output_file_path,
                )
        else:
            logger.error(f'Unknown batch mode action "{action}"!')
            return -1

        return 0

    @property
    def is_developer_mode_enabled(self) -> bool:
        return logger.getEffectiveLevel() <= logging.DEBUG
