from __future__ import annotations
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

from ptychodus.api.plugins import PluginRegistry

from .agent import AgentCore
from .analysis import AnalysisCore
from .diffraction import DiffractionCore
from .fluorescence import FluorescenceCore
from .genesis import GenesisCore
from .globus import GlobusCore
from .memory import MemoryPresenter
from .plugins import PluginChooser
from .processing import ProcessingCore
from .product import ProductCore
from .settings import SettingsRegistry
from .task_manager import TaskManager

logger = logging.getLogger(__name__)


def configure_logger(*, log_level: int) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
        level=log_level,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info('Ptychodus ' + version('ptychodus'))
    logger.info('NumPy ' + version('numpy'))
    logger.info('Matplotlib ' + version('matplotlib'))
    logger.info('Colorcet ' + version('colorcet'))
    logger.info('HDF5Plugin ' + version('hdf5plugin'))
    logger.info('H5Py ' + version('h5py'))
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


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
            PluginChooser(self.plugin_registry.bad_pixels_file_readers),
            PluginChooser(self.plugin_registry.diffraction_file_readers),
            PluginChooser(self.plugin_registry.diffraction_file_writers),
            self.settings_registry,
            PluginChooser(self.plugin_registry.background_file_readers),
            PluginChooser(self.plugin_registry.flat_field_file_readers),
        )
        self.product_core = ProductCore(
            self.rng,
            self.settings_registry,
            self.diffraction_core.diffraction_api,
            self.plugin_registry.probe_position_generators,
            PluginChooser(self.plugin_registry.probe_position_file_readers),
            PluginChooser(self.plugin_registry.probe_position_file_writers),
            self.plugin_registry.probe_generators,
            self.plugin_registry.fresnel_zone_plates,
            self.plugin_registry.kb_mirrors,
            PluginChooser(self.plugin_registry.probe_file_readers),
            PluginChooser(self.plugin_registry.probe_file_writers),
            self.plugin_registry.object_generators,
            PluginChooser(self.plugin_registry.object_file_readers),
            PluginChooser(self.plugin_registry.object_file_writers),
            PluginChooser(self.plugin_registry.product_file_readers),
            PluginChooser(self.plugin_registry.product_file_writers),
            self.settings_registry,
            self._task_manager,
        )
        self.processing_core = ProcessingCore(
            self._task_manager,
            self.settings_registry,
            self.product_core.product_api,
            PluginChooser(self.plugin_registry.reconstructor_libraries),
            self.plugin_registry.training_data_file_writers,
        )
        self.fluorescence_core = FluorescenceCore(
            self._task_manager,
            self.settings_registry,
            self.product_core.product_api,
            self.product_core.product_repository,
            PluginChooser(self.plugin_registry.fluorescence_enhancers),
            self.processing_core.device_selection,
            PluginChooser(self.plugin_registry.fluorescence_file_readers),
            PluginChooser(self.plugin_registry.fluorescence_file_writers),
            self.settings_registry,
        )
        self.analysis_core = AnalysisCore(
            self.rng,
            self.settings_registry,
            self.diffraction_core.repository,
            self.product_core.product_repository,
            self.product_core.probe_positions_repository,
            self.product_core.product_api,
        )
        self.globus_core = GlobusCore(
            self.settings_registry,
            self.product_core.product_api,
            self.processing_core.processing_api,
        )
        self.genesis_core = GenesisCore(
            self._task_manager,
            self.settings_registry,
            self.product_core.product_api,
            self.processing_core.processing_api,
        )
        self.agent_core = AgentCore(
            self.settings_registry,
            PluginChooser(self.plugin_registry.language_model_providers),
        )

        if settings_file:
            self.settings_registry.open_settings(settings_file)

    def __enter__(self) -> ModelCore:
        self._task_manager.start()
        self.genesis_core.start()
        self.globus_core.start()
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
        self.globus_core.stop()
        self.genesis_core.stop()
        self._task_manager.stop(await_finish=False)

    def run_tasks(self) -> None:
        self._task_manager.run_foreground_tasks()
        self.globus_core.run_foreground_tasks()
        self.genesis_core.run_foreground_tasks()

    @property
    def is_developer_mode_enabled(self) -> bool:
        return logger.getEffectiveLevel() <= logging.DEBUG
