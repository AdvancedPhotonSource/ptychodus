from collections.abc import Mapping, Sequence
import logging

from ptychodus.api.fluorescence import (
    FluorescenceEnhancer,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    SubprocessFluorescenceEnhancer,
)

from ..devices import DeviceSelection
from ..observer import Observable, Observer
from ..plugins import PluginChooser, PluginChooserParameter
from ..processing.parameter_store import OptionsBranch, ProcessingParameterStore
from ..product import ProductAPI, ProductRepository
from ..settings import SettingsRegistry
from ..task_manager import TaskManager
from .api import FluorescenceAPI
from .monitor import FluorescenceTaskMonitor
from .repository import FluorescenceRepository
from .settings import ENHANCER_LIBRARY, FluorescenceSettings

logger = logging.getLogger(__name__)

# Which group backs each enhancer, keyed by the name it registered under. The same
# table the reconstructor backends have: the fields carry their own bounds and
# display text, the field name is the key, and this says only where each enhancer's
# options persist. One group each, so no two enhancers share a key.
_BRANCHES_BY_ENHANCER: Mapping[str, Sequence[OptionsBranch]] = {
    'two_step': (OptionsBranch(f'{ENHANCER_LIBRARY}TwoStep'),),
    'vspi': (OptionsBranch(f'{ENHANCER_LIBRARY}VSPI'),),
    'ptychozoon': (OptionsBranch(f'{ENHANCER_LIBRARY}Ptychozoon'),),
}


class FluorescenceCore(Observer):
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        product_repository: ProductRepository,
        enhancer_chooser: PluginChooser[FluorescenceEnhancer],
        device_selection: DeviceSelection,
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
        reinit_observable: Observable,
    ) -> None:
        super().__init__()
        self._product_repository = product_repository
        self.settings = FluorescenceSettings(settings_registry)
        self.enhancer_parameter_store = ProcessingParameterStore(settings_registry)
        self.task_monitor = FluorescenceTaskMonitor(task_manager)
        log_handler = self.task_monitor.get_log_handler()
        package_loggers = {'ptychodus.model.fluorescence'}

        # Every enhancer, not only the selected one: the settings file must hold the
        # same groups however the application was driven.
        for plugin in enhancer_chooser:
            enhancer = plugin.strategy
            branches = _BRANCHES_BY_ENHANCER.get(plugin.simple_name)

            if branches is None:
                logger.warning(
                    f'No settings group was named for enhancer "{plugin.simple_name}"; its '
                    'options will not be editable and every run will use their defaults.'
                )

            self.enhancer_parameter_store.register_declared(
                ENHANCER_LIBRARY,
                plugin.simple_name,
                enhancer.get_default_options(),
                branches or (),
            )

            # Placement is ptychodus's to decide, so it is attached here rather than
            # declared by the backend or passed through every call.
            if isinstance(enhancer, SubprocessFluorescenceEnhancer):
                enhancer.device_environment = device_selection.environment

            # The package rather than the module, so a backend's forwarded child logs
            # reach the monitor too.
            package_loggers.add(type(enhancer).__module__.rpartition('.')[0])

        for name in sorted(package_loggers):
            logging.getLogger(name).addHandler(log_handler)

        # Display-name views of each chooser, bound to the settings parameter that
        # persists the selection. These are what the GUI binds combo boxes to.
        self.enhancer_parameter = PluginChooserParameter(enhancer_chooser, self.settings.algorithm)
        self.file_reader_parameter = PluginChooserParameter(
            file_reader_chooser, self.settings.file_type
        )

        # Deliberately unbound: the writer shares file_type with the reader above, so
        # binding both would make them fight over the same parameter.
        file_writer_chooser.set_current_plugin(self.settings.file_type.get_value())

        self.repository = FluorescenceRepository(product_repository)
        self.fluorescence_api = FluorescenceAPI(
            task_manager,
            self.settings,
            product_api,
            self.repository,
            enhancer_chooser,
            self.enhancer_parameter_store,
            self.task_monitor,
            file_reader_chooser,
            file_writer_chooser,
        )

        self._reinit_observable = reinit_observable
        reinit_observable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinit_observable:
            # Runs after ProductCore has inserted the settings-driven product, so the
            # measured dataset binds to it. The default file path is a placeholder, so
            # stay quiet unless it names a real file.
            file_path = self.settings.file_path.get_value()

            if file_path.is_file() and len(self._product_repository) > 0:
                self.fluorescence_api.open_measured_dataset(
                    file_path,
                    len(self._product_repository) - 1,
                    file_type=self.settings.file_type.get_value(),
                )
