from collections.abc import Sequence

from ptychodus.api.reconstruct import ReconstructorLibrary, SubprocessReconstructor
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductAPI
from ..task_manager import TaskManager
from .api import ProcessingAPI, ProcessingAlgorithmParameter
from .devices import DeviceSelection
from .monitor import ProcessingTaskMonitor
from .parameter_store import ProcessingParameterStore
from .settings import ProcessingSettings


class ProcessingCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        algorithm_libraries: Sequence[ReconstructorLibrary],
    ) -> None:
        self._settings = ProcessingSettings(settings_registry)
        self.algorithm_parameter = ProcessingAlgorithmParameter(
            self._settings.algorithm, algorithm_libraries
        )
        self._task_monitor = ProcessingTaskMonitor(task_manager)
        self.parameter_store = ProcessingParameterStore(settings_registry)
        self.device_selection = DeviceSelection(self._settings.devices)

        # Every reconstructor is filed under the same key the algorithm selection
        # uses, so a panel and the options handed to a run cannot disagree.
        for library in algorithm_libraries:
            for reconstructor in library:
                self.parameter_store.register(
                    ProcessingAlgorithmParameter.generate_key(library.name, reconstructor.name),
                    reconstructor,
                )

                # Placement is ptychodus's to decide, so it is attached here
                # rather than passed through every backend's factory.
                if isinstance(reconstructor, SubprocessReconstructor):
                    reconstructor.device_environment = self.device_selection.environment

        self.processing_api = ProcessingAPI(
            task_manager,
            product_api,
            self.algorithm_parameter,
            self._task_monitor,
            self.parameter_store,
        )

        for library in algorithm_libraries:
            library.get_logger().addHandler(self._task_monitor.get_log_handler())
