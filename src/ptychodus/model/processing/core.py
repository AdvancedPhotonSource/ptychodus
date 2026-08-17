from collections.abc import Sequence

from ptychodus.api.reconstruct import ReconstructorLibrary
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductAPI
from ..task_manager import TaskManager
from .api import ProcessingAPI, ProcessingAlgorithmParameter
from .monitor import ProcessingTaskMonitor
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
        self.processing_api = ProcessingAPI(
            task_manager,
            product_api,
            self.algorithm_parameter,
            self._task_monitor,
        )

        for library in algorithm_libraries:
            library.get_logger().addHandler(self._task_monitor.get_log_handler())
