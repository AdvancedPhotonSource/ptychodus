from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import overload
from uuid import UUID
import logging
import threading

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from ..reconstructor import ReconstructorAPI
from .api import ConcreteWorkflowAPI
from .authorizer import WorkflowAuthorizer
from .executor import WorkflowExecutor
from .locator import DataLocator, OutputDataLocator, SimpleDataLocator
from .settings import WorkflowSettings
from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)


class WorkflowParametersPresenter(Observable, Observer):
    def __init__(
        self,
        settings: WorkflowSettings,
        input_data_locator: DataLocator,
        compute_data_locator: DataLocator,
        output_data_locator: OutputDataLocator,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._input_data_locator = input_data_locator
        self._compute_data_locator = compute_data_locator
        self._output_data_locator = output_data_locator

        settings.add_observer(self)
        input_data_locator.add_observer(self)
        compute_data_locator.add_observer(self)
        output_data_locator.add_observer(self)

    def set_input_data_endpoint_id(self, endpoint_id: UUID) -> None:
        self._input_data_locator.set_endpoint_id(endpoint_id)

    def get_input_data_endpoint_id(self) -> UUID:
        return self._input_data_locator.get_endpoint_id()

    def set_input_data_globus_path(self, globus_path: str) -> None:
        self._input_data_locator.set_globus_path(globus_path)

    def get_input_data_globus_path(self) -> str:
        return self._input_data_locator.get_globus_path()

    def set_input_data_posix_path(self, posix_path: Path) -> None:
        self._input_data_locator.set_posix_path(posix_path)

    def get_input_data_posix_path(self) -> Path:
        return self._input_data_locator.get_posix_path()

    def set_compute_endpoint_id(self, endpoint_id: UUID) -> None:
        self._settings.compute_endpoint_id.set_value(endpoint_id)

    def get_compute_endpoint_id(self) -> UUID:
        return self._settings.compute_endpoint_id.get_value()

    def set_compute_data_endpoint_id(self, endpoint_id: UUID) -> None:
        self._compute_data_locator.set_endpoint_id(endpoint_id)

    def get_compute_data_endpoint_id(self) -> UUID:
        return self._compute_data_locator.get_endpoint_id()

    def set_compute_data_globus_path(self, globus_path: str) -> None:
        self._compute_data_locator.set_globus_path(globus_path)

    def get_compute_data_globus_path(self) -> str:
        return self._compute_data_locator.get_globus_path()

    def set_compute_data_posix_path(self, posix_path: Path) -> None:
        self._compute_data_locator.set_posix_path(posix_path)

    def get_compute_data_posix_path(self) -> Path:
        return self._compute_data_locator.get_posix_path()

    def set_round_trip_enabled(self, enable: bool) -> None:
        self._output_data_locator.set_round_trip_enabled(enable)

    def is_round_trip_enabled(self) -> bool:
        return self._output_data_locator.is_round_trip_enabled()

    def set_output_data_endpoint_id(self, endpoint_id: UUID) -> None:
        self._output_data_locator.set_endpoint_id(endpoint_id)

    def get_output_data_endpoint_id(self) -> UUID:
        return self._output_data_locator.get_endpoint_id()

    def set_output_data_globus_path(self, globus_path: str) -> None:
        self._output_data_locator.set_globus_path(globus_path)

    def get_output_data_globus_path(self) -> str:
        return self._output_data_locator.get_globus_path()

    def set_output_data_posix_path(self, posix_path: Path) -> None:
        self._output_data_locator.set_posix_path(posix_path)

    def get_output_data_posix_path(self) -> Path:
        return self._output_data_locator.get_posix_path()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()
        elif observable in (
            self._input_data_locator,
            self._compute_data_locator,
            self._output_data_locator,
        ):
            self.notify_observers()


class WorkflowAuthorizationPresenter:
    def __init__(self, authorizer: WorkflowAuthorizer) -> None:
        self._authorizer = authorizer

    @property
    def is_authorized(self) -> bool:
        return self._authorizer.is_authorized

    def get_authorize_url(self) -> str:
        return self._authorizer.get_authorize_url()

    def set_code_from_authorize_url(self, code: str) -> None:
        self._authorizer.set_code_from_authorize_url(code)


class WorkflowStatusPresenter(Observable, Observer):
    def __init__(
        self, settings: WorkflowSettings, status_repository: WorkflowStatusRepository
    ) -> None:
        super().__init__()
        self._settings = settings
        self._status_repository = status_repository

        settings.add_observer(self)

    def get_refresh_interval_limits_s(self) -> Interval[int]:
        return Interval[int](10, 86400)

    def get_refresh_interval_s(self) -> int:
        limits = self.get_refresh_interval_limits_s()
        return limits.clamp(self._settings.status_refresh_interval_s.get_value())

    def set_refresh_interval_s(self, seconds: int) -> None:
        self._settings.status_refresh_interval_s.set_value(seconds)

    @overload
    def __getitem__(self, index: int) -> WorkflowStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WorkflowStatus]: ...

    def __getitem__(self, index: int | slice) -> WorkflowStatus | Sequence[WorkflowStatus]:
        return self._status_repository[index]

    def __len__(self) -> int:
        return len(self._status_repository)

    def get_status_date_time(self) -> datetime:
        return self._status_repository.get_status_date_time()

    def refresh_status(self) -> None:
        self._status_repository.refresh_status()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()


class WorkflowExecutionPresenter:
    def __init__(self, executor: WorkflowExecutor) -> None:
        self._executor = executor

    def run_flow(self, input_product_index: int) -> None:
        self._executor.run_flow(input_product_index)


class WorkflowCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        scan_api: ScanAPI,
        probe_api: ProbeAPI,
        object_api: ObjectAPI,
        reconstructor_api: ReconstructorAPI,
    ) -> None:
        self._settings = WorkflowSettings(settings_registry)
        self._input_data_locator = SimpleDataLocator(self._settings._group, 'Input')
        self._compute_data_locator = SimpleDataLocator(self._settings._group, 'Compute')
        self._output_data_locator = OutputDataLocator(
            self._settings._group, 'Output', self._input_data_locator
        )
        self._authorizer = WorkflowAuthorizer()
        self._status_repository = WorkflowStatusRepository()
        self._executor = WorkflowExecutor(
            self._settings,
            self._input_data_locator,
            self._compute_data_locator,
            self._output_data_locator,
            settings_registry,
            diffraction_api,
            product_api,
        )
        self.workflow_api = ConcreteWorkflowAPI(
            settings_registry,
            diffraction_api,
            product_api,
            scan_api,
            probe_api,
            object_api,
            reconstructor_api,
            self._executor,
        )
        self._thread: threading.Thread | None = None

        try:
            from .globus import GlobusWorkflowThread
        except ModuleNotFoundError:
            logger.info('Globus not found.')
        else:
            self._thread = GlobusWorkflowThread.create_instance(
                self._authorizer, self._status_repository, self._executor
            )

        self.parameters_presenter = WorkflowParametersPresenter(
            self._settings,
            self._input_data_locator,
            self._compute_data_locator,
            self._output_data_locator,
        )
        self.authorization_presenter = WorkflowAuthorizationPresenter(self._authorizer)
        self.status_presenter = WorkflowStatusPresenter(self._settings, self._status_repository)
        self.execution_presenter = WorkflowExecutionPresenter(self._executor)

    @property
    def is_supported(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        logger.info('Starting workflow thread...')

        if self._thread:
            self._thread.start()

        logger.info('Workflow thread started.')

    def stop(self) -> None:
        logger.info('Stopping workflow thread...')
        self._executor.job_queue.join()
        self._authorizer.shutdown_event.set()

        if self._thread:
            self._thread.join()

        logger.info('Workflow thread stopped.')
