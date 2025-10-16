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
from ..product import ProductAPI
from .authorizer import GlobusAuthorizer
from .executor import GlobusExecutor
from .locator import DataLocator, OutputDataLocator, SimpleDataLocator
from .settings import GlobusSettings
from .status import GlobusStatus, GlobusStatusRepository

logger = logging.getLogger(__name__)


class GlobusParametersPresenter(Observable, Observer):
    def __init__(
        self,
        settings: GlobusSettings,
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


class GlobusAuthorizationPresenter:
    def __init__(self, authorizer: GlobusAuthorizer) -> None:
        self._authorizer = authorizer

    @property
    def is_authorized(self) -> bool:
        return self._authorizer.is_authorized

    def get_authorize_url(self) -> str:
        return self._authorizer.get_authorize_url()

    def set_code_from_authorize_url(self, code: str) -> None:
        self._authorizer.set_code_from_authorize_url(code)


class GlobusStatusPresenter(Observable, Observer):
    def __init__(self, settings: GlobusSettings, status_repository: GlobusStatusRepository) -> None:
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
    def __getitem__(self, index: int) -> GlobusStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GlobusStatus]: ...

    def __getitem__(self, index: int | slice) -> GlobusStatus | Sequence[GlobusStatus]:
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


class GlobusExecutionPresenter:
    def __init__(self, executor: GlobusExecutor) -> None:
        self._executor = executor

    def run_flow(self, input_product_index: int) -> None:
        self._executor.run_flow(input_product_index)


class GlobusCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
    ) -> None:
        self._settings = GlobusSettings(settings_registry)
        self._input_data_locator = SimpleDataLocator(self._settings._group, 'Input')
        self._compute_data_locator = SimpleDataLocator(self._settings._group, 'Compute')
        self._output_data_locator = OutputDataLocator(
            self._settings._group, 'Output', self._input_data_locator
        )
        self._authorizer = GlobusAuthorizer()
        self._status_repository = GlobusStatusRepository()
        self.executor = GlobusExecutor(
            self._settings,
            self._input_data_locator,
            self._compute_data_locator,
            self._output_data_locator,
            settings_registry,
            diffraction_api,
            product_api,
        )
        self._thread: threading.Thread | None = None

        try:
            from .globus import GlobusThread
        except ModuleNotFoundError:
            logger.info('Globus not found.')
        else:
            self._thread = GlobusThread.create_instance(
                self._authorizer, self._status_repository, self.executor
            )

        self.parameters_presenter = GlobusParametersPresenter(
            self._settings,
            self._input_data_locator,
            self._compute_data_locator,
            self._output_data_locator,
        )
        self.authorization_presenter = GlobusAuthorizationPresenter(self._authorizer)
        self.status_presenter = GlobusStatusPresenter(self._settings, self._status_repository)
        self.execution_presenter = GlobusExecutionPresenter(self.executor)

    @property
    def is_supported(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        logger.info('Starting Globus thread...')

        if self._thread:
            self._thread.start()

        logger.info('Globus thread started.')

    def stop(self) -> None:
        logger.info('Stopping Globus thread...')
        self.executor.job_queue.join()
        self._authorizer.shutdown_event.set()

        if self._thread:
            self._thread.join()

        logger.info('Globus thread stopped.')
