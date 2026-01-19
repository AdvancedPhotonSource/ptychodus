import logging
import threading

from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from .authorizer import GlobusAuthorizer
from .executor import GlobusExecutor
from .settings import GlobusSettings
from .status import GlobusStatusRepository

logger = logging.getLogger(__name__)


class GlobusCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
    ) -> None:
        self.settings = GlobusSettings(settings_registry)
        self.authorizer = GlobusAuthorizer()
        self.status_repository = GlobusStatusRepository()
        self.executor = GlobusExecutor(
            self.settings,
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
                self.authorizer, self.status_repository, self.executor
            )

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
        self.authorizer.shutdown_event.set()

        if self._thread:
            self._thread.join()

        logger.info('Globus thread stopped.')
