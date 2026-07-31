from importlib.metadata import version
import logging
import queue

from ptychodus.api.settings import SettingsRegistry

from ..processing import ProcessingAPI
from ..product import ProductAPI
from .authorizer import GlobusAuthorizer
from .client import FakeGlobusClient, GlobusClient, GlobusStatus
from .executor import GlobusExecutor
from .settings import GlobusSettings
from .status import GlobusStatusRepository

logger = logging.getLogger(__name__)


class GlobusCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
    ) -> None:
        status_q: queue.Queue[GlobusStatus] = queue.Queue()

        self.settings = GlobusSettings(settings_registry)
        self.authorizer = GlobusAuthorizer()

        try:
            from .globus import RealGlobusClient
        except ModuleNotFoundError:
            logger.info('Globus not found.')
            self._client: GlobusClient = FakeGlobusClient()
        else:
            logger.info('Globus SDK ' + version('globus-sdk'))
            self._client = RealGlobusClient(self.settings, self.authorizer, status_q)

        self.status_repository = GlobusStatusRepository(self.settings, self._client, status_q)
        self.executor = GlobusExecutor(
            self.settings,
            settings_registry,
            product_api,
            processing_api,
            self._client,
        )

    @property
    def is_supported(self) -> bool:
        return self._client.is_supported

    def start(self) -> None:
        self._client.start()

    def stop(self) -> None:
        self._client.stop()

    def run_foreground_tasks(self) -> None:
        self.authorizer.run_foreground_tasks()
        self.status_repository.run_foreground_tasks()
