import logging

from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from ..processing import ProcessingAPI
from ..task_manager import TaskManager
from .authorizer import GlobusAuthorizer
from .client import FakeGlobusClient, GlobusClient
from .executor import GlobusExecutor
from .settings import GlobusSettings
from .status import GlobusStatusRepository

logger = logging.getLogger(__name__)


class GlobusCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
    ) -> None:
        self.settings = GlobusSettings(settings_registry)
        self.authorizer = GlobusAuthorizer()
        self.status_repository = GlobusStatusRepository()

        try:
            from ._globus_client import RealGlobusClient
        except ModuleNotFoundError:
            logger.info('Globus not found.')
            self._client: GlobusClient = FakeGlobusClient()
        else:
            self._client = RealGlobusClient(
                task_manager, self.settings, self.authorizer, self.status_repository
            )

        self.executor = GlobusExecutor(
            self.settings,
            settings_registry,
            diffraction_api,
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
