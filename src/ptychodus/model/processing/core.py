from collections.abc import Sequence
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .api import ProcessingAPI
from .context import ProcessingContext
from .settings import ProcessingSettings

logger = logging.getLogger(__name__)


class ProcessingCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        library_seq: Sequence[ReconstructorLibrary],
    ) -> None:
        self.settings = ProcessingSettings(settings_registry)
        self._plugin_chooser = PluginChooser[Reconstructor]()
        self._context = ProcessingContext(task_manager)

        for library in library_seq:
            for reconstructor in library:
                reconstructor_name = reconstructor.get_name()
                self._plugin_chooser.register_plugin(
                    reconstructor,
                    simple_name=f'{library.name}_{reconstructor_name}',
                    display_name=f'{library.name}/{reconstructor_name}',
                )

            library_logger = library.get_logger()
            library_logger.addHandler(self._context.get_log_handler())

        if not self._plugin_chooser:
            self._plugin_chooser.register_plugin(
                NullReconstructor('None'), display_name='None/None'
            )

        self.processing_api = ProcessingAPI(
            task_manager,
            diffraction_api,
            product_api,
            self.settings,
            self._context,
            self._plugin_chooser,
        )
