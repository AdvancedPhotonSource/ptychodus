from collections.abc import Sequence
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductAPI
from .api import ReconstructorAPI
from .log import ReconstructorLogHandler
from .matcher import DiffractionPatternPositionMatcher
from .presenter import ReconstructorPresenter
from .queue import ReconstructionQueue
from .settings import ReconstructorSettings


class ReconstructorCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        dataset: AssembledDiffractionDataset,
        product_api: ProductAPI,
        library_seq: Sequence[ReconstructorLibrary],
    ) -> None:
        self.settings = ReconstructorSettings(settings_registry)
        self._plugin_chooser = PluginChooser[Reconstructor]()
        self._log_handler = ReconstructorLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )

        for library in library_seq:
            for reconstructor in library:
                self._plugin_chooser.register_plugin(
                    reconstructor,
                    simple_name=f'{library.name}_{reconstructor.name}',
                    display_name=f'{library.name}/{reconstructor.name}',
                )

            library_logger = library.get_logger()
            library_logger.addHandler(self._log_handler)

        if not self._plugin_chooser:
            self._plugin_chooser.register_plugin(
                NullReconstructor('None'), display_name='None/None'
            )

        self.data_matcher = DiffractionPatternPositionMatcher(dataset, product_api)
        self._reconstruction_queue = ReconstructionQueue(self.data_matcher)
        self.reconstructor_api = ReconstructorAPI(
            self._reconstruction_queue, self.data_matcher, product_api, self._plugin_chooser
        )
        self.presenter = ReconstructorPresenter(
            self.settings,
            self._plugin_chooser,
            self._log_handler,
            self.reconstructor_api,
        )

    def start(self) -> None:
        self._reconstruction_queue.start()

    def stop(self) -> None:
        self._reconstruction_queue.stop()
