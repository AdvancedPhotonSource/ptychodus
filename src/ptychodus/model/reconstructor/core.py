from collections.abc import Sequence
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ..patterns import AssembledDiffractionDataset
from ..product import ProductRepository
from .api import ReconstructorAPI
from .log import ReconstructorLogHandler
from .matcher import DiffractionPatternPositionMatcher
from .presenter import ReconstructorPresenter
from .queue import ReconstructionQueue
from .settings import ReconstructorSettings


class ReconstructorCore:
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        diffractionDataset: AssembledDiffractionDataset,
        productRepository: ProductRepository,
        librarySeq: Sequence[ReconstructorLibrary],
    ) -> None:
        self.settings = ReconstructorSettings(settingsRegistry)
        self._pluginChooser = PluginChooser[Reconstructor]()
        self._logHandler = ReconstructorLogHandler()
        self._logHandler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )

        for library in librarySeq:
            for reconstructor in library:
                self._pluginChooser.registerPlugin(
                    reconstructor,
                    simpleName=f'{library.name}_{reconstructor.name}',
                    displayName=f'{library.name}/{reconstructor.name}',
                )

            libraryLogger = logging.getLogger(library.logger_name)
            libraryLogger.addHandler(self._logHandler)

        if not self._pluginChooser:
            self._pluginChooser.registerPlugin(NullReconstructor('None'), displayName='None/None')

        self._reconstructionQueue = ReconstructionQueue()
        self.dataMatcher = DiffractionPatternPositionMatcher(diffractionDataset, productRepository)
        self.reconstructorAPI = ReconstructorAPI(
            self._reconstructionQueue, self.dataMatcher, productRepository, self._pluginChooser
        )
        self.presenter = ReconstructorPresenter(
            self.settings,
            self._pluginChooser,
            self._logHandler,
            self.reconstructorAPI,
            settingsRegistry,
        )

    def start(self) -> None:
        self._reconstructionQueue.start()

    def stop(self) -> None:
        self._reconstructionQueue.stop()
