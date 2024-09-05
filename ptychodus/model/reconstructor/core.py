from collections.abc import Sequence
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ptychodus.api.settings import SettingsRegistry

from ..patterns import ActiveDiffractionDataset
from ..product import ProductRepository
from .api import ReconstructorAPI
from .matcher import DiffractionPatternPositionMatcher
from .presenter import ReconstructorPresenter
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 diffractionDataset: ActiveDiffractionDataset,
                 productRepository: ProductRepository,
                 librarySeq: Sequence[ReconstructorLibrary]) -> None:
        self.settings = ReconstructorSettings(settingsRegistry)
        self._pluginChooser = PluginChooser[Reconstructor]()

        for library in librarySeq:
            for reconstructor in library:
                self._pluginChooser.registerPlugin(
                    reconstructor,
                    displayName=f'{library.name}/{reconstructor.name}',
                )

        if not self._pluginChooser:
            self._pluginChooser.registerPlugin(NullReconstructor('None'), displayName='None/None')

        self.dataMatcher = DiffractionPatternPositionMatcher(diffractionDataset, productRepository)
        self.reconstructorAPI = ReconstructorAPI(self.dataMatcher, productRepository,
                                                 self._pluginChooser)
        self.presenter = ReconstructorPresenter(self.settings, self._pluginChooser,
                                                self.reconstructorAPI, settingsRegistry)
