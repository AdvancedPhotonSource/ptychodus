from collections.abc import Sequence
import logging

from ...api.plugins import PluginChooser
from ...api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from ..patterns import ActiveDiffractionDataset
from ..product import ProductRepository
from .presenter import ReconstructorPresenter
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 diffractionDataset: ActiveDiffractionDataset,
                 productRepository: ProductRepository,
                 librarySeq: Sequence[ReconstructorLibrary]) -> None:
        self.settings = ReconstructorSettings.createInstance(settingsRegistry)
        self._pluginChooser = PluginChooser[Reconstructor]()

        for library in librarySeq:
            for reconstructor in library:
                self._pluginChooser.registerPlugin(
                    reconstructor,
                    displayName=f'{library.name}/{reconstructor.name}',
                )

        if not self._pluginChooser:
            self._pluginChooser.registerPlugin(NullReconstructor('None'), displayName='None/None')

        self.presenter = ReconstructorPresenter.createInstance(self.settings, diffractionDataset,
                                                               productRepository,
                                                               self._pluginChooser,
                                                               settingsRegistry)
