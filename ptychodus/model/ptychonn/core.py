from __future__ import annotations
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import ActiveDiffractionDataset
from ..object import Object
from ..probe import Apparatus
from ..reconstructor import Reconstructor, NullReconstructor
from ..scan import Scan
from .settings import PtychoNNSettings

logger = logging.getLogger(__name__)


class PtychoNNBackend:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = PtychoNNSettings.createInstance(settingsRegistry)
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls,
                       settingsRegistry: SettingsRegistry,
                       diffractionDataset: ActiveDiffractionDataset,
                       scan: Scan,
                       apparatus: Apparatus,
                       object_: Object,
                       isDeveloperModeEnabled: bool = False) -> PtychoNNBackend:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import PtychoNNReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('PtychoNN', 'PtychoNN'))
        else:
            core.reconstructorList.append(
                PtychoNNReconstructor(core._settings, apparatus, scan, object_,
                                      diffractionDataset))

        return core
