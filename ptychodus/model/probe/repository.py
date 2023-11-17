from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Final
import logging

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.probe import Probe
from ..itemRepository import ItemRepository
from .modes import MultimodalProbeFactory, ProbeModeDecayType
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeInitializer(ABC, Observable):
    '''ABC for plugins that can initialize probes'''

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def syncFromSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state to settings'''
        pass

    @abstractmethod
    def __call__(self) -> Probe:
        '''produces an initial probe guess'''
        pass


class ProbeRepositoryItem(Observable, Observer):
    '''container for items that can be stored in a probe repository'''
    SIMPLE_NAME: Final[str] = 'FromMemory'
    DISPLAY_NAME: Final[str] = 'From Memory'
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, modesFactory: MultimodalProbeFactory, nameHint: str) -> None:
        super().__init__()
        self._modesFactory = modesFactory
        self._nameHint = nameHint
        self._probe = Probe()
        self._initializer: ProbeInitializer | None = None
        self._numberOfModes = 0
        self._orthogonalizeModesEnabled = False
        self._modeDecayType = ProbeModeDecayType.POLYNOMIAL
        self._modeDecayRatio = Decimal(1)

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    def getProbe(self) -> Probe:
        return self._probe

    def setProbe(self, probe_: Probe) -> None:
        self._initializer = None
        self._probe = probe_
        self.notifyObservers()

    def reinitialize(self) -> None:
        if self._initializer is None:
            logger.error('Missing probe initializer!')
            return

        try:
            probe = self._initializer()
        except Exception:
            logger.exception('Failed to reinitialize probe!')
            return

        if self._numberOfModes > 0:
            probe = self._modesFactory.build(probe, self._numberOfModes,
                                             self._orthogonalizeModesEnabled, self._modeDecayType,
                                             self._modeDecayRatio)

        self._probe = probe
        self.notifyObservers()

    def getInitializerSimpleName(self) -> str:
        return self.SIMPLE_NAME if self._initializer is None else self._initializer.simpleName

    def getInitializerDisplayName(self) -> str:
        return self.DISPLAY_NAME if self._initializer is None else self._initializer.displayName

    def getInitializer(self) -> ProbeInitializer | None:
        return self._initializer

    def setInitializer(self, initializer: ProbeInitializer) -> None:
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state from settings'''
        self._numberOfModes = settings.numberOfModes.value
        self._orthogonalizeModesEnabled = settings.orthogonalizeModesEnabled.value

        try:
            self._modeDecayType = ProbeModeDecayType[settings.modeDecayType.value.upper()]
        except KeyError:
            self._modeDecayType = ProbeModeDecayType.POLYNOMIAL

        self._modeDecayRatio = settings.modeDecayRatio.value
        self.reinitialize()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state to settings'''
        settings.numberOfModes.value = self._numberOfModes
        settings.orthogonalizeModesEnabled.value = self._orthogonalizeModesEnabled
        settings.modeDecayType.value = self._modeDecayType.name
        settings.modeDecayRatio.value = self._modeDecayRatio

    def getNumberOfModesLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfModes(self) -> int:
        return self._probe.getNumberOfModes()

    def setNumberOfModes(self, number: int) -> None:
        if self._numberOfModes != number:
            self._numberOfModes = number
            self.reinitialize()

    @property
    def isOrthogonalizeModesEnabled(self) -> bool:
        return self._orthogonalizeModesEnabled

    def setOrthogonalizeModesEnabled(self, value: bool) -> None:
        if self._orthogonalizeModesEnabled != value:
            self._orthogonalizeModesEnabled = value
            self.reinitialize()

    def getModeDecayType(self) -> ProbeModeDecayType:
        return self._modeDecayType

    def setModeDecayType(self, value: ProbeModeDecayType) -> None:
        if self._modeDecayType != value:
            self._modeDecayType = value
            self.reinitialize()

    def getModeDecayRatioLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getModeDecayRatio(self) -> Decimal:
        limits = self.getModeDecayRatioLimits()
        return limits.clamp(self._modeDecayRatio)

    def setModeDecayRatio(self, value: Decimal) -> None:
        if self._modeDecayRatio != value:
            self._modeDecayRatio = value
            self.reinitialize()

    def update(self, observable: Observable) -> None:
        if observable is self._modesFactory:
            self.reinitialize()
        elif observable is self._initializer:
            self.reinitialize()


ProbeRepository = ItemRepository[ProbeRepositoryItem]
