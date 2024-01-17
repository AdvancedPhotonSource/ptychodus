from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ...api.observer import Observable, Observer
from ...api.probe import Probe

logger = logging.getLogger(__name__)


class ProbeBuilder(ABC, Observable):

    @abstractmethod
    def build(self) -> Probe:
        pass


class ProbeRepositoryItem(Observable, Observer):

    def __init__(self, probe: Probe) -> None:
        super().__init__()
        self._probe = probe  # FIXME handle pixel size = 0
        self._builder: ProbeBuilder | None = None

    def getProbe(self) -> Probe:
        return self._probe

    def setProbe(self, probe: Probe) -> None:
        self._probe = probe
        self._builder = None
        self.notifyObservers()

    def rebuild(self) -> None:
        if self._builder is None:
            logger.error('Missing probe builder!')
            return

        try:
            probe = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize probe!')
        else:
            self._probe = probe
            self.notifyObservers()

    def getBuilder(self) -> ProbeBuilder | None:
        return self._builder

    def setBuilder(self, builder: ProbeBuilder) -> None:
        if self._builder is not None:
            self._builder.removeObserver(self)

        self._builder = builder
        builder.addObserver(self)
        self.rebuild()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self.rebuild()


# TODO class ProbeRepositoryItem(Observable, Observer):
# TODO     '''container for items that can be stored in a probe repository'''
# TODO     SIMPLE_NAME: Final[str] = 'FromMemory'
# TODO     DISPLAY_NAME: Final[str] = 'From Memory'
# TODO     MAX_INT: Final[int] = 0x7FFFFFFF
# TODO
# TODO     def __init__(self, modesFactory: MultimodalProbeFactory, nameHint: str) -> None:
# TODO         super().__init__()
# TODO         self._modesFactory = modesFactory
# TODO         self._nameHint = nameHint
# TODO         self._probe = Probe()
# TODO         self._initializer: ProbeBuilder | None = None
# TODO         self._numberOfModes = 0
# TODO         self._orthogonalizeModesEnabled = False
# TODO         self._modeDecayType = ProbeModeDecayType.POLYNOMIAL
# TODO         self._modeDecayRatio = Decimal(1)
# TODO
# TODO     @property
# TODO     def nameHint(self) -> str:
# TODO         '''returns a name hint that is appropriate for a settings file'''
# TODO         return self._nameHint
# TODO
# TODO     def getProbe(self) -> Probe:
# TODO         return self._probe
# TODO
# TODO     def setProbe(self, probe_: Probe) -> None:
# TODO         self._initializer = None
# TODO         self._probe = probe_
# TODO         self.notifyObservers()
# TODO
# TODO     def reinitialize(self) -> None:
# TODO         if self._initializer is None:
# TODO             logger.error('Missing probe initializer!')
# TODO             return
# TODO
# TODO         try:
# TODO             probe = self._initializer()
# TODO         except Exception:
# TODO             logger.exception('Failed to reinitialize probe!')
# TODO             return
# TODO
# TODO         if self._numberOfModes > 0:
# TODO             probe = self._modesFactory.build(probe, self._numberOfModes,
# TODO                                              self._orthogonalizeModesEnabled, self._modeDecayType,
# TODO                                              self._modeDecayRatio)
# TODO
# TODO         self._probe = probe
# TODO         self.notifyObservers()
# TODO
# TODO     def getBuilderSimpleName(self) -> str:
# TODO         return self.SIMPLE_NAME if self._initializer is None else self._initializer.simpleName
# TODO
# TODO     def getBuilderDisplayName(self) -> str:
# TODO         return self.DISPLAY_NAME if self._initializer is None else self._initializer.displayName
# TODO
# TODO     def getBuilder(self) -> ProbeBuilder | None:
# TODO         return self._initializer
# TODO
# TODO     def setBuilder(self, initializer: ProbeBuilder) -> None:
# TODO         if self._initializer is not None:
# TODO             self._initializer.removeObserver(self)
# TODO
# TODO         self._initializer = initializer
# TODO         initializer.addObserver(self)
# TODO         self.reinitialize()
# TODO
# TODO     def syncFromSettings(self, settings: ProbeSettings) -> None:
# TODO         '''synchronizes item state from settings'''
# TODO         self._numberOfModes = settings.numberOfModes.value
# TODO         self._orthogonalizeModesEnabled = settings.orthogonalizeModesEnabled.value
# TODO
# TODO         try:
# TODO             self._modeDecayType = ProbeModeDecayType[settings.modeDecayType.value.upper()]
# TODO         except KeyError:
# TODO             self._modeDecayType = ProbeModeDecayType.POLYNOMIAL
# TODO
# TODO         self._modeDecayRatio = settings.modeDecayRatio.value
# TODO         self.reinitialize()
# TODO
# TODO     def syncToSettings(self, settings: ProbeSettings) -> None:
# TODO         '''synchronizes item state to settings'''
# TODO         settings.numberOfModes.value = self._numberOfModes
# TODO         settings.orthogonalizeModesEnabled.value = self._orthogonalizeModesEnabled
# TODO         settings.modeDecayType.value = self._modeDecayType.name
# TODO         settings.modeDecayRatio.value = self._modeDecayRatio
# TODO
# TODO     def getNumberOfModesLimits(self) -> Interval[int]:
# TODO         return Interval[int](1, self.MAX_INT)
# TODO
# TODO     def getNumberOfModes(self) -> int:
# TODO         return self._probe.getNumberOfModes()
# TODO
# TODO     def setNumberOfModes(self, number: int) -> None:
# TODO         if self._numberOfModes != number:
# TODO             self._numberOfModes = number
# TODO             self.reinitialize()
# TODO
# TODO     @property
# TODO     def isOrthogonalizeModesEnabled(self) -> bool:
# TODO         return self._orthogonalizeModesEnabled
# TODO
# TODO     def setOrthogonalizeModesEnabled(self, value: bool) -> None:
# TODO         if self._orthogonalizeModesEnabled != value:
# TODO             self._orthogonalizeModesEnabled = value
# TODO             self.reinitialize()
# TODO
# TODO     def getModeDecayType(self) -> ProbeModeDecayType:
# TODO         return self._modeDecayType
# TODO
# TODO     def setModeDecayType(self, value: ProbeModeDecayType) -> None:
# TODO         if self._modeDecayType != value:
# TODO             self._modeDecayType = value
# TODO             self.reinitialize()
# TODO
# TODO     def getModeDecayRatioLimits(self) -> Interval[Decimal]:
# TODO         return Interval[Decimal](Decimal(0), Decimal(1))
# TODO
# TODO     def getModeDecayRatio(self) -> Decimal:
# TODO         limits = self.getModeDecayRatioLimits()
# TODO         return limits.clamp(self._modeDecayRatio)
# TODO
# TODO     def setModeDecayRatio(self, value: Decimal) -> None:
# TODO         if self._modeDecayRatio != value:
# TODO             self._modeDecayRatio = value
# TODO             self.reinitialize()
# TODO
# TODO     def update(self, observable: Observable) -> None:
# TODO         if observable is self._modesFactory:
# TODO             self.reinitialize()
# TODO         elif observable is self._initializer:
# TODO             self.reinitialize()
# TODO
# TODO
# TODO ProbeRepository = ItemRepository[ProbeRepositoryItem]
