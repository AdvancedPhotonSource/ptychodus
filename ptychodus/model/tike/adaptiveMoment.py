from decimal import Decimal
from typing import Generic, TypeVar

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsGroup


class TikeAdaptiveMomentSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useAdaptiveMoment = settingsGroup.createBooleanEntry('UseAdaptiveMoment', False)
        self.mdecay = settingsGroup.createRealEntry('MDecay', '0.9')
        self.vdecay = settingsGroup.createRealEntry('VDecay', '0.999')
        settingsGroup.addObserver(self)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


T = TypeVar('T', bound=TikeAdaptiveMomentSettings)


class TikeAdaptiveMomentPresenter(Generic[T], Observable, Observer):

    def __init__(self, settings: T) -> None:
        super().__init__()
        self._settings = settings
        settings.addObserver(self)

    def isAdaptiveMomentEnabled(self) -> bool:
        return self._settings.useAdaptiveMoment.value

    def setAdaptiveMomentEnabled(self, enabled: bool) -> None:
        self._settings.useAdaptiveMoment.value = enabled

    def getMDecayLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMDecay(self) -> Decimal:
        limits = self.getMDecayLimits()
        return limits.clamp(self._settings.mdecay.value)

    def setMDecay(self, value: Decimal) -> None:
        self._settings.mdecay.value = value

    def getVDecayLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getVDecay(self) -> Decimal:
        limits = self.getVDecayLimits()
        return limits.clamp(self._settings.vdecay.value)

    def setVDecay(self, value: Decimal) -> None:
        self._settings.vdecay.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
