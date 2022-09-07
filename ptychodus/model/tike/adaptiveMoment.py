from decimal import Decimal
from typing import Generic, TypeVar

from ...api.observer import Observable, Observer
from ...api.settings import SettingsGroup


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

    def getMinMDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxMDecay(self) -> Decimal:
        return Decimal(1)

    def getMDecay(self) -> Decimal:
        return self._clamp(self._settings.mdecay.value, self.getMinMDecay(), self.getMaxMDecay())

    def setMDecay(self, value: Decimal) -> None:
        self._settings.mdecay.value = value

    def getMinVDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxVDecay(self) -> Decimal:
        return Decimal(1)

    def getVDecay(self) -> Decimal:
        return self._clamp(self._settings.vdecay.value, self.getMinVDecay(), self.getMaxVDecay())

    def setVDecay(self, value: Decimal) -> None:
        self._settings.vdecay.value = value

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
