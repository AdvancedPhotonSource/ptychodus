from __future__ import annotations
from typing import Final

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsGroup, SettingsRegistry


class TikeMultigridSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useMultigrid = settingsGroup.createBooleanEntry('UseMultigrid', False)
        self.numLevels = settingsGroup.createIntegerEntry('NumLevels', 3)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeMultigridSettings:
        settings = cls(settingsRegistry.createGroup('TikeMultigrid'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeMultigridPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: TikeMultigridSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: TikeMultigridSettings) -> TikeMultigridPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def isMultigridEnabled(self) -> bool:
        return self._settings.useMultigrid.value

    def setMultigridEnabled(self, enabled: bool) -> None:
        self._settings.useMultigrid.value = enabled

    def getNumLevelsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumLevels(self) -> int:
        limits = self.getNumLevelsLimits()
        return limits.clamp(self._settings.numLevels.value)

    def setNumLevels(self, value: int) -> None:
        self._settings.numLevels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
