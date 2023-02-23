from __future__ import annotations
from pathlib import Path

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from .settings import AutomationSettings
from .watcher import DataDirectoryWatcher


class AutomationPresenter(Observable, Observer):

    def __init__(self, settings: AutomationSettings, watcher: DataDirectoryWatcher) -> None:
        super().__init__()
        self._settings = settings
        self._watcher = watcher

    @classmethod
    def createInstance(cls, settings: AutomationSettings,
                       watcher: DataDirectoryWatcher) -> AutomationPresenter:
        presenter = cls(settings, watcher)
        settings.addObserver(presenter)
        watcher.addObserver(presenter)
        return presenter

    def isWatchdogEnabled(self) -> bool:
        return self._watcher.isAlive

    def setWatchdogEnabled(self, enable: bool) -> None:
        if enable:
            self._watcher.start()
        else:
            self._watcher.stop()

    def getWatchdogDirectory(self) -> Path:
        return self._settings.watchdogDirectory.value

    def setWatchdogDirectory(self, directory: Path) -> None:
        self._settings.watchdogDirectory.value = directory

    def getStrategyList(self) -> list[str]:
        return ['CNM/APS Hard X-Ray Nanoprobe']

    def getStrategy(self) -> str:
        return self._settings.strategy.value

    def setStrategy(self, strategy: str) -> None:
        self._settings.strategy.value = strategy

    def getWatchdogDelayLimitsInSeconds(self) -> Interval[int]:
        return Interval[int](0, 600)

    def getWatchdogDelayInSeconds(self) -> int:
        limits = self.getWatchdogDelayLimitsInSeconds()
        return limits.clamp(self._settings.watchdogDelayInSeconds.value)

    def setWatchdogDelayInSeconds(self, value: int) -> None:
        self._settings.watchdogDelayInSeconds.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._watcher:
            self.notifyObservers()


class AutomationCore:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = AutomationSettings.createInstance(settingsRegistry)
        self._watcher = DataDirectoryWatcher.createInstance(self._settings)
        self.presenter = AutomationPresenter.createInstance(self._settings, self._watcher)
