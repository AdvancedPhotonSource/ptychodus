from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class PtychoNNSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.modelInputSize = settingsGroup.createIntegerEntry('ModelInputSize', 128)
        self.modelOutputSize = settingsGroup.createIntegerEntry('ModelOutputSize', 128)
        self.modelStateFilePath = settingsGroup.createPathEntry('ModelStateFilePath',
                                                                Path('/dev/null'))
        self.batchSize = settingsGroup.createIntegerEntry('BatchSize', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNSettings:
        settings = cls(settingsRegistry.createGroup('PtychoNN'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
