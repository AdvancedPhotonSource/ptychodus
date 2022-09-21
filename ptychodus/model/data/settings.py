from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class DataSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.flipX = settingsGroup.createBooleanEntry('FlipX', False)
        self.flipY = settingsGroup.createBooleanEntry('FlipY', False)
        self.threshold = settingsGroup.createIntegerEntry('Threshold', 0)
        self.arrayDataType = settingsGroup.createStringEntry('ArrayDataType', 'uint16')
        self.scratchDirectory = settingsGroup.createPathEntry('ScratchDirectory',
                                                              Path('/dev/null'))
        self.numberOfDataThreads = settingsGroup.createIntegerEntry('NumberOfDataThreads', 8)
        self.watchForFiles = settingsGroup.createBooleanEntry('WatchForFiles', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DataSettings:
        settings = cls(settingsRegistry.createGroup('Data'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
