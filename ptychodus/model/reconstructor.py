from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

import numpy

from .observer import Observable, Observer
from .settings import SettingsGroup, SettingsRegistry


class ReconstructorSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.algorithm = settingsGroup.createStringEntry('Algorithm', 'rPIE')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ReconstructorSettings:
        settings = cls(settingsRegistry.createGroup('Reconstructor'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Reconstructor(Observable, ABC):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def backendName(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self) -> int:
        pass


class NullReconstructor(Reconstructor):
    def __init__(self, name: str = 'None', backendName: str = 'Backend') -> None:
        super().__init__()
        self._name = name
        self._backendName = backendName

    @property
    def name(self) -> str:
        return self._name

    @property
    def backendName(self) -> str:
        return self._backendName

    def reconstruct(self) -> int:
        return 0


class SelectableReconstructor(Reconstructor, Observer):
    def __init__(self, settings: ReconstructorSettings,
                 reconstructorList: list[Reconstructor]) -> None:
        super().__init__()
        self._settings = settings
        self._reconstructorList = reconstructorList
        self._reconstructor = reconstructorList[0]

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       reconstructorList: list[Reconstructor]) -> SelectableReconstructor:
        if not reconstructorList:
            reconstructorList.append(NullReconstructor())

        selectableReconstructor = cls(settings, reconstructorList)
        selectableReconstructor.setCurrentAlgorithmFromSettings()
        settings.algorithm.addObserver(selectableReconstructor)
        return selectableReconstructor

    @property
    def name(self) -> str:
        return self._reconstructor.name

    @property
    def backendName(self) -> str:
        return self._reconstructor.backendName

    def reconstruct(self) -> int:
        return self._reconstructor.reconstruct()

    def getAlgorithmDict(self) -> dict[str, str]:
        return {
            reconstructor.name: reconstructor.backendName
            for reconstructor in self._reconstructorList
        }

    def getCurrentAlgorithm(self) -> str:
        return self.name

    def setCurrentAlgorithm(self, name: str) -> None:
        try:
            reconstructor = next(recon for recon in self._reconstructorList
                                 if name.casefold() == recon.name.casefold())
        except StopIteration:
            return

        if reconstructor is not self._reconstructor:
            self._reconstructor.removeObserver(self)
            self._reconstructor = reconstructor
            self._settings.algorithm.value = self._reconstructor.name
            self._reconstructor.addObserver(self)
            self.notifyObservers()

    def setCurrentAlgorithmFromSettings(self) -> None:
        self.setCurrentAlgorithm(self._settings.algorithm.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.algorithm:
            self.setCurrentAlgorithmFromSettings()


class ReconstructorPresenter(Observable, Observer):
    def __init__(self, settings: ReconstructorSettings,
                 selectableReconstructor: SelectableReconstructor) -> None:
        super().__init__()
        self._settings = settings
        self._selectableReconstructor = selectableReconstructor

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       selectableReconstructor: SelectableReconstructor) -> ReconstructorPresenter:
        presenter = cls(settings, selectableReconstructor)
        selectableReconstructor.addObserver(presenter)
        return presenter

    def getAlgorithmDict(self) -> dict[str, str]:
        return self._selectableReconstructor.getAlgorithmDict()

    def getCurrentAlgorithm(self) -> str:
        return self._selectableReconstructor.getCurrentAlgorithm()

    def setCurrentAlgorithm(self, name: str) -> None:
        self._selectableReconstructor.setCurrentAlgorithm(name)

    def reconstruct(self) -> int:
        return self._selectableReconstructor.reconstruct()

    def update(self, observable: Observable) -> None:
        if observable is self._selectableReconstructor:
            self.notifyObservers()
