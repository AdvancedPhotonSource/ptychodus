from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from .chooser import StrategyChooser, StrategyEntry
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
        raise NotImplementedError
        return 0


class SelectableReconstructor(Reconstructor, Observer):
    @staticmethod
    def _createAlgorithmEntry(reconstructor: Reconstructor) -> StrategyEntry[Reconstructor]:
        return StrategyEntry[Reconstructor](simpleName=reconstructor.name,
                                            displayName=reconstructor.name,
                                            strategy=reconstructor)

    def __init__(self, settings: ReconstructorSettings,
                 reconstructorList: list[Reconstructor]) -> None:
        super().__init__()
        self._settings = settings
        self._algorithmChooser = StrategyChooser[Reconstructor].createFromList([
            SelectableReconstructor._createAlgorithmEntry(reconstructor)
            for reconstructor in reconstructorList
        ])

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       reconstructorList: list[Reconstructor]) -> SelectableReconstructor:
        if not reconstructorList:
            reconstructorList.append(NullReconstructor())

        reconstructor = cls(settings, reconstructorList)
        settings.algorithm.addObserver(reconstructor)
        reconstructor._algorithmChooser.addObserver(reconstructor)
        reconstructor._syncAlgorithmFromSettings()
        return reconstructor

    @property
    def name(self) -> str:
        return self._algorithmChooser.getCurrentDisplayName()

    @property
    def backendName(self) -> str:
        algorithm = self._algorithmChooser.getCurrentStrategy()
        return algorithm.backendName

    def reconstruct(self) -> int:
        algorithm = self._algorithmChooser.getCurrentStrategy()
        return algorithm.reconstruct()

    def getAlgorithmDict(self) -> dict[str, str]:
        return {entry.displayName: entry.strategy.backendName for entry in self._algorithmChooser}

    def getAlgorithm(self) -> str:
        return self._algorithmChooser.getCurrentDisplayName()

    def setAlgorithm(self, name: str) -> None:
        self._algorithmChooser.setFromDisplayName(name)

    def _syncAlgorithmFromSettings(self) -> None:
        self._algorithmChooser.setFromSimpleName(self._settings.algorithm.value)

    def _syncAlgorithmToSettings(self) -> None:
        self._settings.algorithm.value = self._algorithmChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.algorithm:
            self._syncAlgorithmFromSettings()
        elif observable is self._algorithmChooser:
            self._syncAlgorithmToSettings()


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

    def getAlgorithm(self) -> str:
        return self._selectableReconstructor.getAlgorithm()

    def setAlgorithm(self, name: str) -> None:
        self._selectableReconstructor.setAlgorithm(name)

    def reconstruct(self) -> int:
        return self._selectableReconstructor.reconstruct()

    def update(self, observable: Observable) -> None:
        if observable is self._selectableReconstructor:
            self.notifyObservers()
