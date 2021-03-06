from __future__ import annotations
from pathlib import Path

from ..api.observer import Observable, Observer
from ..api.reconstructor import Reconstructor
from ..api.settings import SettingsGroup, SettingsRegistry
from ..api.plugins import PluginChooser, PluginEntry


class ReconstructorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.algorithm = settingsGroup.createStringEntry('Algorithm', 'rPIE')
        self.outputFileType = settingsGroup.createStringEntry('OutputFileType', 'NPZ')
        self.outputFilePath = settingsGroup.createPathEntry('OutputFilePath',
                                                            Path('ptychodus.npz'))

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ReconstructorSettings:
        settings = cls(settingsRegistry.createGroup('Reconstructor'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


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


class SelectableReconstructor(Reconstructor, Observable, Observer):  # TODO refactor

    @staticmethod
    def _createAlgorithmEntry(reconstructor: Reconstructor) -> PluginEntry[Reconstructor]:
        return PluginEntry[Reconstructor](simpleName=reconstructor.name,
                                          displayName=reconstructor.name,
                                          strategy=reconstructor)

    def __init__(self, settings: ReconstructorSettings,
                 reconstructorList: list[Reconstructor]) -> None:
        super().__init__()
        self._settings = settings
        self._algorithmChooser = PluginChooser[Reconstructor].createFromList([
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


class ReconstructorPlotPresenter(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._xlabel: str = 'Iteration'
        self._xvalues: list[float] = list()
        self._ylabel: str = 'Objective'
        self._yvalues: list[float] = list()

    @property
    def xlabel(self) -> str:
        return self._xlabel

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        if self._xlabel != value:
            self._xlabel = value
            self.notifyObservers()

    @property
    def ylabel(self) -> str:
        return self._ylabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        if self._ylabel != value:
            self._ylabel = value
            self.notifyObservers()

    @property
    def xvalues(self) -> list[float]:
        return self._xvalues

    @property
    def yvalues(self) -> list[float]:
        return self._yvalues

    def setValues(self, xvalues: list[float], yvalues: list[float]) -> None:
        self._xvalues = xvalues
        self._yvalues = yvalues
        self.notifyObservers()

    def setEnumeratedYValues(self, yvalues: list[float]) -> None:
        xvalues = [float(idx) for idx, _ in enumerate(yvalues)]
        self.setValues(xvalues, yvalues)
