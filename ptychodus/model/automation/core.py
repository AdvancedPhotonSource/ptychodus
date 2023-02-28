from __future__ import annotations
from pathlib import Path
import queue

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import DataCore
from ..object import ObjectCore
from ..probe import ProbeCore
from ..scan import ScanCore
from ..workflow import WorkflowCore
from .buffer import AutomationDatasetBuffer
from .processor import AutomationDatasetProcessor
from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings
from .watcher import DataDirectoryWatcher
from .workflow import AutomationDatasetWorkflow


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


class AutomationProcessingPresenter(Observable, Observer):

    def __init__(self, settings: AutomationSettings, repository: AutomationDatasetRepository,
                 processor: AutomationDatasetProcessor) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._processor = processor

    @classmethod
    def createInstance(cls, settings: AutomationSettings, repository: AutomationDatasetRepository,
                       processor: AutomationDatasetProcessor) -> AutomationProcessingPresenter:
        presenter = cls(settings, repository, processor)
        settings.addObserver(presenter)
        repository.addObserver(presenter)
        return presenter

    def getDatasetLabel(self, index: int) -> str:
        return self._repository.getLabel(index)

    def getDatasetState(self, index: int) -> AutomationDatasetState:
        return self._repository.getState(index)

    def getNumberOfDatasets(self) -> int:
        return len(self._repository)

    def isProcessingEnabled(self) -> bool:
        return self._processor.isAlive

    def setProcessingEnabled(self, enable: bool) -> None:
        if enable:
            self._processor.start()
        else:
            self._processor.stop()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._repository:
            self.notifyObservers()


class AutomationCore:

    def __init__(self, settingsRegistry: SettingsRegistry, dataCore: DataCore, scanCore: ScanCore,
                 probeCore: ProbeCore, objectCore: ObjectCore, workflowCore: WorkflowCore) -> None:
        self._settings = AutomationSettings.createInstance(settingsRegistry)
        self.repository = AutomationDatasetRepository(self._settings)
        self._processingQueue: queue.Queue[Path] = queue.Queue()
        self._buffer = AutomationDatasetBuffer(self._settings, self.repository,
                                               self._processingQueue)
        self._workflow = AutomationDatasetWorkflow(dataCore, scanCore, probeCore, objectCore,
                                                   workflowCore)
        self._processor = AutomationDatasetProcessor(self._settings, self.repository,
                                                     self._workflow, self._processingQueue)
        self._watcher = DataDirectoryWatcher.createInstance(self._settings, self._buffer)
        self.presenter = AutomationPresenter.createInstance(self._settings, self._watcher)
        self.processingPresenter = AutomationProcessingPresenter.createInstance(
            self._settings, self.repository, self._processor)

    def start(self) -> None:
        self._buffer.start()

    def stop(self) -> None:
        self._processor.stop()
        self._watcher.stop()
        self._buffer.stop()
