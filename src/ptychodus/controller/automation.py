from __future__ import annotations
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QTimer
from PyQt5.QtGui import QFont

from ptychodus.api.observer import Observable, Observer

from ..model.automation import (
    AutomationCore,
    AutomationDatasetState,
    AutomationPresenter,
    AutomationProcessingPresenter,
)
from ..view.automation import (
    AutomationView,
    AutomationProcessingView,
    AutomationWatchdogView,
)
from .data import FileDialogFactory


class AutomationProcessingController(Observer):
    def __init__(
        self,
        presenter: AutomationPresenter,
        view: AutomationProcessingView,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(
        cls,
        presenter: AutomationPresenter,
        view: AutomationProcessingView,
        fileDialogFactory: FileDialogFactory,
    ) -> AutomationProcessingController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.add_observer(controller)

        for strategy in presenter.getStrategyList():
            view.strategyComboBox.addItem(strategy)

        view.strategyComboBox.textActivated.connect(presenter.setStrategy)
        view.directoryLineEdit.editingFinished.connect(controller._syncDirectoryToModel)
        view.directoryBrowseButton.clicked.connect(controller._browseDirectory)
        view.intervalSpinBox.valueChanged.connect(presenter.setProcessingIntervalInSeconds)

        controller._syncModelToView()

        return controller

    def _syncDirectoryToModel(self) -> None:
        dataDirectory = Path(self._view.directoryLineEdit.text())
        self._presenter.setDataDirectory(dataDirectory)

    def _browseDirectory(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(
            self._view, 'Choose Data Directory'
        )

        if dirPath:
            self._presenter.setDataDirectory(dirPath)

    def _syncModelToView(self) -> None:
        self._view.strategyComboBox.blockSignals(True)
        self._view.strategyComboBox.setCurrentText(self._presenter.getStrategy())
        self._view.strategyComboBox.blockSignals(False)

        dataDirectory = self._presenter.getDataDirectory()

        if dataDirectory:
            self._view.directoryLineEdit.setText(str(dataDirectory))
        else:
            self._view.directoryLineEdit.clear()

        intervalLimitsInSeconds = self._presenter.getProcessingIntervalLimitsInSeconds()

        self._view.intervalSpinBox.blockSignals(True)
        self._view.intervalSpinBox.setRange(
            intervalLimitsInSeconds.lower, intervalLimitsInSeconds.upper
        )
        self._view.intervalSpinBox.setValue(self._presenter.getProcessingIntervalInSeconds())
        self._view.intervalSpinBox.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationWatchdogController(Observer):
    def __init__(self, presenter: AutomationPresenter, view: AutomationWatchdogView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls, presenter: AutomationPresenter, view: AutomationWatchdogView
    ) -> AutomationWatchdogController:
        controller = cls(presenter, view)
        presenter.add_observer(controller)

        view.delaySpinBox.valueChanged.connect(presenter.setWatchdogDelayInSeconds)
        view.usePollingObserverCheckBox.toggled.connect(presenter.setWatchdogPollingObserverEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        delayLimitsInSeconds = self._presenter.getWatchdogDelayLimitsInSeconds()

        self._view.delaySpinBox.blockSignals(True)
        self._view.delaySpinBox.setRange(delayLimitsInSeconds.lower, delayLimitsInSeconds.upper)
        self._view.delaySpinBox.setValue(self._presenter.getWatchdogDelayInSeconds())
        self._view.delaySpinBox.blockSignals(False)

        self._view.usePollingObserverCheckBox.setChecked(
            self._presenter.isWatchdogPollingObserverEnabled()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationProcessingListModel(QAbstractListModel):
    def __init__(
        self, presenter: AutomationProcessingPresenter, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return self._presenter.getDatasetLabel(index.row())
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                state = self._presenter.getDatasetState(index.row())

                if state == AutomationDatasetState.WAITING:
                    font.setItalic(True)
                elif state == AutomationDatasetState.PROCESSING:
                    font.setBold(True)

                return font

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfDatasets()


class AutomationController(Observer):
    def __init__(
        self,
        core: AutomationCore,
        presenter: AutomationPresenter,
        processingPresenter: AutomationProcessingPresenter,
        view: AutomationView,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._core = core
        self._presenter = presenter
        self._processingController = AutomationProcessingController.createInstance(
            presenter, view.processingView, fileDialogFactory
        )
        self._watchdogController = AutomationWatchdogController.createInstance(
            presenter, view.watchdogView
        )
        self._processingPresenter = processingPresenter
        self._listModel = AutomationProcessingListModel(processingPresenter)
        self._view = view
        self._executeWaitingTasksTimer = QTimer()
        self._automationTimer = QTimer()

    @classmethod
    def createInstance(
        cls,
        core: AutomationCore,
        presenter: AutomationPresenter,
        processingPresenter: AutomationProcessingPresenter,
        view: AutomationView,
        fileDialogFactory: FileDialogFactory,
    ) -> AutomationController:
        controller = cls(core, presenter, processingPresenter, view, fileDialogFactory)
        processingPresenter.add_observer(controller)

        view.processingListView.setModel(controller._listModel)

        view.loadButton.clicked.connect(presenter.loadExistingDatasetsToRepository)
        view.watchButton.setCheckable(True)
        view.watchButton.toggled.connect(presenter.setWatchdogEnabled)
        view.processButton.setCheckable(True)
        view.processButton.toggled.connect(processingPresenter.setProcessingEnabled)
        view.clearButton.clicked.connect(presenter.clearDatasetRepository)

        controller._syncModelToView()

        controller._executeWaitingTasksTimer.timeout.connect(core.executeWaitingTasks)
        controller._executeWaitingTasksTimer.start(60 * 1000)  # TODO customize (in milliseconds)

        controller._automationTimer.timeout.connect(core.refreshDatasetRepository)
        controller._automationTimer.start(10 * 1000)  # TODO customize (in milliseconds)

        return controller

    def _syncModelToView(self) -> None:
        self._view.processButton.setChecked(self._processingPresenter.isProcessingEnabled())
        self._listModel.beginResetModel()
        self._listModel.endResetModel()

        self._view.watchButton.setChecked(self._presenter.isWatchdogEnabled())
        self._view.processButton.setChecked(self._processingPresenter.isProcessingEnabled())

    def _update(self, observable: Observable) -> None:
        if observable is self._processingPresenter:
            self._syncModelToView()
