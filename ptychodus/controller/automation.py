from __future__ import annotations
from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QTimer, QVariant
from PyQt5.QtGui import QFont

from ..api.observer import Observable, Observer
from ..model.automation import (AutomationCore, AutomationDatasetState, AutomationPresenter,
                                AutomationProcessingPresenter)
from ..view.automation import (AutomationView, AutomationParametersView, AutomationProcessingView,
                               AutomationWatchdogView)
from .data import FileDialogFactory


class AutomationParametersController(Observer):

    def __init__(self, presenter: AutomationPresenter, view: AutomationParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: AutomationPresenter, view: AutomationParametersView,
                       fileDialogFactory: FileDialogFactory) -> AutomationParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        for strategy in presenter.getStrategyList():
            view.strategyComboBox.addItem(strategy)

        controller._syncModelToView()

        view.strategyComboBox.textActivated.connect(presenter.setStrategy)
        view.directoryLineEdit.editingFinished.connect(controller._syncDirectoryToModel)
        view.directoryBrowseButton.clicked.connect(controller._browseDirectory)
        view.intervalSpinBox.valueChanged.connect(presenter.setProcessingIntervalInSeconds)
        view.executeButton.clicked.connect(presenter.execute)

        return controller

    def _syncDirectoryToModel(self) -> None:
        dataDirectory = Path(self._view.directoryLineEdit.text())
        self._presenter.setDataDirectory(dataDirectory)

    def _browseDirectory(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(self._view,
                                                                   'Choose Data Directory')

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
        self._view.intervalSpinBox.setRange(intervalLimitsInSeconds.lower,
                                            intervalLimitsInSeconds.upper)
        self._view.intervalSpinBox.setValue(self._presenter.getProcessingIntervalInSeconds())
        self._view.intervalSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationWatchdogController(Observer):

    def __init__(self, presenter: AutomationPresenter, view: AutomationWatchdogView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: AutomationPresenter,
                       view: AutomationWatchdogView) -> AutomationWatchdogController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()

        view.delaySpinBox.valueChanged.connect(presenter.setWatchdogDelayInSeconds)
        view.usePollingObserverCheckBox.toggled.connect(
            presenter.setWatchdogPollingObserverEnabled)
        view.watchButton.toggled.connect(presenter.setWatchdogEnabled)

        return controller

    def _syncModelToView(self) -> None:
        delayLimitsInSeconds = self._presenter.getWatchdogDelayLimitsInSeconds()

        self._view.delaySpinBox.blockSignals(True)
        self._view.delaySpinBox.setRange(delayLimitsInSeconds.lower, delayLimitsInSeconds.upper)
        self._view.delaySpinBox.setValue(self._presenter.getWatchdogDelayInSeconds())
        self._view.delaySpinBox.blockSignals(False)

        self._view.usePollingObserverCheckBox.setChecked(
            self._presenter.isWatchdogPollingObserverEnabled())
        self._view.watchButton.setChecked(self._presenter.isWatchdogEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationProcessingListModel(QAbstractListModel):

    def __init__(self,
                 presenter: AutomationProcessingPresenter,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                label = self._presenter.getDatasetLabel(index.row())
                value = QVariant(label)
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                state = self._presenter.getDatasetState(index.row())

                if state == AutomationDatasetState.WAITING:
                    font.setItalic(True)
                elif state == AutomationDatasetState.PROCESSING:
                    font.setBold(True)

                value = QVariant(font)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfDatasets()


class AutomationProcessingController(Observer):

    def __init__(self, presenter: AutomationProcessingPresenter,
                 view: AutomationProcessingView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._listModel = AutomationProcessingListModel(presenter)

    @classmethod
    def createInstance(cls, presenter: AutomationProcessingPresenter,
                       view: AutomationProcessingView) -> AutomationProcessingController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.listView.setModel(controller._listModel)
        view.processButton.setCheckable(True)
        controller._syncModelToView()
        view.processButton.toggled.connect(presenter.setProcessingEnabled)

        return controller

    def _syncModelToView(self) -> None:
        self._view.processButton.setChecked(self._presenter.isProcessingEnabled())
        self._listModel.beginResetModel()
        self._listModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationController:

    def __init__(self, core: AutomationCore, presenter: AutomationPresenter,
                 processingPresenter: AutomationProcessingPresenter, view: AutomationView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._core = core
        self._parametersController = AutomationParametersController.createInstance(
            presenter, view.parametersView, fileDialogFactory)
        self._watchdogController = AutomationWatchdogController.createInstance(
            presenter, view.watchdogView)
        self._processingController = AutomationProcessingController.createInstance(
            processingPresenter, view.processingView)
        self._timer = QTimer()

    @classmethod
    def createInstance(cls, core: AutomationCore, presenter: AutomationPresenter,
                       processingPresenter: AutomationProcessingPresenter, view: AutomationView,
                       fileDialogFactory: FileDialogFactory) -> AutomationController:
        controller = cls(core, presenter, processingPresenter, view, fileDialogFactory)

        controller._timer.timeout.connect(core.executeWaitingTasks)
        controller._timer.start(60 * 1000)  # TODO customize (in milliseconds)

        view.processingView.processButton.setEnabled(False)  # TODO

        return controller
