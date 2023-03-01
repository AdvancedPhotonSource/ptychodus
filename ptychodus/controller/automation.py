from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QTimer, QVariant
from PyQt5.QtGui import QFont

from ..api.observer import Observable, Observer
from ..model.automation import (AutomationCore, AutomationDatasetState, AutomationPresenter,
                                AutomationProcessingPresenter)
from ..view import AutomationParametersView, AutomationProcessingView, AutomationWatchdogView
from .data import FileDialogFactory


class AutomationWatchdogController(Observer):

    def __init__(self, presenter: AutomationPresenter, view: AutomationWatchdogView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: AutomationPresenter, view: AutomationWatchdogView,
                       fileDialogFactory: FileDialogFactory) -> AutomationWatchdogController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        for strategy in presenter.getStrategyList():
            view.strategyComboBox.addItem(strategy)

        view.watchButton.setCheckable(True)

        controller._syncModelToView()

        view.strategyComboBox.currentTextChanged.connect(presenter.setStrategy)
        view.directoryLineEdit.editingFinished.connect(controller._syncDirectoryToModel)
        view.directoryBrowseButton.clicked.connect(controller._browseDirectory)
        view.delaySpinBox.valueChanged.connect(presenter.setWatchdogDelayInSeconds)
        view.watchButton.toggled.connect(presenter.setWatchdogEnabled)

        return controller

    def _syncDirectoryToModel(self) -> None:
        watchdogDirectory = Path(self._view.directoryLineEdit.text())
        self._presenter.setWatchdogDirectory(watchdogDirectory)

    def _browseDirectory(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(self._view,
                                                                   'Choose Watchdog Directory')

        if dirPath:
            self._presenter.setWatchdogDirectory(dirPath)

    def _syncModelToView(self) -> None:
        self._view.strategyComboBox.blockSignals(True)
        self._view.strategyComboBox.setCurrentText(self._presenter.getStrategy())
        self._view.strategyComboBox.blockSignals(False)

        watchdogDirectory = self._presenter.getWatchdogDirectory()

        if watchdogDirectory:
            self._view.directoryLineEdit.setText(str(watchdogDirectory))
        else:
            self._view.directoryLineEdit.clear()

        delayLimitsInSeconds = self._presenter.getWatchdogDelayLimitsInSeconds()

        self._view.delaySpinBox.blockSignals(True)
        self._view.delaySpinBox.setRange(delayLimitsInSeconds.lower, delayLimitsInSeconds.upper)
        self._view.delaySpinBox.setValue(self._presenter.getWatchdogDelayInSeconds())
        self._view.delaySpinBox.blockSignals(False)

        self._view.watchButton.setChecked(self._presenter.isWatchdogEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class AutomationProcessingListModel(QAbstractListModel):

    def __init__(self,
                 presenter: AutomationProcessingPresenter,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole:
                label = self._presenter.getDatasetLabel(index.row())
                value = QVariant(label)
            elif role == Qt.FontRole:
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
                 processingPresenter: AutomationProcessingPresenter,
                 view: AutomationParametersView, fileDialogFactory: FileDialogFactory) -> None:
        self._core = core
        self._watchdogController = AutomationWatchdogController.createInstance(
            presenter, view.watchdogView, fileDialogFactory)
        self._processingController = AutomationProcessingController.createInstance(
            processingPresenter, view.processingView)
        self._timer = QTimer()

    @classmethod
    def createInstance(cls, core: AutomationCore, presenter: AutomationPresenter,
                       processingPresenter: AutomationProcessingPresenter,
                       view: AutomationParametersView,
                       fileDialogFactory: FileDialogFactory) -> AutomationController:
        controller = cls(core, presenter, processingPresenter, view, fileDialogFactory)

        controller._timer.timeout.connect(core.executeWaitingTasks)
        controller._timer.start(1000)  # TODO customize

        view.processingView.processButton.setEnabled(False)  # TODO

        return controller
