from __future__ import annotations
from pathlib import Path

from ..api.observer import Observable, Observer
from ..model.automation import AutomationPresenter
from ..view import AutomationParametersView, AutomationWatchdogView
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


class AutomationController:

    def __init__(self, presenter: AutomationPresenter, view: AutomationParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._watchdogController = AutomationWatchdogController.createInstance(
            presenter, view.watchdogView, fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: AutomationPresenter, view: AutomationParametersView,
                       fileDialogFactory: FileDialogFactory) -> AutomationController:
        return cls(presenter, view, fileDialogFactory)
