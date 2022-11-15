from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QRegularExpression,
                          QSortFilterProxyModel, QTimer, QUrl, QVariant)
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QRegularExpressionValidator
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableView, QWidget

from ..api.observer import Observable, Observer
from ..model import WorkflowPresenter, WorkflowRun
from ..view import WorkflowComputeView, WorkflowDataView, WorkflowParametersView

logger = logging.getLogger(__name__)


class WorkflowInputDataController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowDataView) -> WorkflowInputDataController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.pathLineEdit.editingFinished.connect(controller._syncPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setInputDataEndpointID(endpointID)

    def _syncPathToModel(self) -> None:
        dataPath = self._view.pathLineEdit.text()
        self._presenter.setInputDataPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getInputDataEndpointID()))
        self._view.pathLineEdit.setText(str(self._presenter.getInputDataPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowOutputDataController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowDataView) -> WorkflowOutputDataController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.pathLineEdit.editingFinished.connect(controller._syncPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setOutputDataEndpointID(endpointID)

    def _syncPathToModel(self) -> None:
        dataPath = self._view.pathLineEdit.text()
        self._presenter.setOutputDataPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getOutputDataEndpointID()))
        self._view.pathLineEdit.setText(str(self._presenter.getOutputDataPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowComputeController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowComputeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowComputeView) -> WorkflowComputeController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.dataEndpointIDLineEdit.editingFinished.connect(controller._syncDataEndpointIDToModel)
        view.pathLineEdit.editingFinished.connect(controller._syncPathToModel)
        view.flowIDLineEdit.editingFinished.connect(controller._syncFlowIDToModel)
        view.reconstructActionIDLineEdit.editingFinished.connect(
            controller._syncReconstructActionIDToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncDataEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.dataEndpointIDLineEdit.text())
        self._presenter.setComputeDataEndpointID(endpointID)

    def _syncPathToModel(self) -> None:
        dataPath = self._view.pathLineEdit.text()
        self._presenter.setComputeDataPath(dataPath)

    def _syncFlowIDToModel(self) -> None:
        flowID = UUID(self._view.flowIDLineEdit.text())
        self._presenter.setFlowID(flowID)

    def _syncReconstructActionIDToModel(self) -> None:
        actionID = UUID(self._view.reconstructActionIDLineEdit.text())
        self._presenter.setReconstructActionID(actionID)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getComputeEndpointID()))
        self._view.dataEndpointIDLineEdit.setText(str(self._presenter.getComputeDataEndpointID()))
        self._view.pathLineEdit.setText(str(self._presenter.getComputeDataPath()))
        self._view.flowIDLineEdit.setText(str(self._presenter.getFlowID()))
        self._view.reconstructActionIDLineEdit.setText(
            str(self._presenter.getReconstructActionID()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowTableModel(QAbstractTableModel):

    def __init__(self, presenter: WorkflowPresenter, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._sectionHeaders = [
            'Label', 'Start Time', 'Completion Time', 'Flow Status', 'Current Action', 'Run ID'
        ]
        self._flowRuns: list[WorkflowRun] = list()

    def refresh(self) -> None:
        self.beginResetModel()
        self._flowRuns = self._presenter.listFlowRuns()
        self.endResetModel()

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                value = QVariant(self._sectionHeaders[section])
            elif orientation == Qt.Vertical:
                value = QVariant(section)

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            flowRun = self._flowRuns[index.row()]

            if role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(flowRun.label)
                elif index.column() == 1:
                    value = QVariant(flowRun.startTime)
                elif index.column() == 2:
                    value = QVariant(flowRun.completionTime)
                elif index.column() == 3:
                    value = QVariant(flowRun.status)
                elif index.column() == 4:
                    value = QVariant(flowRun.action)
                elif index.column() == 5:
                    value = QVariant(flowRun.runID)
            elif index.column() == 5:
                if role == Qt.ToolTipRole:
                    value = QVariant(flowRun.runURL)
                elif role == Qt.FontRole:
                    font = QFont()
                    font.setUnderline(True)
                    value = QVariant(font)
                elif role == Qt.ForegroundRole:
                    color = QColor(Qt.blue)
                    value = QVariant(color)
                elif role == Qt.UserRole:
                    value = QVariant(QUrl(flowRun.runURL))

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._flowRuns)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._sectionHeaders)


class WorkflowController(Observer):

    def __init__(self, presenter: WorkflowPresenter, parametersView: WorkflowParametersView,
                 tableView: QTableView) -> None:
        super().__init__()
        self._presenter = presenter
        self._parametersView = parametersView
        self._inputDataController = WorkflowInputDataController.createInstance(
            presenter, parametersView.inputDataView)
        self._outputDataController = WorkflowOutputDataController.createInstance(
            presenter, parametersView.outputDataView)
        self._computeController = WorkflowComputeController.createInstance(
            presenter, parametersView.computeView)
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()
        self._statusRefreshTimer = QTimer()

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter, parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(presenter, parametersView, tableView)
        presenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.clicked.connect(controller._handleTableViewClick)

        parametersView.statusView.refreshIntervalSpinBox.setRange(1, 600)
        parametersView.statusView.refreshIntervalSpinBox.valueChanged.connect(
            presenter.setStatusRefreshIntervalInSeconds)
        parametersView.buttonBox.authorizeButton.clicked.connect(controller._startAuthorization)
        parametersView.buttonBox.executeButton.clicked.connect(controller._execute)
        parametersView.authorizeDialog.finished.connect(controller._finishAuthorization)

        parametersView.authorizeDialog.lineEdit.textChanged.connect(
            controller._setAuthorizeDialogButtonsEnabled)
        controller._setAuthorizeDialogButtonsEnabled()

        controller._statusRefreshTimer.setSingleShot(True)
        controller._statusRefreshTimer.timeout.connect(controller._refreshStatus)
        controller._syncModelToView()

        return controller

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.UserRole)
            logger.debug(f'Opening URL: \"{url.toString()}\"')
            QDesktopServices.openUrl(url)

    def _setAuthorizeDialogButtonsEnabled(self) -> None:
        text = self._parametersView.authorizeDialog.lineEdit.text()
        self._parametersView.authorizeDialog.okButton.setEnabled(len(text) > 0)

    def _restartStatusRefreshTimer(self) -> None:
        seconds = 1000 * self._presenter.getStatusRefreshIntervalInSeconds()

        if seconds > 0:
            self._statusRefreshTimer.start(seconds)
        else:
            # TODO can't restart
            self._statusRefreshTimer.stop()

    def _startAuthorization(self) -> None:
        authorizeURL = self._presenter.getAuthorizeURL()
        text = f'Input the Globus authorization code from <a href="{authorizeURL}">this link</a>:'

        self._parametersView.authorizeDialog.label.setText(text)
        self._parametersView.authorizeDialog.lineEdit.clear()
        self._parametersView.authorizeDialog.open()

    def _finishAuthorization(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        authCode = self._parametersView.authorizeDialog.lineEdit.text()
        self._presenter.setAuthorizationCode(authCode)
        self._restartStatusRefreshTimer()

    def _refreshStatus(self) -> None:
        self._tableModel.refresh()
        self._restartStatusRefreshTimer()

    def _execute(self) -> None:
        self._presenter.runFlow()

    def _syncModelToView(self) -> None:
        self._parametersView.statusView.refreshIntervalSpinBox.setValue(
            self._presenter.getStatusRefreshIntervalInSeconds())

        isAuthorized = self._presenter.isAuthorized()
        self._parametersView.buttonBox.authorizeButton.setEnabled(not isAuthorized)
        self._parametersView.buttonBox.executeButton.setEnabled(isAuthorized)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
