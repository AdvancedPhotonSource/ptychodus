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
from ..model.workflow import (WorkflowAuthorizationPresenter, WorkflowExecutionPresenter,
                              WorkflowParametersPresenter, WorkflowRun)
from ..view import WorkflowComputeView, WorkflowDataView, WorkflowParametersView

logger = logging.getLogger(__name__)


class WorkflowInputDataController(Observer):

    def __init__(self, presenter: WorkflowParametersPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowParametersPresenter,
                       view: WorkflowDataView) -> WorkflowInputDataController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.globusPathLineEdit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.posixPathLineEdit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setInputDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.globusPathLineEdit.text()
        self._presenter.setInputDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = self._view.posixPathLineEdit.text()
        self._presenter.setInputDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getInputDataEndpointID()))
        self._view.globusPathLineEdit.setText(str(self._presenter.getInputDataGlobusPath()))
        self._view.posixPathLineEdit.setText(str(self._presenter.getInputDataPosixPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowOutputDataController(Observer):

    def __init__(self, presenter: WorkflowParametersPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowParametersPresenter,
                       view: WorkflowDataView) -> WorkflowOutputDataController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.globusPathLineEdit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.posixPathLineEdit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setOutputDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.globusPathLineEdit.text()
        self._presenter.setOutputDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = self._view.posixPathLineEdit.text()
        self._presenter.setOutputDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getOutputDataEndpointID()))
        self._view.globusPathLineEdit.setText(str(self._presenter.getOutputDataGlobusPath()))
        self._view.posixPathLineEdit.setText(str(self._presenter.getOutputDataPosixPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowComputeController(Observer):

    def __init__(self, presenter: WorkflowParametersPresenter, view: WorkflowComputeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowParametersPresenter,
                       view: WorkflowComputeView) -> WorkflowComputeController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.funcXEndpointIDLineEdit.editingFinished.connect(
            controller._syncFuncXEndpointIDToModel)
        view.dataEndpointIDLineEdit.editingFinished.connect(controller._syncDataEndpointIDToModel)
        view.dataGlobusPathLineEdit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.dataPosixPathLineEdit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._syncModelToView()

        return controller

    def _syncFuncXEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.funcXEndpointIDLineEdit.text())
        self._presenter.setComputeFuncXEndpointID(endpointID)

    def _syncDataEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.dataEndpointIDLineEdit.text())
        self._presenter.setComputeDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.dataGlobusPathLineEdit.text()
        self._presenter.setComputeDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = self._view.dataPosixPathLineEdit.text()
        self._presenter.setComputeDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.funcXEndpointIDLineEdit.setText(str(
            self._presenter.getComputeFuncXEndpointID()))
        self._view.dataEndpointIDLineEdit.setText(str(self._presenter.getComputeDataEndpointID()))
        self._view.dataGlobusPathLineEdit.setText(str(self._presenter.getComputeDataGlobusPath()))
        self._view.dataPosixPathLineEdit.setText(str(self._presenter.getComputeDataPosixPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: WorkflowExecutionPresenter,
                 parent: Optional[QObject] = None) -> None:
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

    def __init__(self, parametersPresenter: WorkflowParametersPresenter,
                 authorizationPresenter: WorkflowAuthorizationPresenter,
                 executionPresenter: WorkflowExecutionPresenter,
                 parametersView: WorkflowParametersView, tableView: QTableView) -> None:
        super().__init__()
        self._parametersPresenter = parametersPresenter
        self._authorizationPresenter = authorizationPresenter
        self._executionPresenter = executionPresenter
        self._parametersView = parametersView
        self._inputDataController = WorkflowInputDataController.createInstance(
            parametersPresenter, parametersView.inputDataView)
        self._outputDataController = WorkflowOutputDataController.createInstance(
            parametersPresenter, parametersView.outputDataView)
        self._computeController = WorkflowComputeController.createInstance(
            parametersPresenter, parametersView.computeView)
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(executionPresenter)
        self._proxyModel = QSortFilterProxyModel()
        self._statusRefreshTimer = QTimer()

    @classmethod
    def createInstance(cls, parametersPresenter: WorkflowParametersPresenter,
                       authorizationPresenter: WorkflowAuthorizationPresenter,
                       executionPresenter: WorkflowExecutionPresenter,
                       parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(parametersPresenter, authorizationPresenter, executionPresenter,
                         parametersView, tableView)
        parametersPresenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.clicked.connect(controller._handleTableViewClick)

        parametersView.statusView.refreshIntervalSpinBox.setRange(1, 600)
        parametersView.statusView.refreshIntervalSpinBox.valueChanged.connect(
            parametersPresenter.setStatusRefreshIntervalInSeconds)
        parametersView.executeButton.clicked.connect(controller._execute)
        # FIXME controller: extract AuthController, use authTimer to present GUI to get code[str] from user
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
        seconds = 1000 * self._parametersPresenter.getStatusRefreshIntervalInSeconds()

        if seconds > 0:
            self._statusRefreshTimer.start(seconds)
        else:
            # FIXME can't restart
            self._statusRefreshTimer.stop()

    def _startAuthorization(self) -> None:  # FIXME need to trigger
        authorizeURL = self._authorizationPresenter.getAuthorizeURL()
        text = f'Input the Globus authorization code from <a href="{authorizeURL}">this link</a>:'

        self._parametersView.authorizeDialog.label.setText(text)
        self._parametersView.authorizeDialog.lineEdit.clear()
        self._parametersView.authorizeDialog.open()

    def _finishAuthorization(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        authCode = self._parametersView.authorizeDialog.lineEdit.text()
        self._authorizationPresenter.setCodeFromAuthorizeURL(authCode)
        self._restartStatusRefreshTimer()

    def _refreshStatus(self) -> None:
        self._tableModel.refresh()
        self._restartStatusRefreshTimer()

    def _execute(self) -> None:
        self._executionPresenter.runFlow(label='Ptychodus')  # TODO label

    def _syncModelToView(self) -> None:
        intervalInSeconds = self._parametersPresenter.getStatusRefreshIntervalInSeconds()
        self._parametersView.statusView.refreshIntervalSpinBox.setValue(intervalInSeconds)

    def update(self, observable: Observable) -> None:
        if observable is self._parametersPresenter:
            self._syncModelToView()
