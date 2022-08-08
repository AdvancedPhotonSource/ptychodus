from __future__ import annotations
from pathlib import Path
from uuid import UUID

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QRegularExpression,
                          QSortFilterProxyModel, QVariant)
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableView, QWidget

from ..api.observer import Observable, Observer
from ..model import WorkflowPresenter, WorkflowRun
from ..view import WorkflowParametersView


class WorkflowTableModel(QAbstractTableModel):

    def __init__(self, presenter: WorkflowPresenter, parent: QObject = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._sectionHeaders = [
            'Label', 'Start Time', 'Completion Time', 'Status', 'Display Status', 'Run ID'
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

        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            value = QVariant(self._sectionHeaders[section])

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            flowRun = self._flowRuns[index.row()]

            if index.column() == 0:
                value = QVariant(flowRun.label)
            elif index.column() == 1:
                value = QVariant(flowRun.startTime)
            elif index.column() == 2:
                value = QVariant(flowRun.completionTime)
            elif index.column() == 3:
                value = QVariant(flowRun.status)
            elif index.column() == 4:
                value = QVariant(flowRun.displayStatus)
            elif index.column() == 5:
                value = QVariant(flowRun.runId)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._flowRuns)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._sectionHeaders)


class WorkflowController(Observer):

    @staticmethod
    def createUUIDValidator() -> QRegularExpressionValidator:
        hexre = '[0-9A-Fa-f]'
        uuidre = f'{hexre}{{8}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{12}}'
        return QRegularExpressionValidator(QRegularExpression(uuidre))

    def __init__(self, presenter: WorkflowPresenter, parametersView: WorkflowParametersView,
                 tableView: QTableView) -> None:
        super().__init__()
        self._presenter = presenter
        self._parametersView = parametersView
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter, parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(presenter, parametersView, tableView)
        presenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        parametersView.dataSourceView.endpointIDLineEdit.setValidator(
            WorkflowController.createUUIDValidator())
        parametersView.dataSourceView.endpointIDLineEdit.editingFinished.connect(
            controller._syncDataSourceEndpointIDToModel)
        parametersView.dataSourceView.pathLineEdit.editingFinished.connect(
            controller._syncDataSourcePathToModel)

        parametersView.dataDestinationView.endpointIDLineEdit.setValidator(
            WorkflowController.createUUIDValidator())
        parametersView.dataDestinationView.endpointIDLineEdit.editingFinished.connect(
            controller._syncDataDestinationEndpointIDToModel)
        parametersView.dataDestinationView.pathLineEdit.editingFinished.connect(
            controller._syncDataDestinationPathToModel)

        parametersView.computeView.endpointIDLineEdit.setValidator(
            WorkflowController.createUUIDValidator())
        parametersView.computeView.endpointIDLineEdit.editingFinished.connect(
            controller._syncComputeEndpointIDToModel)
        parametersView.computeView.flowIDLineEdit.setValidator(
            WorkflowController.createUUIDValidator())
        parametersView.computeView.flowIDLineEdit.editingFinished.connect(
            controller._syncFlowIDToModel)

        parametersView.buttonBox.authorizeButton.clicked.connect(controller._startAuthorization)
        parametersView.buttonBox.listFlowsButton.clicked.connect(presenter.listFlows)
        parametersView.buttonBox.listFlowRunsButton.clicked.connect(controller._tableModel.refresh)
        parametersView.buttonBox.deployFlowButton.clicked.connect(presenter.deployFlow)
        parametersView.buttonBox.runFlowButton.clicked.connect(controller._runFlow)
        parametersView.authorizeDialog.finished.connect(controller._finishAuthorization)

        controller._syncModelToView()

        return controller

    def _syncDataSourceEndpointIDToModel(self) -> None:
        endpointID = UUID(self._parametersView.dataSourceView.endpointIDLineEdit.text())
        self._presenter.setDataSourceEndpointID(endpointID)

    def _syncDataSourcePathToModel(self) -> None:
        dataSourcePath = Path(self._parametersView.dataSourceView.pathLineEdit.text())
        self._presenter.setDataSourcePath(dataSourcePath)

    def _syncDataDestinationEndpointIDToModel(self) -> None:
        endpointID = UUID(self._parametersView.dataDestinationView.endpointIDLineEdit.text())
        self._presenter.setDataDestinationEndpointID(endpointID)

    def _syncDataDestinationPathToModel(self) -> None:
        dataDestinationPath = Path(self._parametersView.dataDestinationView.pathLineEdit.text())
        self._presenter.setDataDestinationPath(dataDestinationPath)

    def _syncComputeEndpointIDToModel(self) -> None:
        endpointID = UUID(self._parametersView.computeView.endpointIDLineEdit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncFlowIDToModel(self) -> None:
        flowID = UUID(self._parametersView.computeView.flowIDLineEdit.text())
        self._presenter.setFlowID(flowID)

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

    def _runFlow(self) -> None:
        self._presenter.runFlow()
        self._tableModel.refresh()

    def _syncModelToView(self) -> None:
        self._parametersView.dataSourceView.endpointIDLineEdit.setText(
            str(self._presenter.getDataSourceEndpointID()))
        self._parametersView.dataSourceView.pathLineEdit.setText(
            str(self._presenter.getDataSourcePath()))
        self._parametersView.dataDestinationView.endpointIDLineEdit.setText(
            str(self._presenter.getDataDestinationEndpointID()))
        self._parametersView.dataDestinationView.pathLineEdit.setText(
            str(self._presenter.getDataDestinationPath()))
        self._parametersView.computeView.endpointIDLineEdit.setText(
            str(self._presenter.getComputeEndpointID()))
        self._parametersView.computeView.flowIDLineEdit.setText(str(self._presenter.getFlowID()))

        isAuthorized = self._presenter.isAuthorized()
        self._parametersView.buttonBox.authorizeButton.setEnabled(not isAuthorized)
        self._parametersView.buttonBox.listFlowsButton.setEnabled(isAuthorized)
        self._parametersView.buttonBox.listFlowRunsButton.setEnabled(isAuthorized)
        self._parametersView.buttonBox.deployFlowButton.setEnabled(isAuthorized)
        self._parametersView.buttonBox.runFlowButton.setEnabled(isAuthorized)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
