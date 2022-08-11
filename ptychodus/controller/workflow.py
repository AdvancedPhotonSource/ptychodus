from __future__ import annotations
from pathlib import Path
from uuid import UUID
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QRegularExpression,
                          QSortFilterProxyModel, QUrl, QVariant)
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QRegularExpressionValidator
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableView, QWidget

from ..api.observer import Observable, Observer
from ..model import WorkflowPresenter, WorkflowRun
from ..view import WorkflowComputeView, WorkflowDataView, WorkflowParametersView

logger = logging.getLogger(__name__)


class WorkflowDataSourceController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowDataView) -> WorkflowDataSourceController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.pathLineEdit.editingFinished.connect(controller._syncPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setDataSourceEndpointID(endpointID)

    def _syncPathToModel(self) -> None:
        dataPath = Path(self._view.pathLineEdit.text())
        self._presenter.setDataSourcePath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getDataSourceEndpointID()))
        self._view.pathLineEdit.setText(str(self._presenter.getDataSourcePath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class WorkflowDataDestinationController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowDataView) -> WorkflowDataDestinationController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.pathLineEdit.editingFinished.connect(controller._syncPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setDataDestinationEndpointID(endpointID)

    def _syncPathToModel(self) -> None:
        dataPath = Path(self._view.pathLineEdit.text())
        self._presenter.setDataDestinationPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getDataDestinationEndpointID()))
        self._view.pathLineEdit.setText(str(self._presenter.getDataDestinationPath()))

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
        view.flowIDLineEdit.editingFinished.connect(controller._syncFlowIDToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncFlowIDToModel(self) -> None:
        flowID = UUID(self._view.flowIDLineEdit.text())
        self._presenter.setFlowID(flowID)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getComputeEndpointID()))
        self._view.flowIDLineEdit.setText(str(self._presenter.getFlowID()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


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
                    value = QVariant(flowRun.displayStatus)
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
        self._dataSourceController = WorkflowDataSourceController.createInstance(
            presenter, parametersView.dataSourceView)
        self._dataDestinationController = WorkflowDataDestinationController.createInstance(
            presenter, parametersView.dataDestinationView)
        self._computeController = WorkflowComputeController.createInstance(
            presenter, parametersView.computeView)
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
        tableView.clicked.connect(controller._handleTableViewClick)

        parametersView.buttonBox.authorizeButton.clicked.connect(controller._startAuthorization)
        #parametersView.buttonBox.listFlowsButton.clicked.connect(presenter.listFlows)
        parametersView.buttonBox.listFlowRunsButton.clicked.connect(controller._tableModel.refresh)
        #parametersView.buttonBox.deployFlowButton.clicked.connect(presenter.deployFlow)
        parametersView.buttonBox.executeButton.clicked.connect(controller._execute)
        parametersView.authorizeDialog.finished.connect(controller._finishAuthorization)

        parametersView.authorizeDialog.lineEdit.textChanged.connect(
            controller._setAuthorizeDialogButtonsEnabled)
        controller._setAuthorizeDialogButtonsEnabled()

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

    def _execute(self) -> None:
        self._presenter.runFlow()
        self._tableModel.refresh()

    def _syncModelToView(self) -> None:
        isAuthorized = self._presenter.isAuthorized()
        self._parametersView.buttonBox.authorizeButton.setEnabled(not isAuthorized)
        #self._parametersView.buttonBox.listFlowsButton.setEnabled(isAuthorized)
        self._parametersView.buttonBox.listFlowRunsButton.setEnabled(isAuthorized)
        #self._parametersView.buttonBox.deployFlowButton.setEnabled(isAuthorized)
        self._parametersView.buttonBox.executeButton.setEnabled(isAuthorized)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
