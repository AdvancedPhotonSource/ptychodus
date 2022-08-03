from __future__ import annotations
from pathlib import Path
from uuid import UUID

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QDialog

from ..api.observer import Observable, Observer
from ..model import WorkflowPresenter
from ..view import WorkflowParametersView


class WorkflowParametersController(Observer):

    @staticmethod
    def createUUIDValidator() -> QRegularExpressionValidator:
        hexre = '[0-9A-Fa-f]'
        uuidre = f'{hexre}{{8}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{12}}'
        return QRegularExpressionValidator(QRegularExpression(uuidre))

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowParametersView) -> WorkflowParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.dataSourceView.endpointIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.dataSourceView.endpointIDLineEdit.editingFinished.connect(
            controller._syncDataSourceEndpointIDToModel)
        view.dataSourceView.pathLineEdit.editingFinished.connect(
            controller._syncDataSourcePathToModel)

        view.dataDestinationView.endpointIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.dataDestinationView.endpointIDLineEdit.editingFinished.connect(
            controller._syncDataDestinationEndpointIDToModel)
        view.dataDestinationView.pathLineEdit.editingFinished.connect(
            controller._syncDataDestinationPathToModel)

        view.computeView.endpointIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.computeView.endpointIDLineEdit.editingFinished.connect(
            controller._syncComputeEndpointIDToModel)
        view.computeView.flowIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.computeView.flowIDLineEdit.editingFinished.connect(controller._syncFlowIDToModel)

        view.buttonBox.authorizeButton.clicked.connect(controller._startAuthorization)
        view.buttonBox.listFlowsButton.clicked.connect(presenter.printFlows)
        view.buttonBox.deployFlowButton.clicked.connect(presenter.deployFlow)
        view.buttonBox.runFlowButton.clicked.connect(presenter.runFlow)
        view.authorizeDialog.finished.connect(controller._finishAuthorization)

        controller._syncModelToView()

        return controller

    def _syncDataSourceEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.dataSourceView.endpointIDLineEdit.text())
        self._presenter.setDataSourceEndpointID(endpointID)

    def _syncDataSourcePathToModel(self) -> None:
        dataSourcePath = Path(self._view.dataSourceView.pathLineEdit.text())
        self._presenter.setDataSourcePath(dataSourcePath)

    def _syncDataDestinationEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.dataDestinationView.endpointIDLineEdit.text())
        self._presenter.setDataDestinationEndpointID(endpointID)

    def _syncDataDestinationPathToModel(self) -> None:
        dataDestinationPath = Path(self._view.dataDestinationView.pathLineEdit.text())
        self._presenter.setDataDestinationPath(dataDestinationPath)

    def _syncComputeEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.computeView.endpointIDLineEdit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncFlowIDToModel(self) -> None:
        flowID = UUID(self._view.computeView.flowIDLineEdit.text())
        self._presenter.setFlowID(flowID)

    def _startAuthorization(self) -> None:
        authorizeURL = self._presenter.getAuthorizeURL()
        text = f'Input the Globus authorization code from <a href="{authorizeURL}">this link</a>:'

        self._view.authorizeDialog.label.setText(text)
        self._view.authorizeDialog.lineEdit.clear()
        self._view.authorizeDialog.open()

    def _finishAuthorization(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        authCode = self._view.authorizeDialog.lineEdit.text()
        self._presenter.setAuthorizationCode(authCode)

    def _syncModelToView(self) -> None:
        self._view.dataSourceView.endpointIDLineEdit.setText(
            str(self._presenter.getDataSourceEndpointID()))
        self._view.dataSourceView.pathLineEdit.setText(str(self._presenter.getDataSourcePath()))
        self._view.dataDestinationView.endpointIDLineEdit.setText(
            str(self._presenter.getDataDestinationEndpointID()))
        self._view.dataDestinationView.pathLineEdit.setText(
            str(self._presenter.getDataDestinationPath()))
        self._view.computeView.endpointIDLineEdit.setText(
            str(self._presenter.getComputeEndpointID()))
        self._view.computeView.flowIDLineEdit.setText(str(self._presenter.getFlowID()))

        isAuthorized = self._presenter.isAuthorized()
        self._view.buttonBox.authorizeButton.setEnabled(not isAuthorized)
        self._view.buttonBox.listFlowsButton.setEnabled(isAuthorized)
        self._view.buttonBox.deployFlowButton.setEnabled(isAuthorized)
        self._view.buttonBox.runFlowButton.setEnabled(isAuthorized)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
