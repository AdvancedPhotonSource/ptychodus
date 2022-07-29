from __future__ import annotations
from pathlib import Path
from uuid import UUID

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator

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

        view.dataSourceView.endpointUUIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.dataSourceView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncDataSourceEndpointUUIDToModel)
        view.dataSourceView.pathLineEdit.editingFinished.connect(
            controller._syncDataSourcePathToModel)

        view.dataDestinationView.endpointUUIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.dataDestinationView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncDataDestinationEndpointUUIDToModel)
        view.dataDestinationView.pathLineEdit.editingFinished.connect(
            controller._syncDataDestinationPathToModel)

        view.computeView.endpointUUIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.computeView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncComputeEndpointUUIDToModel)
        view.computeView.flowUUIDLineEdit.setValidator(
            WorkflowParametersController.createUUIDValidator())
        view.computeView.flowUUIDLineEdit.editingFinished.connect(controller._syncFlowUUIDToModel)

        view.launchButton.clicked.connect(presenter.launchWorkflow)

        controller._syncModelToView()

        return controller

    def _syncDataSourceEndpointUUIDToModel(self) -> None:
        endpointUUID = UUID(self._view.dataSourceView.endpointUUIDLineEdit.text())
        self._presenter.setDataSourceEndpointUUID(endpointUUID)

    def _syncDataSourcePathToModel(self) -> None:
        dataSourcePath = Path(self._view.dataSourceView.pathLineEdit.text())
        self._presenter.setDataSourcePath(dataSourcePath)

    def _syncDataDestinationEndpointUUIDToModel(self) -> None:
        endpointUUID = UUID(self._view.dataDestinationView.endpointUUIDLineEdit.text())
        self._presenter.setDataDestinationEndpointUUID(endpointUUID)

    def _syncDataDestinationPathToModel(self) -> None:
        dataDestinationPath = Path(self._view.dataDestinationView.pathLineEdit.text())
        self._presenter.setDataDestinationPath(dataDestinationPath)

    def _syncComputeEndpointUUIDToModel(self) -> None:
        endpointUUID = UUID(self._view.computeView.endpointUUIDLineEdit.text())
        self._presenter.setComputeEndpointUUID(endpointUUID)

    def _syncFlowUUIDToModel(self) -> None:
        flowUUID = UUID(self._view.computeView.flowUUIDLineEdit.text())
        self._presenter.setFlowUUID(flowUUID)

    def _syncModelToView(self) -> None:
        self._view.dataSourceView.endpointUUIDLineEdit.setText(
            str(self._presenter.getDataSourceEndpointUUID()))
        self._view.dataSourceView.pathLineEdit.setText(str(self._presenter.getDataSourcePath()))
        self._view.dataDestinationView.endpointUUIDLineEdit.setText(
            str(self._presenter.getDataDestinationEndpointUUID()))
        self._view.dataDestinationView.pathLineEdit.setText(
            str(self._presenter.getDataDestinationPath()))
        self._view.computeView.endpointUUIDLineEdit.setText(
            str(self._presenter.getComputeEndpointUUID()))
        self._view.computeView.flowUUIDLineEdit.setText(str(self._presenter.getFlowUUID()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
