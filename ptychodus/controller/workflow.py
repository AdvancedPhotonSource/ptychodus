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

        view.launchButton.clicked.connect(presenter.launchWorkflow)

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

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
