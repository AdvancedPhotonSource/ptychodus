from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ..api.observer import Observable, Observer
from ..model import WorkflowPresenter
from ..view import WorkflowParametersView


class WorkflowParametersController(Observer):

    def __init__(self, presenter: WorkflowPresenter, view: WorkflowParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowPresenter,
                       view: WorkflowParametersView) -> WorkflowParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.dataSourceView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncDataSourceEndpointUUIDToModel)
        view.dataSourceView.pathLineEdit.editingFinished.connect(
            controller._syncDataSourcePathToModel)

        view.dataDestinationView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncDataDestinationEndpointUUIDToModel)
        view.dataDestinationView.pathLineEdit.editingFinished.connect(
            controller._syncDataDestinationPathToModel)

        # TODO UUID QRegularExpressionValidator
        view.computeView.endpointUUIDLineEdit.editingFinished.connect(
            controller._syncComputeEndpointUUIDToModel)

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

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
