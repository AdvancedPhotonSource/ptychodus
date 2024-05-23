from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer

from ...model.workflow import WorkflowParametersPresenter
from ...view.workflow import WorkflowComputeView


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

        view.computeEndpointIDLineEdit.editingFinished.connect(
            controller._syncComputeEndpointIDToModel)
        view.dataEndpointIDLineEdit.editingFinished.connect(controller._syncDataEndpointIDToModel)
        view.dataGlobusPathLineEdit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.dataPosixPathLineEdit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._syncModelToView()

        return controller

    def _syncComputeEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.computeEndpointIDLineEdit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncDataEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.dataEndpointIDLineEdit.text())
        self._presenter.setComputeDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.dataGlobusPathLineEdit.text()
        self._presenter.setComputeDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = Path(self._view.dataPosixPathLineEdit.text())
        self._presenter.setComputeDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.computeEndpointIDLineEdit.setText(str(self._presenter.getComputeEndpointID()))
        self._view.dataEndpointIDLineEdit.setText(str(self._presenter.getComputeDataEndpointID()))
        self._view.dataGlobusPathLineEdit.setText(str(self._presenter.getComputeDataGlobusPath()))
        self._view.dataPosixPathLineEdit.setText(str(self._presenter.getComputeDataPosixPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
