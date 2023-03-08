from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ...api.observer import Observable, Observer
from ...model.workflow import WorkflowParametersPresenter
from ...view import WorkflowInputDataView


class WorkflowInputDataController(Observer):

    def __init__(self, presenter: WorkflowParametersPresenter,
                 view: WorkflowInputDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: WorkflowParametersPresenter,
                       view: WorkflowInputDataView) -> WorkflowInputDataController:
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
        dataPath = Path(self._view.posixPathLineEdit.text()).expanduser()
        self._presenter.setInputDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        self._view.endpointIDLineEdit.setText(str(self._presenter.getInputDataEndpointID()))
        self._view.globusPathLineEdit.setText(str(self._presenter.getInputDataGlobusPath()))
        self._view.posixPathLineEdit.setText(str(self._presenter.getInputDataPosixPath()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
