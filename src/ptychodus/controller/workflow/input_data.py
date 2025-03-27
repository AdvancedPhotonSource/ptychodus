from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer

from ...model.workflow import WorkflowParametersPresenter
from ...view.workflow import WorkflowInputDataView


class WorkflowInputDataController(Observer):
    def __init__(self, presenter: WorkflowParametersPresenter, view: WorkflowInputDataView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def create_instance(
        cls, presenter: WorkflowParametersPresenter, view: WorkflowInputDataView
    ) -> WorkflowInputDataController:
        controller = cls(presenter, view)
        presenter.add_observer(controller)

        view.endpoint_id_line_edit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.globus_path_line_edit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.posix_path_line_edit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._sync_model_to_view()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpoint_id_line_edit.text())
        self._presenter.setInputDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.globus_path_line_edit.text()
        self._presenter.setInputDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = Path(self._view.posix_path_line_edit.text()).expanduser()
        self._presenter.setInputDataPosixPath(dataPath)

    def _sync_model_to_view(self) -> None:
        self._view.endpoint_id_line_edit.setText(str(self._presenter.getInputDataEndpointID()))
        self._view.globus_path_line_edit.setText(str(self._presenter.getInputDataGlobusPath()))
        self._view.posix_path_line_edit.setText(str(self._presenter.getInputDataPosixPath()))

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
