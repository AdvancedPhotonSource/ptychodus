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
    def create_instance(
        cls, presenter: WorkflowParametersPresenter, view: WorkflowComputeView
    ) -> WorkflowComputeController:
        controller = cls(presenter, view)
        presenter.add_observer(controller)

        view.compute_endpoint_id_line_edit.editingFinished.connect(
            controller._syncComputeEndpointIDToModel
        )
        view.data_endpoint_id_line_edit.editingFinished.connect(
            controller._syncDataEndpointIDToModel
        )
        view.data_globus_path_line_edit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.data_posix_path_line_edit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._sync_model_to_view()

        return controller

    def _syncComputeEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.compute_endpoint_id_line_edit.text())
        self._presenter.setComputeEndpointID(endpointID)

    def _syncDataEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.data_endpoint_id_line_edit.text())
        self._presenter.setComputeDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.data_globus_path_line_edit.text()
        self._presenter.setComputeDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = Path(self._view.data_posix_path_line_edit.text())
        self._presenter.setComputeDataPosixPath(dataPath)

    def _sync_model_to_view(self) -> None:
        self._view.compute_endpoint_id_line_edit.setText(
            str(self._presenter.getComputeEndpointID())
        )
        self._view.data_endpoint_id_line_edit.setText(
            str(self._presenter.getComputeDataEndpointID())
        )
        self._view.data_globus_path_line_edit.setText(
            str(self._presenter.getComputeDataGlobusPath())
        )
        self._view.data_posix_path_line_edit.setText(str(self._presenter.getComputeDataPosixPath()))

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
