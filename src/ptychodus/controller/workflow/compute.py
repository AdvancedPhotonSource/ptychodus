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
            controller._sync_compute_endpoint_id_to_model
        )
        view.data_endpoint_id_line_edit.editingFinished.connect(
            controller._sync_data_endpoint_id_to_model
        )
        view.data_globus_path_line_edit.editingFinished.connect(
            controller._sync_globus_path_to_model
        )
        view.data_posix_path_line_edit.editingFinished.connect(controller._sync_posix_path_to_model)

        controller._sync_model_to_view()

        return controller

    def _sync_compute_endpoint_id_to_model(self) -> None:
        endpoint_id = UUID(self._view.compute_endpoint_id_line_edit.text())
        self._presenter.set_compute_endpoint_id(endpoint_id)

    def _sync_data_endpoint_id_to_model(self) -> None:
        endpoint_id = UUID(self._view.data_endpoint_id_line_edit.text())
        self._presenter.set_compute_data_endpoint_id(endpoint_id)

    def _sync_globus_path_to_model(self) -> None:
        data_path = self._view.data_globus_path_line_edit.text()
        self._presenter.set_compute_data_globus_path(data_path)

    def _sync_posix_path_to_model(self) -> None:
        data_path = Path(self._view.data_posix_path_line_edit.text())
        self._presenter.set_compute_data_posix_path(data_path)

    def _sync_model_to_view(self) -> None:
        self._view.compute_endpoint_id_line_edit.setText(
            str(self._presenter.get_compute_endpoint_id())
        )
        self._view.data_endpoint_id_line_edit.setText(
            str(self._presenter.get_compute_data_endpoint_id())
        )
        self._view.data_globus_path_line_edit.setText(
            str(self._presenter.get_compute_data_globus_path())
        )
        self._view.data_posix_path_line_edit.setText(
            str(self._presenter.get_compute_data_posix_path())
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
