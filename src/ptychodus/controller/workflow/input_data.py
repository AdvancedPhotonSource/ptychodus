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

        view.endpoint_id_line_edit.editingFinished.connect(controller._sync_endpoint_id_to_model)
        view.globus_path_line_edit.editingFinished.connect(controller._sync_globus_path_to_model)
        view.posix_path_line_edit.editingFinished.connect(controller._sync_posix_path_to_model)

        controller._sync_model_to_view()

        return controller

    def _sync_endpoint_id_to_model(self) -> None:
        endpoint_id = UUID(self._view.endpoint_id_line_edit.text())
        self._presenter.set_input_data_endpoint_id(endpoint_id)

    def _sync_globus_path_to_model(self) -> None:
        data_path = self._view.globus_path_line_edit.text()
        self._presenter.set_input_data_globus_path(data_path)

    def _sync_posix_path_to_model(self) -> None:
        data_path = Path(self._view.posix_path_line_edit.text()).expanduser()
        self._presenter.set_input_data_posix_path(data_path)

    def _sync_model_to_view(self) -> None:
        self._view.endpoint_id_line_edit.setText(str(self._presenter.get_input_data_endpoint_id()))
        self._view.globus_path_line_edit.setText(str(self._presenter.get_input_data_globus_path()))
        self._view.posix_path_line_edit.setText(str(self._presenter.get_input_data_posix_path()))

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
