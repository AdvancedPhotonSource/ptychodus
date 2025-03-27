from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer

from ...model.workflow import WorkflowParametersPresenter
from ...view.workflow import WorkflowOutputDataView


class WorkflowOutputDataController(Observer):
    def __init__(
        self, presenter: WorkflowParametersPresenter, view: WorkflowOutputDataView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def create_instance(
        cls, presenter: WorkflowParametersPresenter, view: WorkflowOutputDataView
    ) -> WorkflowOutputDataController:
        controller = cls(presenter, view)
        presenter.add_observer(controller)

        view.round_trip_check_box.toggled.connect(presenter.setRoundTripEnabled)
        view.endpoint_id_line_edit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.globus_path_line_edit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.posix_path_line_edit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._sync_model_to_view()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpoint_id_line_edit.text())
        self._presenter.setOutputDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.globus_path_line_edit.text()
        self._presenter.setOutputDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = Path(self._view.posix_path_line_edit.text())
        self._presenter.setOutputDataPosixPath(dataPath)

    def _sync_model_to_view(self) -> None:
        isRoundTripEnabled = self._presenter.isRoundTripEnabled()
        self._view.round_trip_check_box.setChecked(isRoundTripEnabled)
        self._view.endpoint_id_line_edit.setText(str(self._presenter.getOutputDataEndpointID()))
        self._view.endpoint_id_line_edit.setEnabled(not isRoundTripEnabled)
        self._view.globus_path_line_edit.setText(str(self._presenter.getOutputDataGlobusPath()))
        self._view.globus_path_line_edit.setEnabled(not isRoundTripEnabled)
        self._view.posix_path_line_edit.setText(str(self._presenter.getOutputDataPosixPath()))
        self._view.posix_path_line_edit.setEnabled(not isRoundTripEnabled)

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
