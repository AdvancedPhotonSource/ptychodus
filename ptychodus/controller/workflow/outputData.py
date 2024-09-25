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
    def createInstance(
        cls, presenter: WorkflowParametersPresenter, view: WorkflowOutputDataView
    ) -> WorkflowOutputDataController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.roundTripCheckBox.toggled.connect(presenter.setRoundTripEnabled)
        view.endpointIDLineEdit.editingFinished.connect(controller._syncEndpointIDToModel)
        view.globusPathLineEdit.editingFinished.connect(controller._syncGlobusPathToModel)
        view.posixPathLineEdit.editingFinished.connect(controller._syncPosixPathToModel)

        controller._syncModelToView()

        return controller

    def _syncEndpointIDToModel(self) -> None:
        endpointID = UUID(self._view.endpointIDLineEdit.text())
        self._presenter.setOutputDataEndpointID(endpointID)

    def _syncGlobusPathToModel(self) -> None:
        dataPath = self._view.globusPathLineEdit.text()
        self._presenter.setOutputDataGlobusPath(dataPath)

    def _syncPosixPathToModel(self) -> None:
        dataPath = Path(self._view.posixPathLineEdit.text())
        self._presenter.setOutputDataPosixPath(dataPath)

    def _syncModelToView(self) -> None:
        isRoundTripEnabled = self._presenter.isRoundTripEnabled()
        self._view.roundTripCheckBox.setChecked(isRoundTripEnabled)
        self._view.endpointIDLineEdit.setText(str(self._presenter.getOutputDataEndpointID()))
        self._view.endpointIDLineEdit.setEnabled(not isRoundTripEnabled)
        self._view.globusPathLineEdit.setText(str(self._presenter.getOutputDataGlobusPath()))
        self._view.globusPathLineEdit.setEnabled(not isRoundTripEnabled)
        self._view.posixPathLineEdit.setText(str(self._presenter.getOutputDataPosixPath()))
        self._view.posixPathLineEdit.setEnabled(not isRoundTripEnabled)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
