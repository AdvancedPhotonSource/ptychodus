from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.data import DiffractionDatasetPresenter
from ...view.detector import InspectDatasetDialog
from ..tree import SimpleTreeModel


class InspectDatasetController(Observer):

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 dialog: InspectDatasetDialog) -> None:
        self._datasetPresenter = datasetPresenter
        self._dialog = dialog
        self._treeModel = SimpleTreeModel(datasetPresenter.getContentsTree())

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       dialog: InspectDatasetDialog) -> InspectDatasetController:
        controller = cls(datasetPresenter, dialog)
        datasetPresenter.addObserver(controller)
        dialog.treeView.setModel(controller._treeModel)
        controller._syncModelToView()
        return controller

    def inspectDataset(self) -> None:
        self._dialog.show()

    def _syncModelToView(self) -> None:
        self._treeModel.setRootNode(self._datasetPresenter.getContentsTree())

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._syncModelToView()
