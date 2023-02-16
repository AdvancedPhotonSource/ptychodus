from __future__ import annotations
from PyQt5.QtWidgets import QDialog, QListView

from ...api.observer import Observable, Observer
from ...model.data import DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter
from ...view import DataNavigationPage, DatasetView
from .dialogFactory import FileDialogFactory


class DatasetController(Observer):

    def __init__(self, inputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                 presenter: DiffractionDatasetPresenter, view: DataNavigationPage[DatasetView],
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._inputOutputPresenter = inputOutputPresenter
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, inputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                       presenter: DiffractionDatasetPresenter,
                       view: DataNavigationPage[DatasetView],
                       fileDialogFactory: FileDialogFactory) -> DatasetController:
        controller = cls(inputOutputPresenter, presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.backwardButton.clicked.connect(inputOutputPresenter.stopProcessingDiffractionPatterns)
        view.forwardButton.clicked.connect(controller._saveDiffractionFile)

        controller._syncModelToView()

        return controller

    def _saveDiffractionFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Data File',
            nameFilters=self._inputOutputPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._inputOutputPresenter.getSaveFileFilter())

        if filePath:
            self._inputOutputPresenter.saveDiffractionFile(filePath)

    def _syncModelToView(self) -> None:
        datasetLabel = self._presenter.getDatasetLabel()
        self._view.contentsView.setTitle(f'Dataset: {datasetLabel}')

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
