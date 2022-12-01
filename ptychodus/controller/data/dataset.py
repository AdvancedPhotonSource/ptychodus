from __future__ import annotations
from PyQt5.QtWidgets import QDialog, QListView

from ...api.observer import Observable, Observer
from ...model import DiffractionDatasetPresenter
from ...view import DataNavigationPage, DatasetView
from .dialogFactory import FileDialogFactory


class DatasetController(Observer):

    def __init__(self, presenter: DiffractionDatasetPresenter,
                 view: DataNavigationPage[DatasetView],
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: DiffractionDatasetPresenter,
                       view: DataNavigationPage[DatasetView],
                       fileDialogFactory: FileDialogFactory) -> DatasetController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.backwardButton.clicked.connect(presenter.stopProcessingDiffractionPatterns)
        view.forwardButton.clicked.connect(controller._saveDiffractionFile)

        controller._syncModelToView()

        return controller

    def _saveDiffractionFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Data File',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveDiffractionFile(filePath)

    def _syncModelToView(self) -> None:
        datasetLabel = self._presenter.getDatasetLabel()
        self._view.contentsView.setTitle(f'Dataset: {datasetLabel}')

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
