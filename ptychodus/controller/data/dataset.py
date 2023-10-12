from __future__ import annotations

from ...model.data import DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter
from ...view.data import DataNavigationPage, DatasetView
from .dialogFactory import FileDialogFactory


class DatasetController:

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

        view.backwardButton.clicked.connect(inputOutputPresenter.stopAssemblingDiffractionPatterns)
        view.forwardButton.clicked.connect(controller._saveDiffractionFile)

        return controller

    def _saveDiffractionFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Data File',
            nameFilters=self._inputOutputPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._inputOutputPresenter.getSaveFileFilter())

        if filePath:
            self._inputOutputPresenter.saveDiffractionFile(filePath)
