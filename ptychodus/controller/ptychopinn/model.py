from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...model.ptychopinn import PtychoPINNModelPresenter
from ...view.ptychopinn import PtychoPINNModelParametersView
from ..data import FileDialogFactory


class PtychoPINNModelParametersController(Observer):

    def __init__(self, presenter: PtychoPINNModelPresenter, view: PtychoPINNModelParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: PtychoPINNModelPresenter, view: PtychoPINNModelParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoPINNModelParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.modelStateLineEdit.editingFinished.connect(controller._syncModelStateFilePathToModel)
        view.modelStateBrowseButton.clicked.connect(controller._openModelState)
        view.gridSizeSpinBox.valueChanged.connect(presenter.setGridSize)
        view.nFiltersScaleSpinBox.valueChanged.connect(presenter.setNFiltersScale)
        view.nPhotonsLineEdit.editingFinished.connect(lambda: presenter.setNPhotons(view.nPhotonsLineEdit.value()))
        view.probeTrainableCheckBox.toggled.connect(presenter.setProbeTrainable)
        view.intensityScaleTrainableCheckBox.toggled.connect(presenter.setIntensityScaleTrainable)
        view.objectBigCheckBox.toggled.connect(presenter.setObjectBig)
        view.probeBigCheckBox.toggled.connect(presenter.setProbeBig)
        view.probeScaleLineEdit.editingFinished.connect(lambda: presenter.setProbeScale(view.probeScaleLineEdit.value()))
        view.probeMaskCheckBox.toggled.connect(presenter.setProbeMask)
        view.ampActivationLineEdit.editingFinished.connect(lambda: presenter.setAmpActivation(view.ampActivationLineEdit.text()))

        controller._syncModelToView()

        return controller

    def _syncModelStateFilePathToModel(self) -> None:
        self._presenter.setStateFilePath(Path(self._view.modelStateLineEdit.text()))

    def _openModelState(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Model State',
            nameFilters=self._presenter.getStateFileFilterList(),
            selectedNameFilter=self._presenter.getStateFileFilter())

        if filePath:
            self._presenter.setStateFilePath(filePath)

    def _syncModelToView(self) -> None:
        modelStateFilePath = self._presenter.getStateFilePath()

        if modelStateFilePath:
            self._view.modelStateLineEdit.setText(str(modelStateFilePath))
        else:
            self._view.modelStateLineEdit.clear()

        self._view.gridSizeSpinBox.setValue(self._presenter.getGridSize())
        self._view.nFiltersScaleSpinBox.setValue(self._presenter.getNFiltersScale())
        self._view.nPhotonsLineEdit.setValue(self._presenter.getNPhotons())
        self._view.probeTrainableCheckBox.setChecked(self._presenter.isProbeTrainable())
        self._view.intensityScaleTrainableCheckBox.setChecked(self._presenter.isIntensityScaleTrainable())
        self._view.objectBigCheckBox.setChecked(self._presenter.isObjectBig())
        self._view.probeBigCheckBox.setChecked(self._presenter.isProbeBig())
        self._view.probeScaleLineEdit.setValue(self._presenter.getProbeScale())
        self._view.probeMaskCheckBox.setChecked(self._presenter.isProbeMask())
        self._view.ampActivationLineEdit.setText(self._presenter.getAmpActivation())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
