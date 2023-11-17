from __future__ import annotations

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator

from ...api.observer import Observable, Observer
from ...model.tike import TikePresenter
from ...view.tike import TikeBasicParametersView


class TikeBasicParametersController(Observer):

    def __init__(self, presenter: TikePresenter, view: TikeBasicParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: TikePresenter,
                       view: TikeBasicParametersView) -> TikeBasicParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numGpusLineEdit.editingFinished.connect(controller._syncNumGpusToModel)
        view.numGpusLineEdit.setValidator(
            QRegularExpressionValidator(QRegularExpression('[\\d,]+')))

        for model in presenter.getNoiseModelList():
            view.noiseModelComboBox.addItem(model)

        view.noiseModelComboBox.textActivated.connect(presenter.setNoiseModel)

        view.numBatchSpinBox.valueChanged.connect(presenter.setNumBatch)

        for method in presenter.getBatchMethodList():
            view.batchMethodComboBox.addItem(method)

        view.batchMethodComboBox.textActivated.connect(presenter.setBatchMethod)

        view.numIterSpinBox.valueChanged.connect(presenter.setNumIter)
        view.convergenceWindowSpinBox.valueChanged.connect(presenter.setConvergenceWindow)
        view.cgIterSpinBox.valueChanged.connect(presenter.setCgIter)

        view.alphaSlider.valueChanged.connect(presenter.setAlpha)
        view.stepLengthSlider.valueChanged.connect(presenter.setStepLength)

        for model in presenter.getLogLevelList():
            view.logLevelComboBox.addItem(model)

        view.logLevelComboBox.textActivated.connect(presenter.setLogLevel)

        controller._syncModelToView()

        return controller

    def _syncNumGpusToModel(self) -> None:
        self._presenter.setNumGpus(self._view.numGpusLineEdit.text())

    def _syncModelToView(self) -> None:
        self._view.numGpusLineEdit.setText(self._presenter.getNumGpus())
        self._view.noiseModelComboBox.setCurrentText(self._presenter.getNoiseModel())

        self._view.numBatchSpinBox.blockSignals(True)
        self._view.numBatchSpinBox.setRange(self._presenter.getNumBatchLimits().lower,
                                            self._presenter.getNumBatchLimits().upper)
        self._view.numBatchSpinBox.setValue(self._presenter.getNumBatch())
        self._view.numBatchSpinBox.blockSignals(False)

        self._view.batchMethodComboBox.setCurrentText(self._presenter.getBatchMethod())

        self._view.numIterSpinBox.blockSignals(True)
        self._view.numIterSpinBox.setRange(self._presenter.getNumIterLimits().lower,
                                           self._presenter.getNumIterLimits().upper)
        self._view.numIterSpinBox.setValue(self._presenter.getNumIter())
        self._view.numIterSpinBox.blockSignals(False)

        self._view.convergenceWindowSpinBox.blockSignals(True)
        self._view.convergenceWindowSpinBox.setRange(
            self._presenter.getConvergenceWindowLimits().lower,
            self._presenter.getConvergenceWindowLimits().upper)
        self._view.convergenceWindowSpinBox.setValue(self._presenter.getConvergenceWindow())
        self._view.convergenceWindowSpinBox.blockSignals(False)

        self._view.cgIterSpinBox.blockSignals(True)
        self._view.cgIterSpinBox.setRange(self._presenter.getCgIterLimits().lower,
                                          self._presenter.getCgIterLimits().upper)
        self._view.cgIterSpinBox.setValue(self._presenter.getCgIter())
        self._view.cgIterSpinBox.blockSignals(False)

        self._view.alphaSlider.setValueAndRange(self._presenter.getAlpha(),
                                                self._presenter.getAlphaLimits(),
                                                blockValueChangedSignal=True)
        self._view.stepLengthSlider.setValueAndRange(self._presenter.getStepLength(),
                                                     self._presenter.getStepLengthLimits(),
                                                     blockValueChangedSignal=True)
        self._view.logLevelComboBox.setCurrentText(self._presenter.getLogLevel())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
