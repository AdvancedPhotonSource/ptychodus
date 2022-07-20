from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QWidget

from ..model import Observable, Observer  # TODO, TikeAdaptiveMomentPresenter, TikeBackend, TikeObjectCorrectionPresenter, TikePositionCorrectionPresenter, TikePresenter, TikeProbeCorrectionPresenter
from ..view import TikeAdaptiveMomentView, TikeBasicParametersView, TikeObjectCorrectionView, \
        TikeParametersView, TikePositionCorrectionView, TikeProbeCorrectionView
from .reconstructor import ReconstructorViewControllerFactory


class TikeAdaptiveMomentController(Observer):

    def __init__(self, presenter: TikeAdaptiveMomentPresenter,
                 view: TikeAdaptiveMomentView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: TikeAdaptiveMomentPresenter,
                       view: TikeAdaptiveMomentView) -> TikeAdaptiveMomentController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setAdaptiveMomentEnabled)

        view.mdecaySlider.valueChanged.connect(presenter.setMDecay)
        view.vdecaySlider.valueChanged.connect(presenter.setVDecay)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isAdaptiveMomentEnabled())

        self._view.mdecaySlider.setValueAndRange(self._presenter.getMDecay(),
                                                 self._presenter.getMinMDecay(),
                                                 self._presenter.getMaxMDecay(),
                                                 blockValueChangedSignal=True)
        self._view.vdecaySlider.setValueAndRange(self._presenter.getVDecay(),
                                                 self._presenter.getMinVDecay(),
                                                 self._presenter.getMaxVDecay(),
                                                 blockValueChangedSignal=True)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class TikeProbeSupportController(Observer):

    def __init__(self, presenter: TikeProbeCorrectionPresenter,
                 view: TikeProbeSupportView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: TikeProbeCorrectionPresenter,
                       view: TikeProbeSupportView) -> TikeProbeSupportController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setFiniteProbeSupportEnabled)

        view.weightLineEdit.valueChanged.connect(presenter.setProbeSupportWeight)
        view.radiusSlider.valueChanged.connect(presenter.setProbeSupportRadius)
        view.degreeLineEdit.valueChanged.connect(presenter.setProbeSupportDegree)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isFiniteProbeSupportEnabled())

        self._view.weightLineEdit.setMinimum(self._presenter.getMinProbeSupportWeight())
        self._view.weightLineEdit.setValue(self._presenter.getProbeSupportWeight())

        self._view.radiusSlider.setValueAndRange(self._presenter.getProbeSupportRadius(),
                                                 self._presenter.getMinProbeSupportRadius(),
                                                 self._presenter.getMaxProbeSupportRadius(),
                                                 blockValueChangedSignal=True)

        self._view.degreeLineEdit.setMinimum(self._presenter.getMinProbeSupportDegree())
        self._view.degreeLineEdit.setValue(self._presenter.getProbeSupportDegree())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class TikeProbeCorrectionController(Observer):

    def __init__(self, presenter: TikeProbeCorrectionPresenter,
                 view: TikeProbeCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._probeSupportController = TikeProbeSupportController.createInstance(
            presenter, view.probeSupportView)
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView)

    @classmethod
    def createInstance(cls, presenter: TikeProbeCorrectionPresenter,
                       view: TikeProbeCorrectionView) -> TikeProbeCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setProbeCorrectionEnabled)

        view.sparsityConstraintSlider.valueChanged.connect(presenter.setSparsityConstraint)
        view.orthogonalityConstraintCheckBox.toggled.connect(
            presenter.setOrthogonalityConstraintEnabled)
        view.centeredIntensityConstraintCheckBox.toggled.connect(
            presenter.setCenteredIntensityConstraintEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isProbeCorrectionEnabled())

        self._view.sparsityConstraintSlider.setValueAndRange(
            self._presenter.getSparsityConstraint(),
            self._presenter.getMinSparsityConstraint(),
            self._presenter.getMaxSparsityConstraint(),
            blockValueChangedSignal=True)
        self._view.orthogonalityConstraintCheckBox.setChecked(
            self._presenter.isOrthogonalityConstraintEnabled())
        self._view.centeredIntensityConstraintCheckBox.setChecked(
            self._presenter.isCenteredIntensityConstraintEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class TikeObjectCorrectionController(Observer):

    def __init__(self, presenter: TikeObjectCorrectionPresenter,
                 view: TikeObjectCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView)

    @classmethod
    def createInstance(cls, presenter: TikeObjectCorrectionPresenter,
                       view: TikeObjectCorrectionView) -> TikeObjectCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setObjectCorrectionEnabled)
        view.positivityConstraintSlider.valueChanged.connect(presenter.setPositivityConstraint)
        view.smoothnessConstraintSlider.valueChanged.connect(presenter.setSmoothnessConstraint)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isObjectCorrectionEnabled())

        self._view.positivityConstraintSlider.setValueAndRange(
            self._presenter.getPositivityConstraint(),
            self._presenter.getMinPositivityConstraint(),
            self._presenter.getMaxPositivityConstraint(),
            blockValueChangedSignal=True)
        self._view.smoothnessConstraintSlider.setValueAndRange(
            self._presenter.getSmoothnessConstraint(),
            self._presenter.getMinSmoothnessConstraint(),
            self._presenter.getMaxSmoothnessConstraint(),
            blockValueChangedSignal=True)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class TikePositionCorrectionController(Observer):

    def __init__(self, presenter: TikePositionCorrectionPresenter,
                 view: TikePositionCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView)

    @classmethod
    def createInstance(cls, presenter: TikePositionCorrectionPresenter,
                       view: TikePositionCorrectionView) -> TikePositionCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setPositionCorrectionEnabled)

        view.positionRegularizationCheckBox.toggled.connect(
            presenter.setPositionRegularizationEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isPositionCorrectionEnabled())

        self._view.positionRegularizationCheckBox.setChecked(
            self._presenter.isPositionRegularizationEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


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

        for model in presenter.getNoiseModelList():
            view.noiseModelComboBox.addItem(model)

        view.useMpiCheckBox.toggled.connect(presenter.setMpiEnabled)

        view.numGpusLineEdit.editingFinished.connect(controller._syncNumGpusToModel)
        view.numGpusLineEdit.setValidator(QRegExpValidator(QRegExp('[\\d,]+')))

        view.noiseModelComboBox.currentTextChanged.connect(presenter.setNoiseModel)

        view.numProbeModesSpinBox.valueChanged.connect(presenter.setNumProbeModes)
        view.numBatchSpinBox.valueChanged.connect(presenter.setNumBatch)
        view.numIterSpinBox.valueChanged.connect(presenter.setNumIter)
        view.cgIterSpinBox.valueChanged.connect(presenter.setCgIter)

        view.alphaSlider.valueChanged.connect(presenter.setAlpha)
        view.stepLengthSlider.valueChanged.connect(presenter.setStepLength)

        controller._syncModelToView()

        return controller

    def _syncNumGpusToModel(self) -> None:
        self._presenter.setNumGpus(self._view.numGpusLineEdit.text())

    def _syncModelToView(self) -> None:
        self._view.useMpiCheckBox.setChecked(self._presenter.isMpiEnabled())
        self._view.numGpusLineEdit.setText(self._presenter.getNumGpus())
        self._view.noiseModelComboBox.setCurrentText(self._presenter.getNoiseModel())

        self._view.numProbeModesSpinBox.blockSignals(True)
        self._view.numProbeModesSpinBox.setRange(self._presenter.getMinNumProbeModes(),
                                                 self._presenter.getMaxNumProbeModes())
        self._view.numProbeModesSpinBox.setValue(self._presenter.getNumProbeModes())
        self._view.numProbeModesSpinBox.blockSignals(False)

        self._view.numBatchSpinBox.blockSignals(True)
        self._view.numBatchSpinBox.setRange(self._presenter.getMinNumBatch(),
                                            self._presenter.getMaxNumBatch())
        self._view.numBatchSpinBox.setValue(self._presenter.getNumBatch())
        self._view.numBatchSpinBox.blockSignals(False)

        self._view.numIterSpinBox.blockSignals(True)
        self._view.numIterSpinBox.setRange(self._presenter.getMinNumIter(),
                                           self._presenter.getMaxNumIter())
        self._view.numIterSpinBox.setValue(self._presenter.getNumIter())
        self._view.numIterSpinBox.blockSignals(False)

        self._view.cgIterSpinBox.blockSignals(True)
        self._view.cgIterSpinBox.setRange(self._presenter.getMinCgIter(),
                                          self._presenter.getMaxCgIter())
        self._view.cgIterSpinBox.setValue(self._presenter.getCgIter())
        self._view.cgIterSpinBox.blockSignals(False)

        self._view.alphaSlider.setValueAndRange(self._presenter.getAlpha(),
                                                self._presenter.getMinAlpha(),
                                                self._presenter.getMaxAlpha(),
                                                blockValueChangedSignal=True)
        self._view.stepLengthSlider.setValueAndRange(self._presenter.getStepLength(),
                                                     self._presenter.getMinStepLength(),
                                                     self._presenter.getMaxStepLength(),
                                                     blockValueChangedSignal=True)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class TikeParametersController:

    def __init__(self, model: TikeBackend, view: TikeParametersView) -> None:
        self._model = model
        self._view = view
        self._positionCorrectionController = TikePositionCorrectionController.createInstance(
            model.positionCorrectionPresenter, view.positionCorrectionView)
        self._probeCorrectionController = TikeProbeCorrectionController.createInstance(
            model.probeCorrectionPresenter, view.probeCorrectionView)
        self._objectCorrectionController = TikeObjectCorrectionController.createInstance(
            model.objectCorrectionPresenter, view.objectCorrectionView)
        self._basicParametersController = TikeBasicParametersController.createInstance(
            model.presenter, view.basicParametersView)

    @classmethod
    def createInstance(cls, model: TikeBackend,
                       view: TikeParametersView) -> TikeParametersController:
        controller = cls(model, view)
        return controller


class TikeViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: TikeBackend) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[TikeParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'Tike'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = None

        if reconstructorName == 'rpie':
            view = TikeParametersView.createInstance(showCgIter=False,
                                                     showAlpha=True,
                                                     showStepLength=False)
        elif reconstructorName == 'adam_grad':
            view = TikeParametersView.createInstance(showCgIter=False,
                                                     showAlpha=True,
                                                     showStepLength=True)
        elif reconstructorName == 'cgrad':
            view = TikeParametersView.createInstance(showCgIter=True,
                                                     showAlpha=False,
                                                     showStepLength=True)
        elif reconstructorName == 'lstsq_grad':
            view = TikeParametersView.createInstance(showCgIter=False,
                                                     showAlpha=False,
                                                     showStepLength=False)
        elif reconstructorName == 'dm':
            view = TikeParametersView.createInstance(showCgIter=False,
                                                     showAlpha=False,
                                                     showStepLength=False)
        else:
            view = TikeParametersView.createInstance(showCgIter=True,
                                                     showAlpha=True,
                                                     showStepLength=True)

        controller = TikeParametersController.createInstance(self._model, view)
        self._controllerList.append(controller)

        return view
