from __future__ import annotations

from ..model import Observer, Observable, ImagePresenter, Probe, ProbePresenter
from ..view import (ImageView, ProbeView, ProbeInitializerView, ProbeParametersView,
                    ProbeSuperGaussianView, ProbeZonePlateView)
from .data import FileDialogFactory
from .image import ImageController


class ProbeController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeView) -> ProbeController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.sizeSpinBox.valueChanged.connect(presenter.setProbeSize)
        view.sizeSpinBox.autoToggled.connect(presenter.setAutomaticProbeSizeEnabled)
        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.sizeSpinBox.setAutomatic(self._presenter.isAutomaticProbeSizeEnabled())
        self._view.sizeSpinBox.setValueAndRange(self._presenter.getProbeSize(),
                                                self._presenter.getProbeMinSize(),
                                                self._presenter.getProbeMaxSize())
        self._view.energyWidget.setEnergyInElectronVolts(
            self._presenter.getProbeEnergyInElectronVolts())
        self._view.wavelengthWidget.setLengthInMeters(self._presenter.getProbeWavelengthInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeSuperGaussianController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeSuperGaussianView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter,
                       view: ProbeSuperGaussianView) -> ProbeSuperGaussianController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.annularRadiusWidget.lengthChanged.connect(
            presenter.setSuperGaussianAnnularRadiusInMeters)
        view.probeWidthWidget.lengthChanged.connect(presenter.setSuperGaussianProbeWidthInMeters)
        view.orderParameterWidget.valueChanged.connect(presenter.setSuperGaussianOrderParameter)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.annularRadiusWidget.setLengthInMeters(
            self._presenter.getSuperGaussianAnnularRadiusInMeters())
        self._view.probeWidthWidget.setLengthInMeters(
            self._presenter.getSuperGaussianProbeWidthInMeters())
        self._view.orderParameterWidget.setValue(self._presenter.getSuperGaussianOrderParameter())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeZonePlateController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeZonePlateView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter,
                       view: ProbeZonePlateView) -> ProbeZonePlateController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.zonePlateRadiusWidget.lengthChanged.connect(presenter.setZonePlateRadiusInMeters)
        view.outermostZoneWidthWidget.lengthChanged.connect(
            presenter.setOutermostZoneWidthInMeters)
        view.beamstopDiameterWidget.lengthChanged.connect(presenter.setBeamstopDiameterInMeters)
        view.defocusDistanceWidget.lengthChanged.connect(presenter.setDefocusDistanceInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.zonePlateRadiusWidget.setLengthInMeters(
            self._presenter.getZonePlateRadiusInMeters())
        self._view.outermostZoneWidthWidget.setLengthInMeters(
            self._presenter.getOutermostZoneWidthInMeters())
        self._view.beamstopDiameterWidget.setLengthInMeters(
            self._presenter.getBeamstopDiameterInMeters())
        self._view.defocusDistanceWidget.setLengthInMeters(
            self._presenter.getDefocusDistanceInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeParametersController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._probeController = ProbeController.createInstance(presenter, view.probeView)
        self._superGaussianController = ProbeSuperGaussianController.createInstance(
            presenter, view.superGaussianView)
        self._zonePlateController = ProbeZonePlateController.createInstance(
            presenter, view.zonePlateView)
        self._initializerGroupBoxes = [view.superGaussianView, view.zonePlateView]

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeParametersView,
                       fileDialogFactory: FileDialogFactory) -> ProbeParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        for initializer in presenter.getInitializerList():
            view.initializerView.initializerComboBox.addItem(initializer)

        view.initializerView.initializerComboBox.currentTextChanged.connect(
            presenter.setInitializer)
        view.initializerView.initializeButton.clicked.connect(presenter.initializeProbe)

        controller._syncModelToView()

        return controller

    def openProbe(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Probe',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openProbe(filePath, nameFilter)

    def saveProbe(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Probe',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveProbe(filePath, nameFilter)

    def _syncModelToView(self) -> None:
        initializer = self._presenter.getInitializer()

        self._view.initializerView.initializerComboBox.setCurrentText(initializer)

        for groupBox in self._initializerGroupBoxes:
            groupBox.setVisible(groupBox.title() == initializer)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeImageController(Observer):

    def __init__(self, presenter: ProbePresenter, imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, imagePresenter: ImagePresenter,
                       view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeImageController:
        controller = cls(presenter, imagePresenter, view, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setTitle('Probe Mode')
        view.imageRibbon.indexGroupBox.indexSpinBox.valueChanged.connect(
            controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._presenter.getProbeMode(index)
        self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        numberOfProbeModes = self._presenter.getNumberOfProbeModes()
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setEnabled(numberOfProbeModes > 0)
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setRange(0, numberOfProbeModes - 1)

        index = self._view.imageRibbon.indexGroupBox.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
