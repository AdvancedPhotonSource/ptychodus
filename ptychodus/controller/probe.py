from __future__ import annotations

from ..model import Observer, Observable, Probe, ProbePresenter
from ..view import ImageView, ProbeProbeView, ProbeInitializerView, ProbeZonePlateView, ProbeParametersView
from .data import FileDialogFactory
from .image import ImageController


class ProbeProbeController(Observer):
    def __init__(self, presenter: ProbePresenter, view: ProbeProbeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter,
                       view: ProbeProbeView) -> ProbeProbeController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.sizeSpinBox.valueChanged.connect(presenter.setProbeSize)
        view.sizeSpinBox.autoToggled.connect(presenter.setAutomaticProbeSizeEnabled)
        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setEnabled(False)
        view.diameterWidget.lengthChanged.connect(presenter.setProbeDiameterInMeters)

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
        self._view.diameterWidget.setLengthInMeters(self._presenter.getProbeDiameterInMeters())

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


class ProbeInitializerController(Observer):
    def __init__(self, presenter: ProbePresenter, view: ProbeInitializerView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter,
                       view: ProbeInitializerView) -> ProbeInitializerController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for initializer in presenter.getInitializerList():
            view.initializerComboBox.addItem(initializer)

        view.initializerComboBox.currentTextChanged.connect(presenter.setInitializer)
        view.initializeButton.clicked.connect(presenter.initializeProbe)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getInitializer())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeParametersController:
    def __init__(self, presenter: ProbePresenter, view: ProbeParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._probeController = ProbeProbeController.createInstance(presenter, view.probeView)
        self._zonePlateController = ProbeZonePlateController.createInstance(
            presenter, view.zonePlateView)
        self._initializerController = ProbeInitializerController.createInstance(
            presenter, view.initializerView)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeParametersView,
                       fileDialogFactory: FileDialogFactory) -> ProbeParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        return controller

    def openProbe(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._view, 'Open Probe', nameFilters=self._presenter.getOpenFileFilterList())

        if filePath:
            self._presenter.openProbe(filePath)

    def saveProbe(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Probe', nameFilters=self._presenter.getSaveFileFilterList())

        if filePath:
            self._presenter.saveProbe(filePath)


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
        view.imageRibbon.indexSpinBox.valueChanged.connect(controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._presenter.getProbeMode(index)
        self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        numberOfProbeModes = self._presenter.getNumberOfProbeModes()
        self._view.imageRibbon.indexSpinBox.setEnabled(numberOfProbeModes > 0)
        self._view.imageRibbon.indexSpinBox.setRange(0, numberOfProbeModes - 1)

        index = self._view.imageRibbon.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
