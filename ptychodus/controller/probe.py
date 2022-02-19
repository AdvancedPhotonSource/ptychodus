from pathlib import Path

from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QFileDialog

from ..model import Observer, Observable, ProbeIO, ProbePresenter
from ..view import ImageView, ProbeProbeView, ProbeInitializerView, ProbeZonePlateView, ProbeParametersView

from .image import ImageController


class ProbeProbeController(Observer):
    def __init__(self, presenter: ProbePresenter, view: ProbeProbeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeProbeView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.shapeSpinBox.valueChanged.connect(presenter.setProbeShape)
        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setEnabled(False)
        view.diameterWidget.lengthChanged.connect(presenter.setProbeDiameterInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.shapeSpinBox.blockSignals(True)
        self._view.shapeSpinBox.setRange(self._presenter.getProbeMinShape(),
                                         self._presenter.getProbeMaxShape())
        self._view.shapeSpinBox.setValue(self._presenter.getProbeShape())
        self._view.shapeSpinBox.blockSignals(False)

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
    def createInstance(cls, presenter: ProbePresenter, view: ProbeZonePlateView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.zonePlateRadiusWidget.lengthChanged.connect(presenter.setZonePlateRadiusInMeters)
        view.outermostZoneWidthWidget.lengthChanged.connect(
            presenter.setOutermostZoneWidthInMeters)
        view.beamstopDiameterWidget.lengthChanged.connect(presenter.setBeamstopDiameterInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.zonePlateRadiusWidget.setLengthInMeters(
            self._presenter.getZonePlateRadiusInMeters())
        self._view.outermostZoneWidthWidget.setLengthInMeters(
            self._presenter.getOutermostZoneWidthInMeters())
        self._view.beamstopDiameterWidget.setLengthInMeters(
            self._presenter.getBeamstopDiameterInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeInitializerController(Observer):
    def __init__(self, presenter: ProbePresenter, view: ProbeInitializerView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._initComboBoxModel = QStandardItemModel()

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeInitializerView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for initializer in presenter.getInitializerList():
            view.initializerComboBox.addItem(initializer)

        view.initializerComboBox.currentTextChanged.connect(presenter.setCurrentInitializer)
        view.initializeButton.clicked.connect(presenter.initializeProbe)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getCurrentInitializer())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeParametersController:
    def __init__(self, presenter: ProbePresenter, view: ProbeParametersView) -> None:
        self._presenter = presenter
        self._view = view
        self._probeController = ProbeProbeController.createInstance(presenter, view.probeView)
        self._zonePlateController = ProbeZonePlateController.createInstance(
            presenter, view.zonePlateView)
        self._initializerController = ProbeInitializerController.createInstance(
            presenter, view.initializerView)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeParametersView):
        controller = cls(presenter, view)
        return controller

    def openProbe(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(self._view, 'Open Probe', str(Path.home()),
                                                  ProbeIO.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.openProbe(filePath)

    def saveProbe(self) -> None:
        fileName, _ = QFileDialog.getSaveFileName(self._view, 'Save Probe', str(Path.home()),
                                                  ProbeIO.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.saveProbe(filePath)


class ProbeImageController(Observer):
    def __init__(self, presenter: ProbePresenter, view: ImageView) -> None:
        super().__init__()
        self._presenter = presenter
        self._image_controller = ImageController.createInstance(view)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ImageView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.frameGroupBox.setVisible(False)
        return controller

    def _syncModelToView(self) -> None:
        estimate = self._presenter.getProbe()
        self._image_controller.renderImageData(estimate)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
