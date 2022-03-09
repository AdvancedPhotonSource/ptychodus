from decimal import Decimal

from PyQt5.QtGui import QDoubleValidator, QStandardItem, QStandardItemModel

from ..model import Observer, Observable, ScanPointIO, ScanPresenter
from ..view import ScanParametersView, ScanPlotView
from .data_file import FileDialogFactory


class ScanParametersController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @staticmethod
    def createPositiveRealValidator() -> QDoubleValidator:
        validator = QDoubleValidator()
        validator.setBottom(0.)
        return validator

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanParametersView,
                       fileDialogFactory: FileDialogFactory):
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        for initializer in presenter.getScanSequenceList():
            view.initializerComboBox.addItem(initializer)

        view.initializerComboBox.currentTextChanged.connect(presenter.setCurrentScanSequence)
        view.numberOfScanPointsSpinBox.setEnabled(False)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentX)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentY)

        view.stepSizeXWidget.lengthChanged.connect(presenter.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(presenter.setStepSizeYInMeters)

        view.jitterRadiusLineEdit.setValidator(cls.createPositiveRealValidator())
        view.jitterRadiusLineEdit.editingFinished.connect(controller._syncJitterRadiusV2M)
        view.jitterRadiusLineEdit.setEnabled(False)

        transformModel = QStandardItemModel()

        for transform in presenter.getTransformXYList():
            row = QStandardItem(transform)
            transformModel.appendRow(row)

        view.transformComboBox.setModel(transformModel)
        view.transformComboBox.currentTextChanged.connect(presenter.setCurrentTransformXY)

        controller._syncModelToView()

        return controller

    def openScan(self) -> None:
        filePath = self._fileDialogFactory.getOpenFilePath(self._view, 'Open Scan',
                                                           ScanPointIO.FILE_FILTER)

        if filePath:
            self._presenter.openScan(filePath)

    def saveScan(self) -> None:
        filePath = self._fileDialogFactory.getSaveFilePath(self._view, 'Save Scan',
                                                           ScanPointIO.FILE_FILTER)

        if filePath:
            self._presenter.saveScan(filePath)

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getCurrentScanSequence())

        self._view.numberOfScanPointsSpinBox.blockSignals(True)
        self._view.numberOfScanPointsSpinBox.setRange(self._presenter.getMinNumberOfScanPoints(),
                                                      self._presenter.getMaxNumberOfScanPoints())
        self._view.numberOfScanPointsSpinBox.setValue(self._presenter.getNumberOfScanPoints())
        self._view.numberOfScanPointsSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getMinExtentX(),
                                           self._presenter.getMaxExtentX())
        self._view.extentXSpinBox.setValue(self._presenter.getExtentX())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getMinExtentY(),
                                           self._presenter.getMaxExtentY())
        self._view.extentYSpinBox.setValue(self._presenter.getExtentY())
        self._view.extentYSpinBox.blockSignals(False)

        self._view.stepSizeXWidget.setLengthInMeters(self._presenter.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._presenter.getStepSizeYInMeters())
        self._view.jitterRadiusLineEdit.setText(str(self._presenter.getJitterRadiusInPixels()))
        self._view.transformComboBox.setCurrentText(self._presenter.getCurrentTransformXY())

    def _syncJitterRadiusV2M(self) -> None:
        self._presenter.setJitterRadiusInPixels(Decimal(self._view.jitterRadiusLineEdit.text()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanPlotController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanPlotView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        scanPath = self._presenter.getScanPointList()

        x = [point.x for point in scanPath]
        y = [point.y for point in scanPath]

        self._view.axes.clear()
        self._view.axes.plot(x, y, '.-', linewidth=2.5)
        self._view.axes.invert_yaxis()
        self._view.axes.axis('equal')
        self._view.axes.grid(True)
        self._view.axes.set_xlabel('X [m]')
        self._view.axes.set_ylabel('Y [m]')
        self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
