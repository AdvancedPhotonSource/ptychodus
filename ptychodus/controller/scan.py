from __future__ import annotations
from decimal import Decimal

from PyQt5.QtGui import QDoubleValidator

from ..model import Observer, Observable, Scan, ScanPresenter
from ..view import ScanScanView, ScanInitializerView, ScanParametersView, ScanPlotView
from .data import FileDialogFactory


class ScanScanController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanScanView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @staticmethod
    def createPositiveRealValidator() -> QDoubleValidator:
        validator = QDoubleValidator()
        validator.setBottom(0.)
        return validator

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanScanView) -> ScanScanController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfScanPointsSpinBox.setEnabled(False)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentX)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentY)

        view.stepSizeXWidget.lengthChanged.connect(presenter.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(presenter.setStepSizeYInMeters)

        view.jitterRadiusLineEdit.setValidator(cls.createPositiveRealValidator())
        view.jitterRadiusLineEdit.editingFinished.connect(controller._syncJitterRadiusV2M)
        view.jitterRadiusLineEdit.setEnabled(False)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfScanPointsSpinBox.blockSignals(True)
        self._view.numberOfScanPointsSpinBox.setRange(
            self._presenter.getNumberOfScanPointsLimits().lower,
            self._presenter.getNumberOfScanPointsLimits().upper)
        self._view.numberOfScanPointsSpinBox.setValue(self._presenter.getNumberOfScanPoints())
        self._view.numberOfScanPointsSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getExtentXLimits().lower,
                                           self._presenter.getExtentXLimits().upper)
        self._view.extentXSpinBox.setValue(self._presenter.getExtentX())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getExtentYLimits().lower,
                                           self._presenter.getExtentYLimits().upper)
        self._view.extentYSpinBox.setValue(self._presenter.getExtentY())
        self._view.extentYSpinBox.blockSignals(False)

        self._view.stepSizeXWidget.setLengthInMeters(self._presenter.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._presenter.getStepSizeYInMeters())
        self._view.jitterRadiusLineEdit.setText(str(self._presenter.getJitterRadiusInPixels()))

    def _syncJitterRadiusV2M(self) -> None:
        self._presenter.setJitterRadiusInPixels(Decimal(self._view.jitterRadiusLineEdit.text()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanInitializerController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanInitializerView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanInitializerView) -> ScanInitializerController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for initializer in presenter.getInitializerList():
            view.initializerComboBox.addItem(initializer)

        view.initializerComboBox.currentTextChanged.connect(presenter.setInitializer)
        view.initializeButton.clicked.connect(presenter.initializeScan)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getInitializer())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanTransformController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanTransformView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for transform in presenter.getTransformList():
            view.transformComboBox.addItem(transform)

        view.transformComboBox.currentTextChanged.connect(presenter.setTransform)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.transformComboBox.setCurrentText(self._presenter.getTransform())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanParametersController:
    def __init__(self, presenter: ScanPresenter, view: ScanParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._scanController = ScanScanController.createInstance(presenter, view.scanView)
        self._initializerController = ScanInitializerController.createInstance(
            presenter, view.initializerView)
        self._transformController = ScanTransformController.createInstance(
            presenter, view.transformView)

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanParametersView,
                       fileDialogFactory: FileDialogFactory) -> ScanParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        return controller

    def openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view, 'Open Scan', nameFilters=self._presenter.getOpenFileFilterList())

        if filePath:
            self._presenter.openScan(filePath, nameFilter)

    def saveScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Scan', nameFilters=self._presenter.getSaveFileFilterList())

        if filePath:
            self._presenter.saveScan(filePath, nameFilter)


class ScanPlotController(Observer):
    def __init__(self, presenter: ScanPresenter, view: ScanPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanPlotView) -> ScanPlotController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        scanPath = self._presenter.getScanPointList()

        x = [point.x for point in scanPath]
        y = [point.y for point in scanPath]

        self._view.axes.clear()
        self._view.axes.plot(x, y, '.-', linewidth=1.5)
        self._view.axes.invert_yaxis()
        self._view.axes.axis('equal')
        self._view.axes.grid(True)
        self._view.axes.set_xlabel('X [m]')
        self._view.axes.set_ylabel('Y [m]')
        self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
