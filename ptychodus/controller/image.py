from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QStandardItem, QStandardItemModel

import matplotlib
import numpy

from ..api.image import ScalarTransformation, ComplexToRealStrategy
from ..api.observer import Observable, Observer
from ..model import ImagePresenter
from ..view import ImageDisplayRangeDialog, ImageColormapGroupBox, \
        ImageDataRangeGroupBox, ImageFileGroupBox, ImageView, ImageWidget
from .data import FileDialogFactory


class ImageFileController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, view: ImageFileGroupBox, imageWidget: ImageWidget,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._view = view
        self._imageWidget = imageWidget
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, view: ImageFileGroupBox, imageWidget: ImageWidget,
                       fileDialogFactory: FileDialogFactory) -> ImageFileController:
        controller = cls(view, imageWidget, fileDialogFactory)
        view.saveButton.clicked.connect(controller._saveImage)
        return controller

    def _saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=ImageFileController.MIME_TYPES)

        if filePath:
            pixmap = self._imageWidget.getPixmap()
            pixmap.save(str(filePath))


class ImageColormapController(Observer):

    def __init__(self, presenter: ImagePresenter, view: ImageColormapGroupBox) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._complexToRealStrategyModel = QStringListModel()
        self._scalarTransformationModel = QStringListModel()
        self._colormapModel = QStringListModel()

    @classmethod
    def createInstance(cls, presenter: ImagePresenter,
                       view: ImageColormapGroupBox) -> ImageColormapController:
        controller = cls(presenter, view)

        view.complexToRealStrategyComboBox.setModel(controller._complexToRealStrategyModel)
        view.scalarTransformationComboBox.setModel(controller._scalarTransformationModel)
        view.colormapComboBox.setModel(controller._colormapModel)

        controller._syncModelToView()
        presenter.addObserver(controller)

        view.complexToRealStrategyComboBox.currentTextChanged.connect(
            presenter.setComplexToRealStrategy)
        view.scalarTransformationComboBox.currentTextChanged.connect(
            presenter.setScalarTransformation)
        view.colormapComboBox.currentTextChanged.connect(presenter.setColormap)

        return controller

    def _syncModelToView(self) -> None:
        self._view.complexToRealStrategyComboBox.blockSignals(True)
        self._complexToRealStrategyModel.setStringList(
            self._presenter.getComplexToRealStrategyList())
        self._view.complexToRealStrategyComboBox.setCurrentText(
            self._presenter.getComplexToRealStrategy())
        self._view.complexToRealStrategyComboBox.blockSignals(False)
        self._view.complexToRealStrategyComboBox.setVisible(self._presenter.isComplexValued())

        self._view.scalarTransformationComboBox.blockSignals(True)
        self._scalarTransformationModel.setStringList(
            self._presenter.getScalarTransformationList())
        self._view.scalarTransformationComboBox.setCurrentText(
            self._presenter.getScalarTransformation())
        self._view.scalarTransformationComboBox.blockSignals(False)

        self._view.colormapComboBox.blockSignals(True)
        self._colormapModel.setStringList(self._presenter.getColormapList())
        self._view.colormapComboBox.setCurrentText(self._presenter.getColormap())
        self._view.colormapComboBox.blockSignals(False)
        self._view.colormapComboBox.setEnabled(self._presenter.isColormapEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ImageDataRangeController(Observer):

    def __init__(self, presenter: ImagePresenter, view: ImageDataRangeGroupBox) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ImagePresenter,
                       view: ImageDataRangeGroupBox) -> ImageDataRangeController:
        controller = cls(presenter, view)

        controller._syncModelToView()
        presenter.addObserver(controller)

        view.minDisplayValueSlider.valueChanged.connect(presenter.setMinDisplayValue)
        view.maxDisplayValueSlider.valueChanged.connect(presenter.setMaxDisplayValue)
        view.autoButton.clicked.connect(presenter.setDisplayRangeToDataRange)
        view.setButton.clicked.connect(controller._setCustomDisplayRange)

        return controller

    def _setCustomDisplayRange(self) -> None:
        if self._view.displayRangeDialog.exec_():
            minValue = self._view.displayRangeDialog.minValue()
            maxValue = self._view.displayRangeDialog.maxValue()
            self._presenter.setCustomDisplayRange(minValue, maxValue)

    def _syncModelToView(self) -> None:
        displayRangeLimits = self._presenter.getDisplayRangeLimits()

        self._view.minDisplayValueSlider.setValueAndRange(self._presenter.getMinDisplayValue(),
                                                          displayRangeLimits.lower,
                                                          displayRangeLimits.upper)
        self._view.maxDisplayValueSlider.setValueAndRange(self._presenter.getMaxDisplayValue(),
                                                          displayRangeLimits.lower,
                                                          displayRangeLimits.upper)
        self._view.displayRangeDialog.setMinAndMaxValues(displayRangeLimits.lower,
                                                         displayRangeLimits.upper)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ImageController(Observer):

    def __init__(self, presenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileController = ImageFileController.createInstance(
            view.imageRibbon.imageFileGroupBox, view.imageWidget, fileDialogFactory)
        self._colormapController = ImageColormapController.createInstance(
            presenter, view.imageRibbon.colormapGroupBox)
        self._dataRangeController = ImageDataRangeController.createInstance(
            presenter, view.imageRibbon.dataRangeGroupBox)

    @classmethod
    def createInstance(cls, presenter: ImagePresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory):
        controller = cls(presenter, view, fileDialogFactory)
        controller._syncModelToView()
        presenter.addObserver(controller)
        return controller

    def _syncModelToView(self) -> None:
        realImage = self._presenter.getImage()
        qpixmap = QPixmap()

        if realImage is not None:
            integerImage = numpy.multiply(realImage, 255).astype(numpy.uint8)

            qimage = QImage(integerImage.data, integerImage.shape[1], integerImage.shape[0],
                            integerImage.strides[0], QImage.Format_RGBA8888)
            qpixmap = QPixmap.fromImage(qimage)

        self._view.imageWidget.setPixmap(qpixmap)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
