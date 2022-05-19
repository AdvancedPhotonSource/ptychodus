from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QStandardItem, QStandardItemModel

import matplotlib
import numpy

from ..api.image import ScalarTransformation, ComplexToRealStrategy
from ..api.observer import Observable, Observer
from ..model import ImagePresenter
from ..view import ImageView, ImageColormapGroupBox, ImageFileGroupBox
from .data import FileDialogFactory


class ImageFileController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, view: ImageFileGroupBox, fileDialogFactory: FileDialogFactory) -> None:
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, view: ImageFileGroupBox,
                       fileDialogFactory: FileDialogFactory) -> ImageFileController:
        controller = cls(view, fileDialogFactory)
        view.saveButton.clicked.connect(controller._saveImage)
        return controller

    def _saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=ImageController.MIME_TYPES)

        if filePath:
            pixmap = self._view.imageWidget.getPixmap()
            pixmap.save(str(filePath))


class ImageColormapController:
    def __init__(self, presenter: ImagePresenter, view: ImageColormapGroupBox) -> None:
        self._presenter = presenter
        self._view = view
        self._complexToRealStrategyModel = QStringListModel()
        self._scalarTransformationModel = QStringListModel()
        self._colormapModel = QStringListModel()

    @classmethod
    def createInstance(cls, presenter: ImagePresenter,
                       view: ImageColormapGroupBox) -> ImageColormapController:
        controller = cls(presenter, view)

        view.complexComponentComboBox.setModel(controller._complexToRealStrategyModel)
        view.scalarTransformComboBox.setModel(controller._scalarTransformationModel)
        view.colormapComboBox.setModel(controller._colormapModel)

        controller._syncModelToView()
        presenter.addObserver(controller)

        view.scalarTransformComboBox.currentTextChanged.connect(presenter.setScalarTransformation)
        view.complexComponentComboBox.currentTextChanged.connect(
            presenter.setComplexToRealStrategy)
        view.colormapComboBox.currentTextChanged.connect(presenter.setColormap)

    def _syncModelToView(self) -> None:
        self._view.scalarTransformComboBox.blockSignals(True)
        self._scalarTransformationModel.setStringList(
            self._presenter.getScalarTransformationList())
        self._view.scalarTransformComboBox.setCurrentText(
            self._presenter.getScalarTransformation())
        self._view.scalarTransformComboBox.blockSignals(False)

        self._view.complexComponentComboBox.blockSignals(True)
        self._complexToRealStrategyModel.setStringList(
            self._presenter.getComplexToRealStrategyList())
        self._view.complexComponentComboBox.setCurrentText(
            self._presenter.getComplexToRealStrategy())
        self._view.complexComponentComboBox.blockSignals(False)
        self._view.complexComponentComboBox.setVisible(self._presenter.isComplexValued())

        self._view.colormapComboBox.blockSignals(True)
        self._colormapModel.setStringList(self._presenter.getColormapList())
        self._view.colormapComboBox.setCurrentText(self._presenter.getColormap())
        self._view.colormapComboBox.blockSignals(False)

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

        view.vminSlider.valueChanged.connect(presenter.setVMinValue)
        view.vmaxSlider.valueChanged.connect(presenter.setVMaxValue)
        view.autoButton.setCheckable(True)
        view.autoButton.toggled.connect(presenter.setAutomaticDataRangeEnabled)

        return controller

    def _syncModelToView(self) -> None:
        isAuto = self._presenter.isAutomaticDataRangeEnabled()

        self._view.vminSlider.setEnabled(not isAuto)
        # FIXME self._view.vminSlider.setValueAndRange(value, minimum, maximum) # FIXME typing
        self._view.vmaxSlider.setEnabled(not isAuto)
        # FIXME self._view.vmaxSlider.setValueAndRange(value, minimum, maximum) # FIXME typing
        self._view.autoButton.setChecked(isAuto)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ImageController(Observer):
    def __init__(self, presenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileController = ImageFileController.createInstance(
            view.imageRibbon.imageFileGroupBox, fileDialogFactory)
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

        if realImage is None:
            return

        integerImage = numpy.multiply(realImage, 255).astype(numpy.uint8)

        qimage = QImage(integerImage.data, integerImage.shape[1], integerImage.shape[0],
                        integerImage.strides[0], QImage.Format_RGBA8888)
        qpixmap = QPixmap.fromImage(qimage)
        self._view.imageWidget.setPixmap(qpixmap)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
