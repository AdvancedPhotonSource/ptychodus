from __future__ import annotations

from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QButtonGroup

import numpy

from ..api.observer import Observable, Observer
from ..model.image import ImagePresenter
from ..view.image import (ImageColorizerGroupBox, ImageDataRangeGroupBox, ImageMouseTool,
                          ImageToolsGroupBox, ImageView, ImageWidget)
from .data import FileDialogFactory


class ImageToolsController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, view: ImageToolsGroupBox, imageWidget: ImageWidget,
                 fileDialogFactory: FileDialogFactory, mouseToolButtonGroup: QButtonGroup) -> None:
        self._view = view
        self._imageWidget = imageWidget
        self._fileDialogFactory = fileDialogFactory
        self._mouseToolButtonGroup = mouseToolButtonGroup

    @classmethod
    def createInstance(cls, view: ImageToolsGroupBox, imageWidget: ImageWidget,
                       fileDialogFactory: FileDialogFactory) -> ImageToolsController:
        view.moveButton.setCheckable(True)
        view.measureButton.setCheckable(True)

        view.moveButton.setChecked(True)

        mouseToolButtonGroup = QButtonGroup()
        mouseToolButtonGroup.addButton(view.moveButton, ImageMouseTool.MOVE_TOOL.value)
        mouseToolButtonGroup.addButton(view.measureButton, ImageMouseTool.MEASURE_TOOL.value)

        controller = cls(view, imageWidget, fileDialogFactory, mouseToolButtonGroup)
        view.homeButton.clicked.connect(imageWidget.zoomToFit)
        view.saveButton.clicked.connect(controller._saveImage)
        mouseToolButtonGroup.idToggled.connect(controller._setMouseTool)
        return controller

    def _saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=ImageToolsController.MIME_TYPES)

        if filePath:
            pixmap = self._imageWidget.getPixmap()
            pixmap.save(str(filePath))

    def _setMouseTool(self, toolId: int, checked: bool) -> None:
        if checked:
            mouseTool = ImageMouseTool(toolId)
            self._imageWidget.setMouseTool(mouseTool)


class ImageColorizerController(Observer):

    def __init__(self, presenter: ImagePresenter, view: ImageColorizerGroupBox) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._colorizerModel = QStringListModel()
        self._scalarTransformModel = QStringListModel()
        self._variantModel = QStringListModel()

    @classmethod
    def createInstance(cls, presenter: ImagePresenter,
                       view: ImageColorizerGroupBox) -> ImageColorizerController:
        controller = cls(presenter, view)

        view.colorizerComboBox.setModel(controller._colorizerModel)
        view.scalarTransformComboBox.setModel(controller._scalarTransformModel)
        view.variantComboBox.setModel(controller._variantModel)

        controller._syncModelToView()
        presenter.addObserver(controller)

        view.colorizerComboBox.currentTextChanged.connect(presenter.setColorizerByName)
        view.scalarTransformComboBox.currentTextChanged.connect(
            presenter.setScalarTransformationByName)
        view.variantComboBox.currentTextChanged.connect(presenter.setVariantByName)

        return controller

    def _syncModelToView(self) -> None:
        self._view.colorizerComboBox.blockSignals(True)
        self._colorizerModel.setStringList(self._presenter.getColorizerNameList())
        self._view.colorizerComboBox.setCurrentText(self._presenter.getColorizerName())
        self._view.colorizerComboBox.blockSignals(False)

        self._view.scalarTransformComboBox.blockSignals(True)
        self._scalarTransformModel.setStringList(self._presenter.getScalarTransformationNameList())
        self._view.scalarTransformComboBox.setCurrentText(
            self._presenter.getScalarTransformationName())
        self._view.scalarTransformComboBox.blockSignals(False)

        self._view.variantComboBox.blockSignals(True)
        self._variantModel.setStringList(self._presenter.getVariantNameList())
        self._view.variantComboBox.setCurrentText(self._presenter.getVariantName())
        self._view.variantComboBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ImageDataRangeController(Observer):

    def __init__(self, presenter: ImagePresenter, view: ImageDataRangeGroupBox) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ImagePresenter, view: ImageDataRangeGroupBox,
                       imageWidget: ImageWidget) -> ImageDataRangeController:
        controller = cls(presenter, view)

        controller._syncModelToView()
        presenter.addObserver(controller)

        view.minDisplayValueSlider.valueChanged.connect(presenter.setMinDisplayValue)
        view.maxDisplayValueSlider.valueChanged.connect(presenter.setMaxDisplayValue)
        view.autoButton.clicked.connect(presenter.setDisplayRangeToDataRange)
        view.editButton.clicked.connect(controller._editDisplayRange)

        view.colorLegendButton.setCheckable(True)
        imageWidget.setColorLegendVisible(view.colorLegendButton.isChecked())
        view.colorLegendButton.toggled.connect(imageWidget.setColorLegendVisible)

        return controller

    def _editDisplayRange(self) -> None:
        if self._view.displayRangeDialog.exec_():
            minValue = self._view.displayRangeDialog.minValue()
            maxValue = self._view.displayRangeDialog.maxValue()
            self._presenter.setCustomDisplayRange(minValue, maxValue)

    def _syncModelToView(self) -> None:
        displayRangeLimits = self._presenter.getDisplayRangeLimits()

        self._view.minDisplayValueSlider.setValueAndRange(self._presenter.getMinDisplayValue(),
                                                          displayRangeLimits)
        self._view.maxDisplayValueSlider.setValueAndRange(self._presenter.getMaxDisplayValue(),
                                                          displayRangeLimits)
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
        self._toolsController = ImageToolsController.createInstance(
            view.imageRibbon.imageToolsGroupBox, view.imageWidget, fileDialogFactory)
        self._colorizerController = ImageColorizerController.createInstance(
            presenter, view.imageRibbon.colormapGroupBox)
        self._dataRangeController = ImageDataRangeController.createInstance(
            presenter, view.imageRibbon.dataRangeGroupBox, view.imageWidget)

    @classmethod
    def createInstance(cls, presenter: ImagePresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ImageController:
        controller = cls(presenter, view, fileDialogFactory)
        controller._syncModelToView()
        presenter.addObserver(controller)
        return controller

    def _syncModelToView(self) -> None:
        realImage = self._presenter.getImage()
        qpixmap = QPixmap()

        if realImage is not None and numpy.size(realImage) > 0:
            # NOTE .copy() ensures integerImage is not a view
            integerImage = numpy.multiply(realImage, 255).astype(numpy.uint8).copy()

            qimage = QImage(integerImage.data, integerImage.shape[1], integerImage.shape[0],
                            integerImage.strides[0], QImage.Format_RGBA8888)
            qpixmap = QPixmap.fromImage(qimage)

        self._view.imageWidget.setPixmap(qpixmap)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
