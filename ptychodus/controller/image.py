from __future__ import annotations
from decimal import Decimal
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QButtonGroup, QDialog, QStatusBar

from ptychodus.api.geometry import Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.visualization import NumberArrayType

from ..model.visualization import VisualizationEngine
from ..view.image import (ImageDataRangeGroupBox, ImageDisplayRangeDialog, ImageRendererGroupBox,
                          ImageToolsGroupBox, ImageView, ImageWidget)
from ..view.visualization import ImageMouseTool
from .data import FileDialogFactory
from .visualization import VisualizationController

logger = logging.getLogger(__name__)


class ImageToolsController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, view: ImageToolsGroupBox, visualizationController: VisualizationController,
                 mouseToolButtonGroup: QButtonGroup) -> None:
        self._view = view
        self._visualizationController = visualizationController
        self._mouseToolButtonGroup = mouseToolButtonGroup

    @classmethod
    def createInstance(cls, view: ImageToolsGroupBox,
                       visualizationController: VisualizationController) -> ImageToolsController:
        view.moveButton.setCheckable(True)
        view.moveButton.setChecked(True)
        view.rulerButton.setCheckable(True)
        view.rectangleButton.setCheckable(True)
        view.lineCutButton.setCheckable(True)

        mouseToolButtonGroup = QButtonGroup()
        mouseToolButtonGroup.addButton(view.moveButton, ImageMouseTool.MOVE_TOOL.value)
        mouseToolButtonGroup.addButton(view.rulerButton, ImageMouseTool.RULER_TOOL.value)
        mouseToolButtonGroup.addButton(view.rectangleButton, ImageMouseTool.RECTANGLE_TOOL.value)
        mouseToolButtonGroup.addButton(view.lineCutButton, ImageMouseTool.LINE_CUT_TOOL.value)

        controller = cls(view, visualizationController, mouseToolButtonGroup)
        view.homeButton.clicked.connect(visualizationController.zoomToFit)
        view.saveButton.clicked.connect(visualizationController.saveImage)
        mouseToolButtonGroup.idToggled.connect(controller._setMouseTool)
        return controller

    def _setMouseTool(self, toolId: int, checked: bool) -> None:
        if checked:
            mouseTool = ImageMouseTool(toolId)
            self._visualizationController.setMouseTool(mouseTool)


class ImageRendererController(Observer):

    def __init__(self, engine: VisualizationEngine, view: ImageRendererGroupBox) -> None:
        super().__init__()
        self._engine = engine
        self._view = view
        self._rendererModel = QStringListModel()
        self._transformationModel = QStringListModel()
        self._variantModel = QStringListModel()

    @classmethod
    def createInstance(cls, engine: VisualizationEngine,
                       view: ImageRendererGroupBox) -> ImageRendererController:
        controller = cls(engine, view)

        view.rendererComboBox.setModel(controller._rendererModel)
        view.transformationComboBox.setModel(controller._transformationModel)
        view.variantComboBox.setModel(controller._variantModel)

        controller._syncModelToView()
        engine.addObserver(controller)

        view.rendererComboBox.textActivated.connect(engine.setRenderer)
        view.transformationComboBox.textActivated.connect(engine.setTransformation)
        view.variantComboBox.textActivated.connect(engine.setVariant)

        return controller

    def _syncModelToView(self) -> None:
        self._view.rendererComboBox.blockSignals(True)
        self._rendererModel.setStringList([name for name in self._engine.renderers()])
        self._view.rendererComboBox.setCurrentText(self._engine.getRenderer())
        self._view.rendererComboBox.blockSignals(False)

        self._view.transformationComboBox.blockSignals(True)
        self._transformationModel.setStringList([name for name in self._engine.transformations()])
        self._view.transformationComboBox.setCurrentText(self._engine.getTransformation())
        self._view.transformationComboBox.blockSignals(False)

        self._view.variantComboBox.blockSignals(True)
        self._variantModel.setStringList([name for name in self._engine.variants()])
        self._view.variantComboBox.setCurrentText(self._engine.getVariant())
        self._view.variantComboBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._syncModelToView()


class ImageDataRangeController(Observer):

    def __init__(self, engine: VisualizationEngine, view: ImageDataRangeGroupBox,
                 imageWidget: ImageWidget, displayRangeDialog: ImageDisplayRangeDialog,
                 visualizationController: VisualizationController) -> None:
        self._engine = engine
        self._view = view
        self._imageWidget = imageWidget
        self._displayRangeDialog = displayRangeDialog
        self._visualizationController = visualizationController

    @classmethod
    def createInstance(
            cls, engine: VisualizationEngine, view: ImageDataRangeGroupBox,
            imageWidget: ImageWidget,
            visualizationController: VisualizationController) -> ImageDataRangeController:
        displayRangeDialog = ImageDisplayRangeDialog.createInstance(view)
        controller = cls(engine, view, imageWidget, displayRangeDialog, visualizationController)
        controller._syncModelToView()
        engine.addObserver(controller)

        view.minDisplayValueSlider.valueChanged.connect(
            lambda value: engine.setMinDisplayValue(float(value)))
        view.maxDisplayValueSlider.valueChanged.connect(
            lambda value: engine.setMaxDisplayValue(float(value)))
        view.autoButton.clicked.connect(controller._autoDisplayRange)
        view.editButton.clicked.connect(displayRangeDialog.open)
        displayRangeDialog.finished.connect(controller._finishEditingDisplayRange)

        view.colorLegendButton.setCheckable(True)
        imageWidget.setColorLegendVisible(view.colorLegendButton.isChecked())
        view.colorLegendButton.toggled.connect(imageWidget.setColorLegendVisible)

        return controller

    def _autoDisplayRange(self) -> None:
        self._visualizationController.rerenderImage(autoscaleColorAxis=True)

    def _finishEditingDisplayRange(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            lower = float(self._displayRangeDialog.minValueLineEdit.getValue())
            upper = float(self._displayRangeDialog.maxValueLineEdit.getValue())
            self._engine.setDisplayValueRange(lower, upper)

    def _syncModelToView(self) -> None:
        print('syncModelToView') # FIXME
        minDisplayValue = Decimal(repr(self._engine.getMinDisplayValue()))
        maxDisplayValue = Decimal(repr(self._engine.getMaxDisplayValue()))

        lowerLimit = min(self._displayRangeDialog.minValueLineEdit.getValue(), minDisplayValue)
        upperLimit = max(self._displayRangeDialog.maxValueLineEdit.getValue(), maxDisplayValue)
        displayRangeLimits = Interval[Decimal](lowerLimit, upperLimit)

        self._view.minDisplayValueSlider.setValueAndRange(minDisplayValue, displayRangeLimits)
        self._view.maxDisplayValueSlider.setValueAndRange(maxDisplayValue, displayRangeLimits)
        self._displayRangeDialog.minValueLineEdit.setValue(lowerLimit)
        self._displayRangeDialog.maxValueLineEdit.setValue(upperLimit)
        self._imageWidget.setColorLegendRange(float(minDisplayValue), float(maxDisplayValue))

        # FIXME xArray = numpy.linspace(0., 1., 256)
        # FIXME rgbaArray = self._engine.getColorSamples(xArray)
        # FIXME self._imageWidget.setColorLegendColors(xArray, rgbaArray, self._engine.isRendererCyclic())

    def update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._syncModelToView()


class ImageController:

    def __init__(self, engine: VisualizationEngine, view: ImageView,
                 visualizationController: VisualizationController) -> None:
        self._visualizationController = visualizationController
        self._toolsController = ImageToolsController.createInstance(
            view.imageRibbon.imageToolsGroupBox, visualizationController)
        self._rendererController = ImageRendererController.createInstance(
            engine, view.imageRibbon.colormapGroupBox)
        self._dataRangeController = ImageDataRangeController.createInstance(
            engine, view.imageRibbon.dataRangeGroupBox, view.imageWidget, visualizationController)

    @classmethod
    def createInstance(cls, engine: VisualizationEngine, view: ImageView, statusBar: QStatusBar,
                       fileDialogFactory: FileDialogFactory) -> ImageController:
        visualizationController = VisualizationController.createInstance(
            engine, view.imageWidget, statusBar, fileDialogFactory)
        return cls(engine, view, visualizationController)

    def setArray(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> None:
        self._visualizationController.setArray(array, pixelGeometry)

    def clearArray(self) -> None:
        self._visualizationController.clearArray()
