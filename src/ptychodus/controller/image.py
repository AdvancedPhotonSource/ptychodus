from __future__ import annotations
from decimal import Decimal
import logging

import numpy

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QButtonGroup, QDialog, QStatusBar

from ptychodus.api.geometry import Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.visualization import NumberArrayType

from ..model.visualization import VisualizationEngine
from ..view.image import (
    ImageDataRangeGroupBox,
    ImageDisplayRangeDialog,
    ImageRendererGroupBox,
    ImageToolsGroupBox,
    ImageView,
    ImageWidget,
)
from ..view.visualization import ImageMouseTool
from .data import FileDialogFactory
from .visualization import VisualizationController

logger = logging.getLogger(__name__)


class ImageToolsController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(
        self,
        view: ImageToolsGroupBox,
        visualizationController: VisualizationController,
        mouseToolButtonGroup: QButtonGroup,
    ) -> None:
        self._view = view
        self._visualizationController = visualizationController
        self._mouseToolButtonGroup = mouseToolButtonGroup

    @classmethod
    def create_instance(
        cls, view: ImageToolsGroupBox, visualizationController: VisualizationController
    ) -> ImageToolsController:
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
    def create_instance(
        cls, engine: VisualizationEngine, view: ImageRendererGroupBox
    ) -> ImageRendererController:
        controller = cls(engine, view)

        view.rendererComboBox.setModel(controller._rendererModel)
        view.transformationComboBox.setModel(controller._transformationModel)
        view.variantComboBox.setModel(controller._variantModel)

        controller._sync_model_to_view()
        engine.add_observer(controller)

        view.rendererComboBox.textActivated.connect(engine.set_renderer)
        view.transformationComboBox.textActivated.connect(engine.set_transformation)
        view.variantComboBox.textActivated.connect(engine.set_variant)

        return controller

    def _sync_model_to_view(self) -> None:
        self._view.rendererComboBox.blockSignals(True)
        self._rendererModel.setStringList([name for name in self._engine.renderers()])
        self._view.rendererComboBox.setCurrentText(self._engine.get_renderer())
        self._view.rendererComboBox.blockSignals(False)

        self._view.transformationComboBox.blockSignals(True)
        self._transformationModel.setStringList([name for name in self._engine.transformations()])
        self._view.transformationComboBox.setCurrentText(self._engine.get_transformation())
        self._view.transformationComboBox.blockSignals(False)

        self._view.variantComboBox.blockSignals(True)
        self._variantModel.setStringList([name for name in self._engine.variants()])
        self._view.variantComboBox.setCurrentText(self._engine.get_variant())
        self._view.variantComboBox.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()


class ImageDataRangeController(Observer):
    def __init__(
        self,
        engine: VisualizationEngine,
        view: ImageDataRangeGroupBox,
        imageWidget: ImageWidget,
        displayRangeDialog: ImageDisplayRangeDialog,
        visualizationController: VisualizationController,
    ) -> None:
        self._engine = engine
        self._view = view
        self._imageWidget = imageWidget
        self._displayRangeDialog = displayRangeDialog
        self._visualizationController = visualizationController
        self._displayRangeIsLocked = True

    @classmethod
    def create_instance(
        cls,
        engine: VisualizationEngine,
        view: ImageDataRangeGroupBox,
        imageWidget: ImageWidget,
        visualizationController: VisualizationController,
    ) -> ImageDataRangeController:
        displayRangeDialog = ImageDisplayRangeDialog.create_instance(view)
        controller = cls(engine, view, imageWidget, displayRangeDialog, visualizationController)
        controller._sync_model_to_view()
        engine.add_observer(controller)

        view.minDisplayValueSlider.valueChanged.connect(
            lambda value: engine.set_min_display_value(float(value))
        )
        view.maxDisplayValueSlider.valueChanged.connect(
            lambda value: engine.set_max_display_value(float(value))
        )
        view.autoButton.clicked.connect(controller._autoDisplayRange)
        view.editButton.clicked.connect(displayRangeDialog.open)
        displayRangeDialog.finished.connect(controller._finishEditingDisplayRange)

        view.colorLegendButton.setCheckable(True)
        imageWidget.setColorLegendVisible(view.colorLegendButton.isChecked())
        view.colorLegendButton.toggled.connect(imageWidget.setColorLegendVisible)

        return controller

    def _autoDisplayRange(self) -> None:
        self._displayRangeIsLocked = False
        self._visualizationController.rerenderImage(autoscaleColorAxis=True)
        self._displayRangeIsLocked = True

    def _finishEditingDisplayRange(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            lower = float(self._displayRangeDialog.minValueLineEdit.getValue())
            upper = float(self._displayRangeDialog.maxValueLineEdit.getValue())

            self._displayRangeIsLocked = False
            self._engine.set_display_value_range(lower, upper)
            self._displayRangeIsLocked = True

    def _syncColorLegendToView(self) -> None:
        values = numpy.linspace(
            self._engine.get_min_display_value(), self._engine.get_max_display_value(), 1000
        )
        self._imageWidget.setColorLegendColors(
            values,
            self._engine.colorize(values),
            self._engine.is_renderer_cyclic(),
        )

    def _sync_model_to_view(self) -> None:
        minValue = Decimal(repr(self._engine.get_min_display_value()))
        maxValue = Decimal(repr(self._engine.get_max_display_value()))

        self._displayRangeDialog.minValueLineEdit.setValue(minValue)
        self._displayRangeDialog.maxValueLineEdit.setValue(maxValue)

        if self._displayRangeIsLocked:
            self._view.minDisplayValueSlider.setValue(minValue)
            self._view.maxDisplayValueSlider.setValue(maxValue)
        else:
            displayRangeLimits = Interval[Decimal](minValue, maxValue)
            self._view.minDisplayValueSlider.setValueAndRange(minValue, displayRangeLimits)
            self._view.maxDisplayValueSlider.setValueAndRange(maxValue, displayRangeLimits)

        self._syncColorLegendToView()

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()


class ImageController:
    def __init__(
        self,
        engine: VisualizationEngine,
        view: ImageView,
        visualizationController: VisualizationController,
    ) -> None:
        self._visualizationController = visualizationController
        self._toolsController = ImageToolsController.create_instance(
            view.imageRibbon.imageToolsGroupBox, visualizationController
        )
        self._rendererController = ImageRendererController.create_instance(
            engine, view.imageRibbon.colormapGroupBox
        )
        self._dataRangeController = ImageDataRangeController.create_instance(
            engine,
            view.imageRibbon.dataRangeGroupBox,
            view.imageWidget,
            visualizationController,
        )

    @classmethod
    def create_instance(
        cls,
        engine: VisualizationEngine,
        view: ImageView,
        statusBar: QStatusBar,
        fileDialogFactory: FileDialogFactory,
    ) -> ImageController:
        visualizationController = VisualizationController.create_instance(
            engine, view.imageWidget, statusBar, fileDialogFactory
        )
        return cls(engine, view, visualizationController)

    def set_array(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> None:
        self._visualizationController.setArray(array, pixelGeometry)

    def clear_array(self) -> None:
        self._visualizationController.clearArray()
