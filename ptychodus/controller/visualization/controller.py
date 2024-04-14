from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import Qt, QLineF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QStatusBar

from ptychodus.api.geometry import Box2D, Line2D, PixelGeometry, Point2D
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.visualization import NumberArrayType

from ...model.visualization import VisualizationEngine
from ...view.visualization import (HistogramDialog, ImageItem, ImageItemEvents, ImageMouseTool,
                                   LineCutDialog, VisualizationView)
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class VisualizationController(Observer):
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, engine: VisualizationEngine, view: VisualizationView, item: ImageItem,
                 statusBar: QStatusBar, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._engine = engine
        self._view = view
        self._item = item
        self._statusBar = statusBar
        self._fileDialogFactory = fileDialogFactory
        self._lineCutDialog = LineCutDialog.createInstance(view)
        self._histogramDialog = HistogramDialog.createInstance(view)

    @classmethod
    def createInstance(cls, engine: VisualizationEngine, view: VisualizationView,
                       statusBar: QStatusBar,
                       fileDialogFactory: FileDialogFactory) -> VisualizationController:
        itemEvents = ImageItemEvents()
        item = ImageItem(itemEvents, statusBar)
        controller = cls(engine, view, item, statusBar, fileDialogFactory)
        engine.addObserver(controller)

        itemEvents.lineCutFinished.connect(controller._analyzeLineCut)
        itemEvents.rectangleFinished.connect(controller._analyzeRegion)

        scene = QGraphicsScene()
        scene.addItem(item)
        view.setScene(scene)

        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        return controller

    def setArray(self,
                 array: NumberArrayType,
                 pixelGeometry: PixelGeometry,
                 *,
                 autoscaleColorAxis: bool = False) -> None:
        try:
            product = self._engine.render(array,
                                          pixelGeometry,
                                          autoscaleColorAxis=autoscaleColorAxis)
        except ValueError as err:
            logger.exception(err)
            ExceptionDialog.showException('Renderer', err)
        else:
            self._item.setProduct(product)

    def clearArray(self) -> None:
        self._item.clearProduct()

    def setMouseTool(self, mouseTool: ImageMouseTool) -> None:
        self._item.setMouseTool(mouseTool)

    def saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=VisualizationController.MIME_TYPES)

        if filePath:
            pixmap = self._item.pixmap()
            pixmap.save(str(filePath))

    def _analyzeLineCut(self, line: QLineF) -> None:
        p1 = Point2D(line.x1(), line.y1())
        p2 = Point2D(line.x2(), line.y2())
        line2D = Line2D(p1, p2)

        product = self._item.getProduct()

        if product is None:
            logger.warning('No visualization product!')
            return

        valueLabel = product.getValueLabel()
        lineCut = product.getLineCut(line2D)

        ax = self._lineCutDialog.axes
        ax.clear()
        ax.plot(lineCut.distanceInMeters, lineCut.value, '.-', linewidth=1.5)
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel(valueLabel)
        ax.grid(True)
        self._lineCutDialog.figureCanvas.draw()
        self._lineCutDialog.open()

    def _analyzeRegion(self, rect: QRectF) -> None:
        box = Box2D(
            x=rect.x(),
            y=rect.y(),
            width=rect.width(),
            height=rect.height(),
        )

        product = self._item.getProduct()

        if product is None:
            logger.warning('No visualization product!')
            return

        valueLabel = product.getValueLabel()
        kde = product.estimateKernelDensity(box)
        values = numpy.linspace(kde.valueLower, kde.valueUpper, 1000)

        ax = self._histogramDialog.axes
        ax.clear()
        ax.plot(values, kde.kde(values), '.-', linewidth=1.5)
        ax.set_xlabel(valueLabel)
        ax.set_ylabel('Density')
        ax.grid(True)
        self._histogramDialog.figureCanvas.draw()

        rectangleView = self._histogramDialog.rectangleView
        rectCenter = rect.center()
        rectangleView.centerXLineEdit.setText(f'{rectCenter.x():.1f}')
        rectangleView.centerYLineEdit.setText(f'{rectCenter.y():.1f}')
        rectangleView.widthLineEdit.setText(f'{rect.width():.1f}')
        rectangleView.heightLineEdit.setText(f'{rect.height():.1f}')

        self._histogramDialog.open()

        print(rect)  # TODO use for crop

    def zoomToFit(self) -> None:
        self._item.setPos(0, 0)
        scene = self._view.scene()
        boundingRect = scene.itemsBoundingRect()
        scene.setSceneRect(boundingRect)
        self._view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def rerenderImage(self, *, autoscaleColorAxis: bool = False) -> None:
        product = self._item.getProduct()

        if product is not None:
            self.setArray(product.getValues(),
                          product.getPixelGeometry(),
                          autoscaleColorAxis=autoscaleColorAxis)

    def update(self, observable: Observable) -> None:
        if observable is self._engine:
            self.rerenderImage()
