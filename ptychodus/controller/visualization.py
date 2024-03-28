from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import Qt, QLineF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QStatusBar

from ptychodus.api.geometry import Box2D, Line2D, Point2D
from ptychodus.api.visualization import VisualizationProduct

from ..model.visualization import VisualizationCore
from ..view.image import (HistogramDialog, ImageItem, ImageItemEvents, ImageMouseTool,
                          LineCutDialog)
from .data import FileDialogFactory

logger = logging.getLogger(__name__)


class VisualizationController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, model: VisualizationCore, view: QGraphicsView, item: ImageItem,
                 statusBar: QStatusBar, fileDialogFactory: FileDialogFactory) -> None:
        self._model = model
        self._view = view
        self._item = ImageItem(ImageItemEvents(), statusBar)
        self._statusBar = statusBar
        self._fileDialogFactory = fileDialogFactory
        self._lineCutDialog = LineCutDialog.createInstance(view)
        self._histogramDialog = HistogramDialog.createInstance(view)

    @classmethod
    def createInstance(cls, model: VisualizationCore, view: QGraphicsView, statusBar: QStatusBar,
                       fileDialogFactory: FileDialogFactory) -> VisualizationController:
        itemEvents = ImageItemEvents()
        item = ImageItem(itemEvents, statusBar)
        controller = cls(model, view, item, statusBar, fileDialogFactory)

        scene = QGraphicsScene()
        scene.addItem(item)
        view.setScene(scene)

        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # TODO itemEvents.rectangleFinished.connect(widget.rectangleFinished)
        # TODO itemEvents.lineCutFinished.connect(widget.lineCutFinished)

        return controller

    def setProduct(self, product: VisualizationProduct) -> None:
        self._item.setProduct(product)

    def setMouseTool(self, mouseTool: ImageMouseTool) -> None:
        self._item.setMouseTool(mouseTool)

    def saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=VisualizationController.MIME_TYPES)

        if filePath:
            pixmap = self._item.pixmap()
            pixmap.save(str(filePath))

    def analyzeLineCut(self, line: QLineF) -> None:
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

    def analyzeRegion(self, rect: QRectF) -> None:
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
        values = numpy.arange(kde.valueLower, kde.valueUpper, 1000)

        ax = self._histogramDialog.axes
        ax.clear()
        ax.plot(values, kde.kde(values), '.-', linewidth=1.5)
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel(valueLabel)
        ax.grid(True)
        self._histogramDialog.figureCanvas.draw()

        # FIXME display rect in histogram dialog

        self._histogramDialog.open()

        print(rect)  # TODO use for crop

    def zoomToFit(self) -> None:
        self._item.setPos(0, 0)
        scene = self._view.scene()
        boundingRect = scene.itemsBoundingRect()
        scene.setSceneRect(boundingRect)
        self._view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
