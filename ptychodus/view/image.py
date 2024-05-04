from __future__ import annotations
from collections.abc import Iterator

import numpy

from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QSize
from PyQt5.QtGui import QColor, QConicalGradient, QIcon, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                             QHBoxLayout, QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
                             QWidget)

from ptychodus.api.visualization import RealArrayType

from .visualization import VisualizationView
from .widgets import BottomTitledGroupBox, DecimalLineEdit, DecimalSlider


class ImageDisplayRangeDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.buttonBox = QDialogButtonBox()
        self.minValueLineEdit = DecimalLineEdit.createInstance()
        self.maxValueLineEdit = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageDisplayRangeDialog:
        dialog = cls(parent)
        dialog.setWindowTitle('Set Display Range')
        dialog.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        dialog.buttonBox.accepted.connect(dialog.accept)
        dialog.buttonBox.addButton(QDialogButtonBox.StandardButton.Cancel)
        dialog.buttonBox.rejected.connect(dialog.reject)

        layout = QFormLayout()
        layout.addRow('Minimum Displayed Value:', dialog.minValueLineEdit)
        layout.addRow('Maximum Displayed Value:', dialog.maxValueLineEdit)
        layout.addRow(dialog.buttonBox)
        dialog.setLayout(layout)

        return dialog


class ImageToolsGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Tools', parent)
        self.homeButton = QToolButton()
        self.saveButton = QToolButton()
        self.moveButton = QToolButton()
        self.rulerButton = QToolButton()
        self.rectangleButton = QToolButton()
        self.lineCutButton = QToolButton()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageToolsGroupBox:
        view = cls(parent)

        view.homeButton.setIcon(QIcon(':/icons/home'))
        view.homeButton.setIconSize(QSize(32, 32))
        view.homeButton.setToolTip('Home')

        view.saveButton.setIcon(QIcon(':/icons/save'))
        view.saveButton.setIconSize(QSize(32, 32))
        view.saveButton.setToolTip('Save Image')

        view.moveButton.setIcon(QIcon(':/icons/move'))
        view.moveButton.setIconSize(QSize(32, 32))
        view.moveButton.setToolTip('Move')

        view.rulerButton.setIcon(QIcon(':/icons/ruler'))
        view.rulerButton.setIconSize(QSize(32, 32))
        view.rulerButton.setToolTip('Ruler')

        view.rectangleButton.setIcon(QIcon(':/icons/rectangle'))
        view.rectangleButton.setIconSize(QSize(32, 32))
        view.rectangleButton.setToolTip('Rectangle')

        view.lineCutButton.setIcon(QIcon(':/icons/line-cut'))
        view.lineCutButton.setIconSize(QSize(32, 32))
        view.lineCutButton.setToolTip('Line-Cut Profile')

        layout = QGridLayout()
        layout.addWidget(view.homeButton, 0, 0)
        layout.addWidget(view.saveButton, 0, 1)
        layout.addWidget(view.moveButton, 0, 2)
        layout.addWidget(view.rulerButton, 1, 0)
        layout.addWidget(view.rectangleButton, 1, 1)
        layout.addWidget(view.lineCutButton, 1, 2)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        return view


class ImageRendererGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Colorize', parent)
        self.rendererComboBox = QComboBox()
        self.transformationComboBox = QComboBox()
        self.variantComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageRendererGroupBox:
        view = cls(parent)

        view.rendererComboBox.setToolTip('Array Component')
        view.transformationComboBox.setToolTip('Transformation')
        view.variantComboBox.setToolTip('Variant')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(view.rendererComboBox)
        layout.addWidget(view.transformationComboBox)
        layout.addWidget(view.variantComboBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        return view


class ImageDataRangeGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Data Range', parent)
        self.minDisplayValueSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.maxDisplayValueSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.autoButton = QPushButton('Auto')
        self.editButton = QPushButton('Edit')
        self.colorLegendButton = QPushButton('Color Legend')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageDataRangeGroupBox:
        view = cls(parent)

        view.minDisplayValueSlider.setToolTip('Minimum Display Value')
        view.maxDisplayValueSlider.setToolTip('Maximum Display Value')
        view.autoButton.setToolTip('Rescale to Data Range')
        view.editButton.setToolTip('Rescale to Custom Range')
        view.colorLegendButton.setToolTip('Toggle Color Legend Visibility')

        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addWidget(view.autoButton)
        buttonLayout.addWidget(view.editButton)
        buttonLayout.addWidget(view.colorLegendButton)

        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addRow('Min:', view.minDisplayValueSlider)
        layout.addRow('Max:', view.maxDisplayValueSlider)
        layout.addRow(buttonLayout)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        return view


class ImageRibbon(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.imageToolsGroupBox = ImageToolsGroupBox.createInstance()
        self.colormapGroupBox = ImageRendererGroupBox.createInstance()
        self.dataRangeGroupBox = ImageDataRangeGroupBox.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageRibbon:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageToolsGroupBox)
        layout.addWidget(view.colormapGroupBox)
        layout.addWidget(view.dataRangeGroupBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        return view


class ImageWidget(VisualizationView):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colorLegendMinValue = 0.
        self._colorLegendMaxValue = 1.
        self._colorLegendStopPoints: list[tuple[float, QColor]] = [
            (0.0, QColor(Qt.GlobalColor.green)),
            (0.5, QColor(Qt.GlobalColor.yellow)),
            (1.0, QColor(Qt.GlobalColor.red)),
        ]
        self._colorLegendNumberOfTicks = 5  # TODO
        self._isColorLegendVisible = False
        self._isColorLegendCyclic = False

    def setColorLegendColors(self, values: RealArrayType, rgbaArray: RealArrayType,
                             isCyclic: bool) -> None:
        colorLegendStopPoints: list[tuple[float, QColor]] = list()
        self._colorLegendMinValue = values.min()
        self._colorLegendMaxValue = values.max()

        valueRange = self._colorLegendMaxValue - self._colorLegendMinValue
        normalizedValues = (values - self._colorLegendMinValue) / valueRange if valueRange > 0 \
                else numpy.full_like(values, 0.5)

        for x, rgba in zip(normalizedValues.clip(0, 1), rgbaArray):
            color = QColor()
            color.setRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
            colorLegendStopPoints.append((x, color))

        self._colorLegendStopPoints = colorLegendStopPoints
        self._isColorLegendCyclic = isCyclic
        self.scene().update()

    def setColorLegendVisible(self, visible: bool) -> None:
        self._isColorLegendVisible = visible
        self.scene().update()

    @property
    def _colorLegendTicks(self) -> Iterator[float]:
        for tick in range(self._colorLegendNumberOfTicks):
            a = tick / (self._colorLegendNumberOfTicks - 1)
            yield (1. - a) * self._colorLegendMinValue + a * self._colorLegendMaxValue

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if not self._isColorLegendVisible:
            return

        fgPainter = QPainter(self.viewport())

        pen = QPen()
        pen.setWidth(3)
        fgPainter.setPen(pen)

        fontMetrics = fgPainter.fontMetrics()
        dx = fontMetrics.horizontalAdvance('m')
        dy = fontMetrics.lineSpacing()

        widgetRect = self.viewport().rect()

        if self._isColorLegendCyclic:
            legendDiameter = 6 * dx
            legendMargin = 2 * dx

            legendRect = QRect(0, 0, legendDiameter, legendDiameter)
            legendRect.moveRight(widgetRect.right() - legendMargin)
            legendRect.moveBottom(widgetRect.height() - legendMargin)

            cgradient = QConicalGradient(legendRect.center(), 90.)
            cgradient.setStops(self._colorLegendStopPoints)
            fgPainter.setBrush(cgradient)
            fgPainter.drawEllipse(legendRect)
        else:
            tickLabels = [f'{tick:5g}' for tick in self._colorLegendTicks]
            tickLabelWidth = max(fontMetrics.width(label) for label in tickLabels)

            legendWidth = 2 * dx
            legendHeight = (2 * len(tickLabels) - 1) * dy
            legendMargin = tickLabelWidth + 2 * dx

            legendRect = QRect(0, 0, legendWidth, legendHeight)
            legendRect.moveRight(widgetRect.right() - legendMargin)
            legendRect.moveTop((widgetRect.height() - legendHeight) // 2)

            lgradient = QLinearGradient(legendRect.bottomLeft(), legendRect.topLeft())
            lgradient.setStops(self._colorLegendStopPoints)
            fgPainter.setBrush(lgradient)
            fgPainter.drawRect(legendRect)

            tickX0 = legendRect.right() + dx
            tickY0 = legendRect.bottom() + fontMetrics.strikeOutPos()

            for tickIndex, tickLabel in enumerate(tickLabels):
                tickDY = (tickIndex * legendRect.height()) // (len(tickLabels) - 1)
                viewportPoint = QPoint(tickX0, tickY0 - tickDY)
                fgPainter.drawText(viewportPoint, tickLabel)


class ImageView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.imageRibbon = ImageRibbon.createInstance()
        self.imageWidget = ImageWidget()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ImageView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(view.imageRibbon)
        layout.addWidget(view.imageWidget)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        return view
