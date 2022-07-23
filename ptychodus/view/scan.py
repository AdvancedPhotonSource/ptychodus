from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHeaderView, QHBoxLayout, QMenu,
                             QPushButton, QSpinBox, QTableView, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .widgets import LengthWidget


class ScanEditorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Editor', parent)
        self.numberOfScanPointsSpinBox = QSpinBox()
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()
        self.stepSizeXWidget = LengthWidget.createInstance()
        self.stepSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanEditorView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Points:', view.numberOfScanPointsSpinBox)
        layout.addRow('Extent X:', view.extentXSpinBox)
        layout.addRow('Extent Y:', view.extentYSpinBox)
        layout.addRow('Step Size X:', view.stepSizeXWidget)
        layout.addRow('Step Size Y:', view.stepSizeYWidget)
        view.setLayout(layout)

        return view


class ScanTransformView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Transform', parent)
        self.transformComboBox = QComboBox()
        self.jitterRadiusWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanTransformView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('(x,y) \u2192', view.transformComboBox)
        layout.addRow('Jitter Radius:', view.jitterRadiusWidget)
        view.setLayout(layout)

        return view


class ScanButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.editButton = QPushButton('Edit')
        self.saveButton = QPushButton('Save')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.editButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class ScanParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Position Data', parent)
        self.tableView = QTableView()
        self.buttonBox = ScanButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanParametersView:
        view = cls(parent)

        view.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
