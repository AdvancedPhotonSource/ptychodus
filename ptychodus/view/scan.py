from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QHeaderView, QHBoxLayout, QMenu, QPushButton, QSpinBox,
                             QTableView, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .widgets import LengthWidget


class ScanEditorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
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

    def __init__(self, parent: Optional[QWidget]) -> None:
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


class ScanEditorDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = ScanEditorView.createInstance()
        self.transformView = ScanTransformView.createInstance()
        self.centerWidget = QWidget()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanEditorDialog:
        view = cls(parent)

        view.setWindowTitle('Edit Scan')

        centerLayout = QVBoxLayout()
        centerLayout.addWidget(view.editorView)
        centerLayout.addWidget(view.transformView)
        view.centerWidget.setLayout(centerLayout)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.centerWidget)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ScanButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.editButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class ScanPositionDataView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Position Data', parent)
        self.tableView = QTableView()
        self.buttonBox = ScanButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanPositionDataView:
        view = cls(parent)

        view.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ScanParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.positionDataView = ScanPositionDataView.createInstance()
        self.editorDialog = ScanEditorDialog.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.positionDataView)
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
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
