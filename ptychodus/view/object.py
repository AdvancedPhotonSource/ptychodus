from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QGridLayout,
                             QLabel, QStatusBar, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .image import ImageView


class FourierRingCorrelationDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.buttonBox = QDialogButtonBox()
        self.name1Label = QLabel('Name 1:')
        self.name1ComboBox = QComboBox()
        self.name2Label = QLabel('Name 2:')
        self.name2ComboBox = QComboBox()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationDialog:
        view = cls(parent)
        view.setWindowTitle('Fourier Ring Correlation')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QGridLayout()
        layout.addWidget(view.name1Label, 0, 0)
        layout.addWidget(view.name1ComboBox, 0, 1)
        layout.addWidget(view.name2Label, 0, 2)
        layout.addWidget(view.name2ComboBox, 0, 3)
        layout.addWidget(view.navigationToolbar, 1, 0, 1, 4)
        layout.addWidget(view.figureCanvas, 2, 0, 1, 4)
        layout.addWidget(view.buttonBox, 3, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class DichroicDialog(QDialog):

    def __init__(self, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.lcircNameLabel = QLabel('Left Circular:')
        self.lcircNameComboBox = QComboBox()
        self.rcircNameLabel = QLabel('Right Circular:')
        self.rcircNameComboBox = QComboBox()
        self.imageView = ImageView.createInstance(statusBar)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> DichroicDialog:
        view = cls(statusBar, parent)
        view.setWindowTitle('Dichroic Analysis')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QGridLayout()
        layout.addWidget(view.lcircNameLabel, 0, 0)
        layout.addWidget(view.lcircNameComboBox, 0, 1)
        layout.addWidget(view.rcircNameLabel, 0, 2)
        layout.addWidget(view.rcircNameComboBox, 0, 3)
        layout.addWidget(view.imageView, 1, 0, 1, 4)
        layout.addWidget(view.buttonBox, 2, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
