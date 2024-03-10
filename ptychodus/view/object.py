from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsView, QGridLayout, QGroupBox, QLabel, QPushButton,
                             QStatusBar, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .image import ImageView


class FourierRingCorrelationDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.name1Label = QLabel('Name 1:')
        self.name1ComboBox = QComboBox()
        self.name2Label = QLabel('Name 2:')
        self.name2ComboBox = QComboBox()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationDialog:
        view = cls(parent)
        view.setWindowTitle('Fourier Ring Correlation')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        parametersLayout = QGridLayout()
        parametersLayout.addWidget(view.name1Label, 0, 0)
        parametersLayout.addWidget(view.name1ComboBox, 0, 1)
        parametersLayout.addWidget(view.name2Label, 0, 2)
        parametersLayout.addWidget(view.name2ComboBox, 0, 3)
        parametersLayout.setColumnStretch(1, 1)
        parametersLayout.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.addLayout(parametersLayout)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class DichroicGraphicsView(QGroupBox):  # TODO remove when able

    def __init__(self, title: str, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.graphicsView = QGraphicsView()

    @classmethod
    def createInstance(cls,
                       title: str,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> DichroicGraphicsView:
        view = cls(title, statusBar, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.graphicsView)
        view.setLayout(layout)

        return view


class DichroicImageView(QGroupBox):

    def __init__(self, title: str, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.imageView = ImageView.createInstance(statusBar)

    @classmethod
    def createInstance(cls,
                       title: str,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> DichroicImageView:
        view = cls(title, statusBar, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageView)
        view.setLayout(layout)

        return view


class DichroicParametersView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.lcircComboBox = QComboBox()
        self.rcircComboBox = QComboBox()
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, title: str, parent: QWidget | None = None) -> DichroicParametersView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QFormLayout()
        layout.addRow('Left Circular:', view.lcircComboBox)
        layout.addRow('Right Circular:', view.rcircComboBox)
        layout.addRow(view.saveButton)
        view.setLayout(layout)

        return view


class DichroicDialog(QDialog):

    def __init__(self, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.numeratorImageView = DichroicGraphicsView.createInstance('Numerator', statusBar)
        self.ratioImageView = DichroicImageView.createInstance('Ratio', statusBar)
        self.denominatorImageView = DichroicGraphicsView.createInstance('Denominator', statusBar)
        self.parametersView = DichroicParametersView.createInstance('Parameters')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> DichroicDialog:
        view = cls(statusBar, parent)
        view.setWindowTitle('Dichroic Analysis')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(view.numeratorImageView, 0, 0)
        contentsLayout.addWidget(view.ratioImageView, 0, 1)
        contentsLayout.addWidget(view.denominatorImageView, 1, 0)
        contentsLayout.addWidget(view.parametersView, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
