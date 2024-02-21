from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QGridLayout,
                             QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class FourierRingCorrelationParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.name1Label = QLabel('Name 1:')
        self.name1ComboBox = QComboBox()
        self.name2Label = QLabel('Name 2:')
        self.name2ComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationParametersView:
        view = cls(parent)
        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        layout = QGridLayout()
        layout.addWidget(view.name1Label, 0, 0)
        layout.addWidget(view.name1ComboBox, 0, 1)
        layout.addWidget(view.name2Label, 0, 2)
        layout.addWidget(view.name2ComboBox, 0, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view


class FourierRingCorrelationDialog(QDialog):

    def __init__(self, buttonBox: QDialogButtonBox, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._buttonBox = buttonBox
        self.parametersView = FourierRingCorrelationParametersView.createInstance()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationDialog:
        buttonBox = QDialogButtonBox()
        view = cls(buttonBox, parent)
        view.setWindowTitle('Fourier Ring Correlation')

        buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        layout.addWidget(buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self._buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
