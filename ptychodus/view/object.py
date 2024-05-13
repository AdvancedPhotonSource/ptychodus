from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QRadioButton, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import DecimalLineEdit


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


class STXMDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.visualizationWidget = VisualizationWidget.createInstance('Transmission')
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton('Save')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> STXMDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(view.visualizationParametersView)
        parameterLayout.addWidget(view.saveButton)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(view.visualizationWidget, 1)
        contentsLayout.addLayout(parameterLayout)

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


class ExposureParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.photonsButton = QRadioButton('Photons Per Pixel')
        self.exposureButton = QRadioButton('Exposure [J/m^2]')
        self.doseButton = QRadioButton('Dose [Gy]')
        self.massAttenuationLineEdit = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExposureParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.photonsButton)
        layout.addRow(view.exposureButton)
        layout.addRow(view.doseButton)
        layout.addRow('Mass Attenuation\nCoefficient [m^2/kg]:', view.massAttenuationLineEdit)
        view.setLayout(layout)

        return view


class ExposureDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.visualizationWidget = VisualizationWidget.createInstance('Exposure')
        self.exposureParametersView = ExposureParametersView.createInstance()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton('Save')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExposureDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(view.exposureParametersView)
        parameterLayout.addWidget(view.visualizationParametersView)
        parameterLayout.addWidget(view.saveButton)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(view.visualizationWidget, 1)
        contentsLayout.addLayout(parameterLayout)

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


class FluorescenceParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.channelComboBox = QComboBox()
        self.upscalingStrategyComboBox = QComboBox()
        self.deconvolutionStrategyComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FluorescenceParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Channel:', view.channelComboBox)
        layout.addRow('Upscaling Strategy:', view.upscalingStrategyComboBox)
        layout.addRow('Deconvolution Strategy:', view.deconvolutionStrategyComboBox)
        view.setLayout(layout)

        return view


class FluorescenceDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.measuredWidget = VisualizationWidget.createInstance('Measured')
        self.enhancedWidget = VisualizationWidget.createInstance('Enhanced')
        self.fluorescenceParametersView = FluorescenceParametersView.createInstance()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FluorescenceDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(view.openButton)
        buttonsLayout.addWidget(view.saveButton)

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(view.fluorescenceParametersView)
        parameterLayout.addWidget(view.visualizationParametersView)
        parameterLayout.addLayout(buttonsLayout)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(view.measuredWidget, 1)
        contentsLayout.addWidget(view.enhancedWidget, 1)
        contentsLayout.addLayout(parameterLayout)

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


class XMCDParametersView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)

        self.polarizationGroupBox = QGroupBox('Polarization')
        self.lcircComboBox = QComboBox()
        self.rcircComboBox = QComboBox()
        self.saveButton = QPushButton('Save')
        self.visualizationParametersView = VisualizationParametersView.createInstance()

    @classmethod
    def createInstance(cls, title: str, parent: QWidget | None = None) -> XMCDParametersView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        polarizationLayout = QFormLayout()
        polarizationLayout.addRow('Left Circular:', view.lcircComboBox)
        polarizationLayout.addRow('Right Circular:', view.rcircComboBox)
        polarizationLayout.addRow(view.saveButton)
        view.polarizationGroupBox.setLayout(polarizationLayout)

        layout = QVBoxLayout()
        layout.addWidget(view.polarizationGroupBox)
        layout.addWidget(view.visualizationParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class XMCDDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.differenceWidget = VisualizationWidget.createInstance('Difference')
        self.ratioWidget = VisualizationWidget.createInstance('Ratio')
        self.sumWidget = VisualizationWidget.createInstance('Sum')
        self.parametersView = XMCDParametersView.createInstance('Parameters')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> XMCDDialog:
        view = cls(parent)
        view.setWindowTitle('XMCD Analysis')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(view.differenceWidget, 0, 0)
        contentsLayout.addWidget(view.ratioWidget, 0, 1)
        contentsLayout.addWidget(view.sumWidget, 1, 0)
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
