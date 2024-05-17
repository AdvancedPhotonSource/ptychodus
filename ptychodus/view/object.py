from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QPushButton, QRadioButton, QStatusBar,
                             QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import DecimalLineEdit


class FourierRingCorrelationDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product1Label = QLabel('Product 1:')
        self.product1ComboBox = QComboBox()
        self.product2Label = QLabel('Product 2:')
        self.product2ComboBox = QComboBox()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

        parametersLayout = QGridLayout()
        parametersLayout.addWidget(self.product1Label, 0, 0)
        parametersLayout.addWidget(self.product1ComboBox, 0, 1)
        parametersLayout.addWidget(self.product2Label, 0, 2)
        parametersLayout.addWidget(self.product2ComboBox, 0, 3)
        parametersLayout.setColumnStretch(1, 1)
        parametersLayout.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.navigationToolbar)
        layout.addWidget(self.figureCanvas)
        layout.addLayout(parametersLayout)
        self.setLayout(layout)


class STXMNormalizationView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Normalization', parent)
        self.quantitativeProbeCheckBox = QCheckBox('Quantitative Probe')
        self.photonFluxLineEdit = DecimalLineEdit.createInstance()
        self.exposureTimeLineEdit = DecimalLineEdit.createInstance()

        layout = QFormLayout()
        layout.addRow(self.quantitativeProbeCheckBox)
        layout.addRow('Photon Flux [ph/s]:', self.photonFluxLineEdit)
        layout.addRow('Exposure Time [s]:', self.exposureTimeLineEdit)
        self.setLayout(layout)


class STXMDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualizationWidget = VisualizationWidget.createInstance('Transmission')
        self.normalizationView = STXMNormalizationView()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton('Save')
        self.statusBar = QStatusBar()

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(self.normalizationView)
        parameterLayout.addWidget(self.visualizationParametersView)
        parameterLayout.addWidget(self.saveButton)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(self.visualizationWidget, 1)
        contentsLayout.addLayout(parameterLayout)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


class ExposureParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.quantitativeProbeCheckBox = QCheckBox('Quantitative Probe')
        self.photonFluxLineEdit = DecimalLineEdit.createInstance()
        self.exposureTimeLineEdit = DecimalLineEdit.createInstance()
        self.massAttenuationLabel = QLabel('Mass Attenuation [m\u00B2/kg]:')
        self.massAttenuationLineEdit = DecimalLineEdit.createInstance()

        layout = QFormLayout()
        layout.addRow(self.quantitativeProbeCheckBox)
        layout.addRow('Photon Flux [ph/s]:', self.photonFluxLineEdit)
        layout.addRow('Exposure Time [s]:', self.exposureTimeLineEdit)
        layout.addRow(self.massAttenuationLabel)
        layout.addRow(self.massAttenuationLineEdit)
        self.setLayout(layout)


class ExposureQuantityView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Quantity', parent)
        self.photonCountButton = QRadioButton('Photon Count')
        self.photonFluxButton = QRadioButton('Photon Flux [Hz]')
        self.exposureButton = QRadioButton('Exposure [J/m\u00B2]')
        self.irradianceButton = QRadioButton('Irradiance [W/m\u00B2]')
        self.doseButton = QRadioButton('Dose [Gy]')
        self.doseRateButton = QRadioButton('Dose Rate [Gy/s]')

        layout = QVBoxLayout()
        layout.addWidget(self.photonCountButton)
        layout.addWidget(self.photonFluxButton)
        layout.addWidget(self.exposureButton)
        layout.addWidget(self.irradianceButton)
        layout.addWidget(self.doseButton)
        layout.addWidget(self.doseRateButton)
        self.setLayout(layout)


class ExposureDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualizationWidget = VisualizationWidget.createInstance('Visualization')
        self.exposureParametersView = ExposureParametersView()
        self.exposureQuantityView = ExposureQuantityView()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton('Save')
        self.statusBar = QStatusBar()

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(self.exposureParametersView)
        parameterLayout.addWidget(self.exposureQuantityView)
        parameterLayout.addWidget(self.visualizationParametersView)
        parameterLayout.addWidget(self.saveButton)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(self.visualizationWidget, 1)
        contentsLayout.addLayout(parameterLayout)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


class FluorescenceParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.channelComboBox = QComboBox()
        self.upscalingStrategyComboBox = QComboBox()
        self.deconvolutionStrategyComboBox = QComboBox()

        layout = QFormLayout()
        layout.addRow('Channel:', self.channelComboBox)
        layout.addRow('Upscaling Strategy:', self.upscalingStrategyComboBox)
        layout.addRow('Deconvolution Strategy:', self.deconvolutionStrategyComboBox)
        self.setLayout(layout)


class FluorescenceDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.measuredWidget = VisualizationWidget.createInstance('Measured')
        self.enhancedWidget = VisualizationWidget.createInstance('Enhanced')
        self.fluorescenceParametersView = FluorescenceParametersView()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')
        self.statusBar = QStatusBar()

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.openButton)
        buttonsLayout.addWidget(self.saveButton)

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(self.fluorescenceParametersView)
        parameterLayout.addWidget(self.visualizationParametersView)
        parameterLayout.addLayout(buttonsLayout)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(self.measuredWidget, 1)
        contentsLayout.addWidget(self.enhancedWidget, 1)
        contentsLayout.addLayout(parameterLayout)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


class XMCDParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)

        self.polarizationGroupBox = QGroupBox('Polarization')
        self.lcircComboBox = QComboBox()
        self.rcircComboBox = QComboBox()
        self.saveButton = QPushButton('Save')
        self.visualizationParametersView = VisualizationParametersView.createInstance()

        polarizationLayout = QFormLayout()
        polarizationLayout.addRow('Left Circular:', self.lcircComboBox)
        polarizationLayout.addRow('Right Circular:', self.rcircComboBox)
        polarizationLayout.addRow(self.saveButton)
        self.polarizationGroupBox.setLayout(polarizationLayout)

        layout = QVBoxLayout()
        layout.addWidget(self.polarizationGroupBox)
        layout.addWidget(self.visualizationParametersView)
        layout.addStretch()
        self.setLayout(layout)


class XMCDDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.differenceWidget = VisualizationWidget.createInstance('Difference')
        self.ratioWidget = VisualizationWidget.createInstance('Ratio')
        self.sumWidget = VisualizationWidget.createInstance('Sum')
        self.parametersView = XMCDParametersView()
        self.statusBar = QStatusBar()

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(self.differenceWidget, 0, 0)
        contentsLayout.addWidget(self.ratioWidget, 0, 1)
        contentsLayout.addWidget(self.sumWidget, 1, 0)
        contentsLayout.addWidget(self.parametersView, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)
