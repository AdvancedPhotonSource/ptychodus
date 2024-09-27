from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import DecimalLineEdit, LengthWidget


class ProbePropagationParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Parameters", parent)
        self.beginCoordinateWidget = LengthWidget.createInstance(isSigned=True)
        self.endCoordinateWidget = LengthWidget.createInstance(isSigned=True)
        self.numberOfStepsSpinBox = QSpinBox()
        self.visualizationParametersView = VisualizationParametersView.createInstance()

        propagationLayout = QFormLayout()
        propagationLayout.addRow("Begin Coordinate:", self.beginCoordinateWidget)
        propagationLayout.addRow("End Coordinate:", self.endCoordinateWidget)
        propagationLayout.addRow("Number of Steps:", self.numberOfStepsSpinBox)

        propagationGroupBox = QGroupBox("Propagation")
        propagationGroupBox.setLayout(propagationLayout)

        layout = QVBoxLayout()
        layout.addWidget(propagationGroupBox)
        layout.addWidget(self.visualizationParametersView)
        layout.addStretch()
        self.setLayout(layout)


class ProbePropagationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.xyView = VisualizationWidget.createInstance("XY Plane")
        self.zxView = VisualizationWidget.createInstance("ZX Plane")
        self.parametersView = ProbePropagationParametersView()
        self.zyView = VisualizationWidget.createInstance("ZY Plane")
        self.propagateButton = QPushButton("Propagate")
        self.saveButton = QPushButton("Save")
        self.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        self.coordinateLabel = QLabel()
        self.statusBar = QStatusBar()

        actionLayout = QHBoxLayout()
        actionLayout.addWidget(self.propagateButton)
        actionLayout.addWidget(self.saveButton)

        coordinateLayout = QHBoxLayout()
        coordinateLayout.setContentsMargins(0, 0, 0, 0)
        coordinateLayout.addWidget(self.coordinateSlider)
        coordinateLayout.addWidget(self.coordinateLabel)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(self.xyView, 0, 0)
        contentsLayout.addWidget(self.zxView, 0, 1)
        contentsLayout.addWidget(self.parametersView, 1, 0)
        contentsLayout.addWidget(self.zyView, 1, 1)
        contentsLayout.addLayout(actionLayout, 2, 0)
        contentsLayout.addLayout(coordinateLayout, 2, 1)
        contentsLayout.setColumnStretch(0, 1)
        contentsLayout.setColumnStretch(1, 2)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


class STXMDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualizationWidget = VisualizationWidget.createInstance("Transmission")
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton("Save")
        self.statusBar = QStatusBar()

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(self.visualizationParametersView)
        parameterLayout.addStretch()
        parameterLayout.addWidget(self.saveButton)

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(self.visualizationWidget, 1)
        contentsLayout.addLayout(parameterLayout)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


class ExposureParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Parameters", parent)
        self.quantitativeProbeCheckBox = QCheckBox("Quantitative Probe")
        self.photonFluxLineEdit = DecimalLineEdit.createInstance()
        self.exposureTimeLineEdit = DecimalLineEdit.createInstance()
        self.massAttenuationLabel = QLabel("Mass Attenuation [m\u00b2/kg]:")
        self.massAttenuationLineEdit = DecimalLineEdit.createInstance()

        layout = QFormLayout()
        layout.addRow(self.quantitativeProbeCheckBox)
        layout.addRow("Photon Flux [ph/s]:", self.photonFluxLineEdit)
        layout.addRow("Exposure Time [s]:", self.exposureTimeLineEdit)
        layout.addRow(self.massAttenuationLabel)
        layout.addRow(self.massAttenuationLineEdit)
        self.setLayout(layout)


class ExposureQuantityView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Quantity", parent)
        self.photonCountButton = QRadioButton("Photon Count")
        self.photonFluxButton = QRadioButton("Photon Flux [Hz]")
        self.exposureButton = QRadioButton("Exposure [J/m\u00b2]")
        self.irradianceButton = QRadioButton("Irradiance [W/m\u00b2]")
        self.doseButton = QRadioButton("Dose [Gy]")
        self.doseRateButton = QRadioButton("Dose Rate [Gy/s]")

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
        self.visualizationWidget = VisualizationWidget.createInstance("Visualization")
        self.exposureParametersView = ExposureParametersView()
        self.exposureQuantityView = ExposureQuantityView()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.saveButton = QPushButton("Save")
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
        super().__init__("Parameters", parent)
        self.openButton = QPushButton("Open")
        self.enhancementStrategyComboBox = QComboBox()
        self.upscalingStrategyComboBox = QComboBox()
        self.deconvolutionStrategyComboBox = QComboBox()
        self.enhanceButton = QPushButton("Enhance")
        self.saveButton = QPushButton("Save")

        layout = QFormLayout()
        layout.addRow("Measured Dataset:", self.openButton)
        layout.addRow("Enhancement Strategy:", self.enhancementStrategyComboBox)
        layout.addRow("Upscaling Strategy:", self.upscalingStrategyComboBox)
        layout.addRow("Deconvolution Strategy:", self.deconvolutionStrategyComboBox)
        layout.addRow(self.enhanceButton)
        layout.addRow("Enhanced Dataset:", self.saveButton)
        self.setLayout(layout)


class FluorescenceDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.measuredWidget = VisualizationWidget.createInstance("Measured")
        self.enhancedWidget = VisualizationWidget.createInstance("Enhanced")
        self.fluorescenceParametersView = FluorescenceParametersView()
        self.fluorescenceChannelListView = QListView()
        self.visualizationParametersView = VisualizationParametersView.createInstance()
        self.statusBar = QStatusBar()

        parameterLayout = QVBoxLayout()
        parameterLayout.addWidget(self.fluorescenceParametersView)
        parameterLayout.addWidget(self.fluorescenceChannelListView, 1)
        parameterLayout.addWidget(self.visualizationParametersView)
        parameterLayout.addStretch()

        contentsLayout = QHBoxLayout()
        contentsLayout.addWidget(self.measuredWidget, 1)
        contentsLayout.addWidget(self.enhancedWidget, 1)
        contentsLayout.addLayout(parameterLayout)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)
