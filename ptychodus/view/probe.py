from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget

from .widgets import DecimalLineEdit, EnergyWidget, LengthWidget, SemiautomaticSpinBox


class ProbeInitializerView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Initializer', parent)
        self.initializerComboBox = QComboBox()
        self.initializeButton = QPushButton('Initialize')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerComboBox)
        layout.addWidget(view.initializeButton)
        view.setLayout(layout)

        return view


class ProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Parameters', parent)
        self.sizeSpinBox = SemiautomaticSpinBox.createInstance()
        self.energyWidget = EnergyWidget.createInstance()
        self.wavelengthWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Size [px]:', view.sizeSpinBox)
        layout.addRow('Energy:', view.energyWidget)
        layout.addRow('Wavelength:', view.wavelengthWidget)
        view.setLayout(layout)

        return view


class ProbeSuperGaussianView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Super Gaussian', parent)
        self.annularRadiusWidget = LengthWidget.createInstance()
        self.probeWidthWidget = LengthWidget.createInstance()
        self.orderParameterWidget = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeSuperGaussianView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Annular Radius:', view.annularRadiusWidget)
        layout.addRow('Probe Width:', view.probeWidthWidget)
        layout.addRow('Order Parameter:', view.orderParameterWidget)
        view.setLayout(layout)

        return view


class ProbeZonePlateView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Fresnel Zone Plate', parent)
        self.zonePlateRadiusWidget = LengthWidget.createInstance()
        self.outermostZoneWidthWidget = LengthWidget.createInstance()
        self.beamstopDiameterWidget = LengthWidget.createInstance()
        self.defocusDistanceWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeZonePlateView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Zone Plate Radius:', view.zonePlateRadiusWidget)
        layout.addRow('Outermost Zone Width:', view.outermostZoneWidthWidget)
        layout.addRow('Beamstop Diameter:', view.beamstopDiameterWidget)
        layout.addRow('Defocus Distance:', view.defocusDistanceWidget)
        view.setLayout(layout)

        return view


class ProbeParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.initializerView = ProbeInitializerView.createInstance()
        self.probeView = ProbeView.createInstance()
        self.superGaussianView = ProbeSuperGaussianView.createInstance()
        self.zonePlateView = ProbeZonePlateView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerView)
        layout.addWidget(view.probeView)
        layout.addWidget(view.superGaussianView)
        layout.addWidget(view.zonePlateView)
        layout.addStretch()
        view.setLayout(layout)

        return view
