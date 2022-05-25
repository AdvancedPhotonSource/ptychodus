from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QGroupBox, QWidget, QSpinBox, QFormLayout, QComboBox, QPushButton, QVBoxLayout

from .widgets import SemiautomaticSpinBox, LengthWidget, EnergyWidget


class ProbeProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Parameters', parent)
        self.sizeSpinBox = SemiautomaticSpinBox.createInstance()
        self.energyWidget = EnergyWidget.createInstance()
        self.wavelengthWidget = LengthWidget.createInstance()
        self.diameterWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Size [px]:', view.sizeSpinBox)
        layout.addRow('Energy:', view.energyWidget)
        layout.addRow('Wavelength:', view.wavelengthWidget)
        layout.addRow('Diameter:', view.diameterWidget)
        view.setLayout(layout)

        return view


class ProbeZonePlateView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Zone Plate', parent)
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


class ProbeParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.initializerView = ProbeInitializerView.createInstance()
        self.probeView = ProbeProbeView.createInstance()
        self.zonePlateView = ProbeZonePlateView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerView)
        layout.addWidget(view.probeView)
        layout.addWidget(view.zonePlateView)
        layout.addStretch()
        view.setLayout(layout)

        return view
