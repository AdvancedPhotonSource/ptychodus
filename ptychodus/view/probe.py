from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QWidget, QSpinBox, QFormLayout, QComboBox, QPushButton, QVBoxLayout

from .widgets import LengthWidget, EnergyWidget


class ProbeProbeView(QGroupBox):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__('Parameters', parent)
        self.shapeSpinBox = QSpinBox()
        self.energyWidget = EnergyWidget.createInstance()
        self.wavelengthWidget = LengthWidget.createInstance()
        self.diameterWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ProbeProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Shape:', view.shapeSpinBox)
        layout.addRow('Energy:', view.energyWidget)
        layout.addRow('Wavelength:', view.wavelengthWidget)
        layout.addRow('Diameter:', view.diameterWidget)
        view.setLayout(layout)

        return view


class ProbeZonePlateView(QGroupBox):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__('Zone Plate', parent)
        self.zonePlateRadiusWidget = LengthWidget.createInstance()
        self.outermostZoneWidthWidget = LengthWidget.createInstance()
        self.beamstopDiameterWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ProbeZonePlateView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Zone Plate Radius:', view.zonePlateRadiusWidget)
        layout.addRow('Outermost Zone Width:', view.outermostZoneWidthWidget)
        layout.addRow('Beamstop Diameter:', view.beamstopDiameterWidget)
        view.setLayout(layout)

        return view


class ProbeInitializerView(QGroupBox):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__('Initializer', parent)
        self.initializerComboBox = QComboBox()
        self.initializeButton = QPushButton('Initialize')

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ProbeInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerComboBox)
        layout.addWidget(view.initializeButton)
        view.setLayout(layout)

        return view


class ProbeParametersView(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.probeView = ProbeProbeView.createInstance()
        self.zonePlateView = ProbeZonePlateView.createInstance()
        self.initializerView = ProbeInitializerView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.probeView)
        layout.addWidget(view.zonePlateView)
        layout.addWidget(view.initializerView)
        layout.addStretch()
        view.setLayout(layout)

        return view
