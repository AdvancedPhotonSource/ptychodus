from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHeaderView, QHBoxLayout, QMenu,
                             QPushButton, QSizePolicy, QStackedWidget, QTableView, QVBoxLayout,
                             QWidget)

from .widgets import DecimalLineEdit, EnergyWidget, LengthWidget, SemiautomaticSpinBox


class ProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
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


class ProbeSuperGaussianView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
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


class ProbeZonePlateView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
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

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Initializer', parent)
        self.superGaussianView = ProbeSuperGaussianView.createInstance()
        self.zonePlateView = ProbeZonePlateView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.superGaussianView)
        layout.addWidget(view.zonePlateView)
        view.setLayout(layout)

        return view


class ProbeModesButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.initializeMenu = QMenu()
        self.initializeButton = QPushButton('Initialize')
        self.saveButton = QPushButton('Save')
        self.pushModeButton = QPushButton('Push Mode')
        self.popModeButton = QPushButton('Pop Mode')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeModesButtonBox:
        view = cls(parent)

        view.initializeButton.setMenu(view.initializeMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.initializeButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.pushModeButton)
        layout.addWidget(view.popModeButton)
        view.setLayout(layout)

        return view


class ProbeModesView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Modes', parent)
        self.tableView = QTableView()
        self.buttonBox = ProbeModesButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeModesView:
        view = cls(parent)

        view.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ProbeParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.probeView = ProbeView.createInstance()
        self.initializerView = ProbeInitializerView.createInstance()
        self.modesView = ProbeModesView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.probeView)
        layout.addWidget(view.modesView)
        view.setLayout(layout)

        return view
