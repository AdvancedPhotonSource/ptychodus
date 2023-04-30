from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QMenu, QPushButton, QSpinBox, QTreeView, QVBoxLayout,
                             QWidget)

from .widgets import DecimalLineEdit, EnergyWidget, LengthWidget, SemiautomaticSpinBox


class ProbeParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.sizeSpinBox = SemiautomaticSpinBox.createInstance()
        self.energyWidget = EnergyWidget.createInstance()
        self.wavelengthWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Size [px]:', view.sizeSpinBox)
        layout.addRow('Energy:', view.energyWidget)
        layout.addRow('Wavelength:', view.wavelengthWidget)
        view.setLayout(layout)

        return view


class SuperGaussianProbeView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.annularRadiusWidget = LengthWidget.createInstance()
        self.probeWidthWidget = LengthWidget.createInstance()
        self.orderParameterWidget = DecimalLineEdit.createInstance()
        self.numberOfModesSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SuperGaussianProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Annular Radius:', view.annularRadiusWidget)
        layout.addRow('Probe Width:', view.probeWidthWidget)
        layout.addRow('Order Parameter:', view.orderParameterWidget)
        layout.addRow('Number of Modes:', view.numberOfModesSpinBox)
        view.setLayout(layout)

        return view


class FresnelZonePlateProbeView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.zonePlateRadiusWidget = LengthWidget.createInstance()
        self.outermostZoneWidthWidget = LengthWidget.createInstance()
        self.beamstopDiameterWidget = LengthWidget.createInstance()
        self.defocusDistanceWidget = LengthWidget.createInstance()
        self.numberOfModesSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> FresnelZonePlateProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Zone Plate Radius:', view.zonePlateRadiusWidget)
        layout.addRow('Outermost Zone Width:', view.outermostZoneWidthWidget)
        layout.addRow('Beamstop Diameter:', view.beamstopDiameterWidget)
        layout.addRow('Defocus Distance:', view.defocusDistanceWidget)
        layout.addRow('Number of Modes:', view.numberOfModesSpinBox)
        view.setLayout(layout)

        return view


class SuperGaussianProbeDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = SuperGaussianProbeView.createInstance()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SuperGaussianProbeDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.addButton(QDialogButtonBox.Cancel)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class FresnelZonePlateProbeDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = FresnelZonePlateProbeView.createInstance()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> FresnelZonePlateProbeDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.addButton(QDialogButtonBox.Cancel)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ProbeModesButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.initializeMenu = QMenu()
        self.initializeButton = QPushButton('Initialize')
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeModesButtonBox:
        view = cls(parent)

        view.initializeButton.setMenu(view.initializeMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.initializeButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class ProbeModesView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Modes', parent)
        self.treeView = QTreeView()
        self.buttonBox = ProbeModesButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeModesView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ProbeView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = ProbeParametersView.createInstance()
        self.modesView = ProbeModesView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.modesView)
        view.setLayout(layout)

        return view
