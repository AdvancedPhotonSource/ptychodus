from __future__ import annotations
from typing import Generic, Optional, TypeVar

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                             QSpinBox, QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, EnergyWidget, LengthWidget, RepositoryTreeView

__all__ = [
    'ProbeEditorDialog',
    'ProbeParametersView',
    'ProbeView',
]

T = TypeVar('T', bound=QGroupBox)


class ProbeParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.energyWidget = EnergyWidget.createInstance()
        self.wavelengthWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Energy:', view.energyWidget)
        layout.addRow('Wavelength:', view.wavelengthWidget)
        view.setLayout(layout)

        return view


class DiskProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.numberOfModesSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DiskProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Modes:', view.numberOfModesSpinBox)
        view.setLayout(layout)

        return view


class SuperGaussianProbeView(QGroupBox):

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


class FresnelZonePlateProbeView(QGroupBox):

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


class ProbeEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       editorView: T,
                       parent: Optional[QWidget] = None) -> ProbeEditorDialog[T]:
        view = cls(editorView, parent)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ProbeView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = ProbeParametersView.createInstance()
        self.repositoryView = RepositoryTreeView.createInstance('Repository')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view
