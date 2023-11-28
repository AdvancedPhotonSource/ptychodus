from __future__ import annotations
from typing import Generic, Optional, TypeVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QRadioButton, QSpinBox,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider, LengthWidget

__all__ = [
    'ProbeEditorDialog',
]

T = TypeVar('T', bound=QGroupBox)


class DiskProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.diameterWidget = LengthWidget.createInstance()
        self.testPatternCheckBox = QCheckBox('Test Pattern')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DiskProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Diameter:', view.diameterWidget)
        layout.addWidget(view.testPatternCheckBox)
        view.setLayout(layout)

        return view


class SuperGaussianProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.annularRadiusWidget = LengthWidget.createInstance()
        self.fwhmWidget = LengthWidget.createInstance()
        self.orderParameterWidget = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SuperGaussianProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Annular Radius:', view.annularRadiusWidget)
        layout.addRow('Full Width at Half Maximum:', view.fwhmWidget)
        layout.addRow('Order Parameter:', view.orderParameterWidget)
        view.setLayout(layout)

        return view


class FresnelZonePlateProbeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.presetsComboBox = QComboBox()
        self.zonePlateDiameterWidget = LengthWidget.createInstance()
        self.outermostZoneWidthWidget = LengthWidget.createInstance()
        self.beamstopDiameterWidget = LengthWidget.createInstance()
        self.defocusDistanceWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> FresnelZonePlateProbeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Presets:', view.presetsComboBox)
        layout.addRow('Zone Plate Diameter:', view.zonePlateDiameterWidget)
        layout.addRow('Outermost Zone Width:', view.outermostZoneWidthWidget)
        layout.addRow('Beamstop Diameter:', view.beamstopDiameterWidget)
        layout.addRow('Defocus Distance:', view.defocusDistanceWidget)
        view.setLayout(layout)

        return view


class ProbeModesView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Additional Modes', parent)
        self.numberOfModesSpinBox = QSpinBox()
        self.orthogonalizeModesCheckBox = QCheckBox('Orthogonalize Modes')
        self.polynomialDecayButton = QRadioButton('Polynomial')
        self.exponentialDecayButton = QRadioButton('Exponential')
        self.decayRatioSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ProbeModesView:
        view = cls(parent)

        decayButtonBox = QHBoxLayout()
        decayButtonBox.addWidget(view.polynomialDecayButton)
        decayButtonBox.addWidget(view.exponentialDecayButton)

        layout = QFormLayout()
        layout.addRow('Number of Modes:', view.numberOfModesSpinBox)
        layout.addRow(view.orthogonalizeModesCheckBox)
        layout.addRow('Decay Type:', decayButtonBox)
        layout.addRow('Decay Ratio:', view.decayRatioSlider)
        view.setLayout(layout)

        return view


class ProbeEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.modesView = ProbeModesView.createInstance(parent)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       title: str,
                       editorView: T,
                       parent: Optional[QWidget] = None) -> ProbeEditorDialog[T]:
        view = cls(editorView, parent)
        view.setWindowTitle(title)
        editorView.setTitle('Primary Mode')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.modesView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
