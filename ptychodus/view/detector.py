from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QFormLayout, QGroupBox, QListView, QSpinBox, QVBoxLayout, QWidget

from .widgets import DecimalLineEdit, LengthWidget


class DetectorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPixelsXSpinBox = QSpinBox()
        self.numberOfPixelsYSpinBox = QSpinBox()
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()
        self.detectorDistanceWidget = LengthWidget.createInstance()
        self.fresnelNumberWidget = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Pixels X:', view.numberOfPixelsXSpinBox)
        layout.addRow('Number of Pixels Y:', view.numberOfPixelsYSpinBox)
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        layout.addRow('Detector-Object Distance:', view.detectorDistanceWidget)
        layout.addRow('Fresnel Number:', view.fresnelNumberWidget)
        view.setLayout(layout)

        return view


class DiffractionPatternView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Patterns', parent)
        self.listView = QListView()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DiffractionPatternView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        view.setLayout(layout)

        return view


class DetectorParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.detectorView = DetectorView.createInstance()
        self.patternView = DiffractionPatternView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorView)
        layout.addWidget(view.patternView)
        view.setLayout(layout)

        return view
