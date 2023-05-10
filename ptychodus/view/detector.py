from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QLabel, QSpinBox, QTreeView, QVBoxLayout,
                             QWidget)

from .widgets import DecimalLineEdit, LengthWidget


class DetectorParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPixelsXSpinBox = QSpinBox()
        self.numberOfPixelsYSpinBox = QSpinBox()
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()
        self.detectorDistanceWidget = LengthWidget.createInstance()
        self.fresnelNumberWidget = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorParametersView:
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


class DetectorDataViewView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Patterns', parent)
        self.treeView = QTreeView()
        self.infoLabel = QLabel()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorDataViewView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.infoLabel)
        view.setLayout(layout)

        return view


class DetectorView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = DetectorParametersView.createInstance()
        self.dataView = DetectorDataViewView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.dataView)
        view.setLayout(layout)

        return view
