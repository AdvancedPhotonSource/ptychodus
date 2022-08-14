from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QListView, QPushButton,
                             QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from .image import ImageView
from .widgets import LengthWidget


class DetectorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPixelsXSpinBox = QSpinBox()
        self.numberOfPixelsYSpinBox = QSpinBox()
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()
        self.detectorDistanceWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Pixels X:', view.numberOfPixelsXSpinBox)
        layout.addRow('Number of Pixels Y:', view.numberOfPixelsYSpinBox)
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        layout.addRow('Detector-Object Distance:', view.detectorDistanceWidget)
        view.setLayout(layout)

        return view


class DatasetButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DatasetButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.openButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class DatasetView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Data', parent)
        self.listView = QListView()
        self.buttonBox = DatasetButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DatasetView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class CropView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Image Crop', parent)
        self.centerXSpinBox = QSpinBox()
        self.centerYSpinBox = QSpinBox()
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CropView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Center X [px]:', view.centerXSpinBox)
        layout.addRow('Center Y [px]:', view.centerYSpinBox)
        layout.addRow('Extent X [px]:', view.extentXSpinBox)
        layout.addRow('Extent Y [px]:', view.extentYSpinBox)
        view.setLayout(layout)

        return view


class DetectorParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.detectorView = DetectorView.createInstance()
        self.imageCropView = CropView.createInstance()
        self.datasetView = DatasetView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DetectorParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorView)
        layout.addWidget(view.imageCropView)
        layout.addWidget(view.datasetView)
        view.setLayout(layout)

        return view
