from PyQt5.QtWidgets import QFormLayout, QGroupBox, QListView, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from .image import ImageView
from .widgets import LengthWidget


class DetectorDetectorView(QGroupBox):
    def __init__(self, parent: QWidget = None):
        super().__init__('Parameters', parent)
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()
        self.detectorDistanceWidget = LengthWidget.createInstance()
        self.defocusDistanceWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None):
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        layout.addRow('Detector-Object Distance:', view.detectorDistanceWidget)
        layout.addRow('Defocus Distance:', view.defocusDistanceWidget)
        view.setLayout(layout)

        return view


class DetectorDatasetView(QGroupBox):
    def __init__(self, parent: QWidget = None):
        super().__init__('Diffraction Data', parent)
        self.dataFileListView = QListView()

    @classmethod
    def createInstance(cls, parent: QWidget = None):
        view = cls(parent)

        view.dataFileListView.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.dataFileListView)
        view.setLayout(layout)

        return view


class DetectorImageCropView(QGroupBox):
    def __init__(self, parent: QWidget = None):
        super().__init__('Image Crop', parent)
        self.centerXSpinBox = QSpinBox()
        self.centerYSpinBox = QSpinBox()
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None):
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Center X [px]:', view.centerXSpinBox)
        layout.addRow('Center Y [px]:', view.centerYSpinBox)
        layout.addRow('Extent X [px]:', view.extentXSpinBox)
        layout.addRow('Extent Y [px]:', view.extentYSpinBox)
        view.setLayout(layout)

        return view


class DetectorParametersView(QWidget):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.detectorView = DetectorDetectorView.createInstance()
        self.datasetView = DetectorDatasetView.createInstance()
        self.imageCropView = DetectorImageCropView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None):
        view = cls(parent)

        view.datasetView.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorView)
        layout.addWidget(view.datasetView)
        layout.addWidget(view.imageCropView)
        view.setLayout(layout)

        return view

