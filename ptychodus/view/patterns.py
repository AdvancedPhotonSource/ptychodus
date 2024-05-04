from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QTableView, QTreeView, QVBoxLayout, QWidget,
                             QWizard, QWizardPage)

from .widgets import LengthWidget


class DetectorView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Detector', parent)
        self.widthInPixelsSpinBox = QSpinBox()
        self.heightInPixelsSpinBox = QSpinBox()
        self.pixelWidthWidget = LengthWidget.createInstance()
        self.pixelHeightWidget = LengthWidget.createInstance()
        self.bitDepthSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> DetectorView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', view.widthInPixelsSpinBox)
        layout.addRow('Detector Height [px]:', view.heightInPixelsSpinBox)
        layout.addRow('Pixel Width:', view.pixelWidthWidget)
        layout.addRow('Pixel Height:', view.pixelHeightWidget)
        layout.addRow('Bit Depth:', view.bitDepthSpinBox)
        view.setLayout(layout)

        return view


class PatternsButtonBox(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')
        self.infoButton = QPushButton('Info')
        self.closeButton = QPushButton('Close')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PatternsButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.openButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.infoButton)
        layout.addWidget(view.closeButton)
        view.setLayout(layout)

        return view


class OpenDatasetWizardPage(QWizardPage):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._isComplete = False

    def isComplete(self) -> bool:
        return self._isComplete

    def _setComplete(self, complete: bool) -> None:
        if self._isComplete != complete:
            self._isComplete = complete
            self.completeChanged.emit()


class OpenDatasetWizardFilesPage(OpenDatasetWizardPage):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.directoryComboBox = QComboBox()
        self.fileSystemTableView = QTableView()
        self.fileTypeComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardFilesPage:
        view = cls(parent)
        view.setTitle('Choose File(s)')

        layout = QVBoxLayout()
        layout.addWidget(view.directoryComboBox)
        layout.addWidget(view.fileSystemTableView)
        layout.addWidget(view.fileTypeComboBox)
        view.setLayout(layout)

        return view


class OpenDatasetWizardMetadataPage(OpenDatasetWizardPage):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.detectorPixelCountCheckBox = QCheckBox('Detector Pixel Count')
        self.detectorPixelSizeCheckBox = QCheckBox('Detector Pixel Size')
        self.detectorBitDepthCheckBox = QCheckBox('Detector Bit Depth')
        self.detectorDistanceCheckBox = QCheckBox('Detector Distance')
        self.patternCropCenterCheckBox = QCheckBox('Pattern Crop Center')
        self.patternCropExtentCheckBox = QCheckBox('Pattern Crop Extent')
        self.probeEnergyCheckBox = QCheckBox('Probe Energy')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardMetadataPage:
        view = cls(parent)
        view.setTitle('Import Metadata')

        layout = QVBoxLayout()
        layout.addWidget(view.detectorPixelCountCheckBox)
        layout.addWidget(view.detectorPixelSizeCheckBox)
        layout.addWidget(view.detectorBitDepthCheckBox)
        layout.addWidget(view.detectorDistanceCheckBox)
        layout.addWidget(view.patternCropCenterCheckBox)
        layout.addWidget(view.patternCropExtentCheckBox)
        layout.addWidget(view.probeEnergyCheckBox)
        layout.addStretch()
        view.setLayout(layout)

        return view


class OpenDatasetWizardPatternLoadView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Load', parent)
        self.numberOfThreadsSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardPatternLoadView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', view.numberOfThreadsSpinBox)
        view.setLayout(layout)

        return view


class OpenDatasetWizardPatternCropView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Crop', parent)
        self.centerLabel = QLabel('Center [px]:')
        self.centerXSpinBox = QSpinBox()
        self.centerYSpinBox = QSpinBox()
        self.extentLabel = QLabel('Extent [px]:')
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardPatternCropView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.centerLabel, 0, 0)
        layout.addWidget(view.centerXSpinBox, 0, 1)
        layout.addWidget(view.centerYSpinBox, 0, 2)
        layout.addWidget(view.extentLabel, 1, 0)
        layout.addWidget(view.extentXSpinBox, 1, 1)
        layout.addWidget(view.extentYSpinBox, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        view.setLayout(layout)

        return view


class OpenDatasetWizardPatternTransformView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Transform', parent)
        self.valueLowerBoundCheckBox = QCheckBox('Value Lower Bound:')
        self.valueLowerBoundSpinBox = QSpinBox()
        self.valueUpperBoundCheckBox = QCheckBox('Value Upper Bound:')
        self.valueUpperBoundSpinBox = QSpinBox()
        self.axesLabel = QLabel('Axes:')
        self.flipXCheckBox = QCheckBox('Flip X')
        self.flipYCheckBox = QCheckBox('Flip Y')

    @classmethod
    def createInstance(cls,
                       parent: QWidget | None = None) -> OpenDatasetWizardPatternTransformView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.valueLowerBoundCheckBox, 0, 0)
        layout.addWidget(view.valueLowerBoundSpinBox, 0, 1, 1, 2)
        layout.addWidget(view.valueUpperBoundCheckBox, 1, 0)
        layout.addWidget(view.valueUpperBoundSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.axesLabel, 2, 0)
        layout.addWidget(view.flipXCheckBox, 2, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(view.flipYCheckBox, 2, 2, Qt.AlignmentFlag.AlignHCenter)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view


class OpenDatasetWizardPatternMemoryMapView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Memory Map Diffraction Data', parent)
        self.scratchDirectoryLabel = QLabel('Scratch Directory:')
        self.scratchDirectoryLineEdit = QLineEdit()
        self.scratchDirectoryBrowseButton = QPushButton('Browse')

    @classmethod
    def createInstance(cls,
                       parent: QWidget | None = None) -> OpenDatasetWizardPatternMemoryMapView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.scratchDirectoryLabel, 1, 0)
        layout.addWidget(view.scratchDirectoryLineEdit, 1, 1)
        layout.addWidget(view.scratchDirectoryBrowseButton, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class OpenDatasetWizardPatternsPage(OpenDatasetWizardPage):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.loadView = OpenDatasetWizardPatternLoadView.createInstance()
        self.memoryMapView = OpenDatasetWizardPatternMemoryMapView.createInstance()
        self.cropView = OpenDatasetWizardPatternCropView.createInstance()
        self.transformView = OpenDatasetWizardPatternTransformView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardPatternsPage:
        view = cls(parent)
        view.setTitle('Pattern Processing')

        layout = QVBoxLayout()
        layout.addWidget(view.loadView)
        layout.addWidget(view.memoryMapView)
        layout.addWidget(view.cropView)
        layout.addWidget(view.transformView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class OpenDatasetWizard(QWizard):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.filesPage = OpenDatasetWizardFilesPage.createInstance()
        self.metadataPage = OpenDatasetWizardMetadataPage.createInstance()
        self.patternsPage = OpenDatasetWizardPatternsPage.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizard:
        view = cls(parent)

        view.setWindowTitle('Open Dataset')
        view.addPage(view.filesPage)
        view.addPage(view.metadataPage)
        view.addPage(view.patternsPage)

        return view


class PatternsInfoDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.treeView = QTreeView()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PatternsInfoDialog:
        view = cls(parent)
        view.treeView.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class PatternsView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.detectorView = DetectorView.createInstance()
        self.treeView = QTreeView()
        self.infoLabel = QLabel()
        self.buttonBox = PatternsButtonBox.createInstance()
        self.openDatasetWizard = OpenDatasetWizard.createInstance(self)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PatternsView:
        view = cls(parent)
        view.treeView.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorView)
        layout.addWidget(view.treeView)
        layout.addWidget(view.infoLabel)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view
