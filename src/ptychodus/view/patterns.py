from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)


class DetectorView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Detector', parent)


class PatternsButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.open_button = QPushButton('Open')
        self.save_button = QPushButton('Save')
        self.info_button = QPushButton('Info')
        self.close_button = QPushButton('Close')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.info_button)
        layout.addWidget(self.close_button)
        self.setLayout(layout)


class OpenDatasetWizardPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._isComplete = False

    def isComplete(self) -> bool:
        """Overrides QWizardPage.isComplete()"""
        return self._isComplete

    def _setComplete(self, complete: bool) -> None:
        if self._isComplete != complete:
            self._isComplete = complete
            self.completeChanged.emit()


class OpenDatasetWizardMetadataPage(OpenDatasetWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.detectorExtentCheckBox = QCheckBox('Detector Extent')
        self.detectorPixelSizeCheckBox = QCheckBox('Detector Pixel Size')
        self.detectorBitDepthCheckBox = QCheckBox('Detector Bit Depth')
        self.detectorDistanceCheckBox = QCheckBox('Detector Distance')
        self.patternCropCenterCheckBox = QCheckBox('Pattern Crop Center')
        self.patternCropExtentCheckBox = QCheckBox('Pattern Crop Extent')
        self.probePhotonCountCheckBox = QCheckBox('Probe Photon Count')
        self.probeEnergyCheckBox = QCheckBox('Probe Energy')

        self.setTitle('Import Metadata')

        layout = QVBoxLayout()
        layout.addWidget(self.detectorExtentCheckBox)
        layout.addWidget(self.detectorPixelSizeCheckBox)
        layout.addWidget(self.detectorBitDepthCheckBox)
        layout.addWidget(self.detectorDistanceCheckBox)
        layout.addWidget(self.patternCropCenterCheckBox)
        layout.addWidget(self.patternCropExtentCheckBox)
        layout.addWidget(self.probePhotonCountCheckBox)
        layout.addWidget(self.probeEnergyCheckBox)
        layout.addStretch()
        self.setLayout(layout)


class PatternsInfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.treeView = QTreeView()
        self.buttonBox = QDialogButtonBox()

        treeHeader = self.treeView.header()
        treeHeader.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        treeHeader.setSectionResizeMode(QHeaderView.ResizeToContents)

        self.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.clicked.connect(self._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.treeView)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class PatternsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.detectorView = DetectorView()
        self.tree_view = QTreeView()
        self.infoLabel = QLabel()
        self.button_box = PatternsButtonBox()

        self.tree_view.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.detectorView)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
