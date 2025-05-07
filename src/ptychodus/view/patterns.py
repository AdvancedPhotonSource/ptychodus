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
        self._is_complete = False

    def isComplete(self) -> bool:  # noqa: N802
        """Overrides QWizardPage.isComplete()"""
        return self._is_complete

    def _set_complete(self, complete: bool) -> None:
        if self._is_complete != complete:
            self._is_complete = complete
            self.completeChanged.emit()


class OpenDatasetWizardMetadataPage(OpenDatasetWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.detector_extent_check_box = QCheckBox('Detector Extent')
        self.detector_pixel_size_check_box = QCheckBox('Detector Pixel Size')
        self.detector_bit_depth_check_box = QCheckBox('Detector Bit Depth')
        self.detector_distance_check_box = QCheckBox('Detector Distance')
        self.pattern_crop_center_check_box = QCheckBox('Pattern Crop Center')
        self.pattern_crop_extent_check_box = QCheckBox('Pattern Crop Extent')
        self.probe_photon_count_check_box = QCheckBox('Probe Photon Count')
        self.probe_energy_check_box = QCheckBox('Probe Energy')

        self.setTitle('Import Metadata')

        layout = QVBoxLayout()
        layout.addWidget(self.detector_extent_check_box)
        layout.addWidget(self.detector_pixel_size_check_box)
        layout.addWidget(self.detector_bit_depth_check_box)
        layout.addWidget(self.detector_distance_check_box)
        layout.addWidget(self.pattern_crop_center_check_box)
        layout.addWidget(self.pattern_crop_extent_check_box)
        layout.addWidget(self.probe_photon_count_check_box)
        layout.addWidget(self.probe_energy_check_box)
        layout.addStretch()
        self.setLayout(layout)


class PatternsInfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree_view = QTreeView()
        self.button_box = QDialogButtonBox()

        tree_header = self.tree_view.header()
        tree_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        tree_header.setSectionResizeMode(QHeaderView.ResizeToContents)

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.clicked.connect(self._handle_button_box_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def _handle_button_box_clicked(self, button: QAbstractButton) -> None:
        if self.button_box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class PatternsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.detector_view = DetectorView()
        self.tree_view = QTreeView()
        self.info_label = QLabel()
        self.button_box = PatternsButtonBox()

        self.tree_view.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.detector_view)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
