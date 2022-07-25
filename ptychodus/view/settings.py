from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QDialog, QDialogButtonBox, QGroupBox,
                             QHBoxLayout, QLabel, QListView, QPushButton, QVBoxLayout, QWidget)


class SettingsButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SettingsButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.openButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class SettingsGroupView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Groups', parent)
        self.listView = QListView()
        self.buttonBox = SettingsButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SettingsGroupView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class SettingsParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.groupView = SettingsGroupView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SettingsParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.groupView)
        view.setLayout(layout)

        return view


class ImportSettingsValuesGroupBox(QGroupBox):

    @staticmethod
    def createCheckBox(text: str) -> QCheckBox:
        widget = QCheckBox(text)
        widget.setChecked(True)
        return widget

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Choose Settings', parent)
        self.detectorPixelCountCheckBox = self.createCheckBox('Detector Pixel Count')
        self.detectorPixelSizeCheckBox = self.createCheckBox('Detector Pixel Size')
        self.detectorDistanceCheckBox = self.createCheckBox('Detector Distance')
        self.imageCropCenterCheckBox = self.createCheckBox('Image Crop Center')
        self.imageCropExtentCheckBox = self.createCheckBox('Image Crop Extent')
        self.probeEnergyCheckBox = self.createCheckBox('Probe Energy')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImportSettingsValuesGroupBox:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorPixelCountCheckBox)
        layout.addWidget(view.detectorPixelSizeCheckBox)
        layout.addWidget(view.detectorDistanceCheckBox)
        layout.addWidget(view.imageCropCenterCheckBox)
        layout.addWidget(view.imageCropExtentCheckBox)
        layout.addWidget(view.probeEnergyCheckBox)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ImportSettingsOptionsGroupBox(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Additional Options', parent)
        self.loadScanCheckBox = QCheckBox('Load Scan')
        self.reinitializeProbeCheckBox = QCheckBox('Reinitialize Probe')
        self.reinitializeObjectCheckBox = QCheckBox('Reinitialize Object')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImportSettingsOptionsGroupBox:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.loadScanCheckBox)
        layout.addWidget(view.reinitializeProbeCheckBox)
        layout.addWidget(view.reinitializeObjectCheckBox)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ImportSettingsDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.valuesGroupBox = ImportSettingsValuesGroupBox.createInstance()
        self.optionsGroupBox = ImportSettingsOptionsGroupBox.createInstance()
        self.centerWidget = QWidget()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImportSettingsDialog:
        view = cls(parent)

        view.setWindowTitle('Import Settings')

        centerLayout = QHBoxLayout()
        centerLayout.addWidget(view.valuesGroupBox)
        centerLayout.addWidget(view.optionsGroupBox)
        view.centerWidget.setLayout(centerLayout)

        view.buttonBox.addButton(QDialogButtonBox.Apply)
        view.buttonBox.addButton(QDialogButtonBox.Cancel)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.centerWidget)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ApplyRole:
            self.accept()
        else:
            self.reject()
