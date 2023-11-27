from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLineEdit, QMenu, QPushButton, QSpinBox, QTableView,
                             QVBoxLayout, QWidget)

from .widgets import EnergyWidget, LengthWidget


class DetectorView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Detector', parent)
        self.numberOfPixelsXSpinBox = QSpinBox()
        self.numberOfPixelsYSpinBox = QSpinBox()
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()
        self.bitDepthSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> DetectorView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Pixels X:', view.numberOfPixelsXSpinBox)
        layout.addRow('Number of Pixels Y:', view.numberOfPixelsYSpinBox)
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        layout.addRow('Bit Depth:', view.bitDepthSpinBox)
        view.setLayout(layout)

        return view


class ExperimentEditorDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.nameLabel = QLineEdit()
        self.energyWidget = EnergyWidget.createInstance()
        self.detectorDistanceWidget = LengthWidget.createInstance()
        self.tableView = QTableView()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentEditorDialog:
        view = cls(parent)
        view.setWindowTitle('Edit Experiment')

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QFormLayout()
        layout.addRow('Name:', view.nameLabel)
        layout.addRow('Probe Energy:', view.energyWidget)
        layout.addRow('Detector-Object Distance:', view.detectorDistanceWidget)
        layout.addRow(view.tableView)
        layout.addRow(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ExperimentButtonBox(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.editButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class ExperimentRepositoryView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Repository', parent)
        self.tableView = QTableView()
        self.buttonBox = ExperimentButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentRepositoryView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ExperimentView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.detectorView = DetectorView.createInstance()
        self.repositoryView = ExperimentRepositoryView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorView)
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view
