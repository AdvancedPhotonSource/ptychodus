from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QMenu, QPushButton, QSpinBox, QTableView,
                             QVBoxLayout, QWidget)

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


class ExperimentInfoDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentInfoDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
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
        self.infoButton = QPushButton('Info')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.infoButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class ExperimentRepositoryView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Repository', parent)
        self.tableView = QTableView()
        self.infoLabel = QLabel()
        self.buttonBox = ExperimentButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ExperimentRepositoryView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.infoLabel)
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
