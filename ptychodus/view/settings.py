from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QListView,
                             QPushButton, QVBoxLayout, QWidget)


class SettingsView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.replacementPathPrefixLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SettingsView:
        view = cls(parent)

        view.replacementPathPrefixLineEdit.setToolTip(
            'Path prefix replacement text used when opening or saving settings files.')

        layout = QFormLayout()
        layout.addRow('Replacement Path Prefix:', view.replacementPathPrefixLineEdit)
        view.setLayout(layout)

        return view


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
        self.settingsView = SettingsView.createInstance()
        self.groupView = SettingsGroupView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SettingsParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.settingsView)
        layout.addWidget(view.groupView)
        view.setLayout(layout)

        return view
