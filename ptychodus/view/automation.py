from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QLabel, QLineEdit, QListView,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)


class AutomationWatchdogView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Watchdog', parent)
        self.strategyLabel = QLabel('Strategy:')
        self.strategyComboBox = QComboBox()
        self.directoryLabel = QLabel('Directory:')
        self.directoryLineEdit = QLineEdit()
        self.directoryBrowseButton = QPushButton('Browse')
        self.delayLabel = QLabel('Delay [sec]:')
        self.delaySpinBox = QSpinBox()
        self.watchButton = QPushButton('Watch')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> AutomationWatchdogView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.strategyLabel, 0, 0)
        layout.addWidget(view.strategyComboBox, 0, 1, 1, 2)
        layout.addWidget(view.directoryLabel, 1, 0)
        layout.addWidget(view.directoryLineEdit, 1, 1)
        layout.addWidget(view.directoryBrowseButton, 1, 2)
        layout.addWidget(view.delayLabel, 2, 0)
        layout.addWidget(view.delaySpinBox, 2, 1, 1, 2)
        layout.addWidget(view.watchButton, 3, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class AutomationDatasetsView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Datasets', parent)
        self.listView = QListView()
        self.processButton = QPushButton('Process')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> AutomationDatasetsView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        layout.addWidget(view.processButton)
        view.setLayout(layout)

        return view


class AutomationParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.watchdogView = AutomationWatchdogView.createInstance()
        self.datasetsView = AutomationDatasetsView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> AutomationParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.watchdogView)
        layout.addWidget(view.datasetsView)
        view.setLayout(layout)

        return view
