from __future__ import annotations

from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListView, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)


class AutomationProcessingView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Processing', parent)
        self.strategyLabel = QLabel('Strategy:')
        self.strategyComboBox = QComboBox()
        self.directoryLabel = QLabel('Directory:')
        self.directoryLineEdit = QLineEdit()
        self.directoryBrowseButton = QPushButton('Browse')
        self.intervalLabel = QLabel('Interval [sec]:')
        self.intervalSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationProcessingView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.strategyLabel, 0, 0)
        layout.addWidget(view.strategyComboBox, 0, 1, 1, 2)
        layout.addWidget(view.directoryLabel, 1, 0)
        layout.addWidget(view.directoryLineEdit, 1, 1)
        layout.addWidget(view.directoryBrowseButton, 1, 2)
        layout.addWidget(view.intervalLabel, 2, 0)
        layout.addWidget(view.intervalSpinBox, 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class AutomationWatchdogView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Watchdog', parent)
        self.delaySpinBox = QSpinBox()
        self.usePollingObserverCheckBox = QCheckBox('Use Polling Observer')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationWatchdogView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Delay [sec]:', view.delaySpinBox)
        layout.addRow(view.usePollingObserverCheckBox)
        view.setLayout(layout)

        return view


class AutomationView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.processingView = AutomationProcessingView.createInstance()
        self.watchdogView = AutomationWatchdogView.createInstance()
        self.processingListView = QListView()
        self.loadButton = QPushButton('Load')
        self.watchButton = QPushButton('Watch')
        self.processButton = QPushButton('Process')
        self.clearButton = QPushButton('Clear')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationView:
        view = cls(parent)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(view.loadButton)
        buttonLayout.addWidget(view.watchButton)
        buttonLayout.addWidget(view.processButton)
        buttonLayout.addWidget(view.clearButton)

        layout = QVBoxLayout()
        layout.addWidget(view.processingView)
        layout.addWidget(view.watchdogView)
        layout.addWidget(view.processingListView)
        layout.addLayout(buttonLayout)
        view.setLayout(layout)

        return view
