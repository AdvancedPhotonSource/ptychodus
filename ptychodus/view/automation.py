from __future__ import annotations

from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QListView, QPushButton, QSpinBox, QVBoxLayout, QWidget)


class AutomationParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.strategyLabel = QLabel('Strategy:')
        self.strategyComboBox = QComboBox()
        self.directoryLabel = QLabel('Directory:')
        self.directoryLineEdit = QLineEdit()
        self.directoryBrowseButton = QPushButton('Browse')
        self.intervalLabel = QLabel('Interval [sec]:')
        self.intervalSpinBox = QSpinBox()
        self.executeButton = QPushButton('Execute')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.strategyLabel, 0, 0)
        layout.addWidget(view.strategyComboBox, 0, 1, 1, 2)
        layout.addWidget(view.directoryLabel, 1, 0)
        layout.addWidget(view.directoryLineEdit, 1, 1)
        layout.addWidget(view.directoryBrowseButton, 1, 2)
        layout.addWidget(view.intervalLabel, 2, 0)
        layout.addWidget(view.intervalSpinBox, 2, 1, 1, 2)
        layout.addWidget(view.executeButton, 4, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class AutomationWatchdogView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Watchdog', parent)
        self.delaySpinBox = QSpinBox()
        self.usePollingObserverCheckBox = QCheckBox('Use Polling Observer')
        self.watchButton = QPushButton('Watch')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationWatchdogView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Delay [sec]:', view.delaySpinBox)
        layout.addRow(view.usePollingObserverCheckBox)
        layout.addRow(view.watchButton)
        view.setLayout(layout)

        return view


class AutomationProcessingView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Processing', parent)
        self.listView = QListView()
        self.processButton = QPushButton('Process')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationProcessingView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        layout.addWidget(view.processButton)
        view.setLayout(layout)

        return view


class AutomationView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.parametersView = AutomationParametersView.createInstance()
        self.watchdogView = AutomationWatchdogView.createInstance()
        self.processingView = AutomationProcessingView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> AutomationView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.watchdogView)
        layout.addWidget(view.processingView)
        view.setLayout(layout)

        return view
