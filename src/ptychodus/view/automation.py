from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class AutomationProcessingView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Processing', parent)
        self.strategy_label = QLabel('Strategy:')
        self.strategy_combo_box = QComboBox()
        self.directory_label = QLabel('Directory:')
        self.directory_line_edit = QLineEdit()
        self.directory_browse_button = QPushButton('Browse')
        self.interval_label = QLabel('Interval [sec]:')
        self.interval_spin_box = QSpinBox()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> AutomationProcessingView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.strategy_label, 0, 0)
        layout.addWidget(view.strategy_combo_box, 0, 1, 1, 2)
        layout.addWidget(view.directory_label, 1, 0)
        layout.addWidget(view.directory_line_edit, 1, 1)
        layout.addWidget(view.directory_browse_button, 1, 2)
        layout.addWidget(view.interval_label, 2, 0)
        layout.addWidget(view.interval_spin_box, 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class AutomationWatchdogView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Watchdog', parent)
        self.delay_spin_box = QSpinBox()
        self.use_polling_observer_check_box = QCheckBox('Use Polling Observer')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> AutomationWatchdogView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Delay [sec]:', view.delay_spin_box)
        layout.addRow(view.use_polling_observer_check_box)
        view.setLayout(layout)

        return view


class AutomationView(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.processing_view = AutomationProcessingView.create_instance()
        self.watchdog_view = AutomationWatchdogView.create_instance()
        self.processing_list_view = QListView()
        self.load_button = QPushButton('Load')
        self.watch_button = QPushButton('Watch')
        self.process_button = QPushButton('Process')
        self.clear_button = QPushButton('Clear')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> AutomationView:
        view = cls(parent)

        button_layout = QHBoxLayout()
        button_layout.addWidget(view.load_button)
        button_layout.addWidget(view.watch_button)
        button_layout.addWidget(view.process_button)
        button_layout.addWidget(view.clear_button)

        layout = QVBoxLayout()
        layout.addWidget(view.processing_view)
        layout.addWidget(view.watchdog_view)
        layout.addWidget(view.processing_list_view)
        layout.addLayout(button_layout)
        view.setLayout(layout)

        return view
