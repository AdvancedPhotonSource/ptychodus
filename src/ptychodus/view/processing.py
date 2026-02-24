from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ProcessingActionsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reconstruct_button = QPushButton()
        self.train_button = QPushButton('Train')
        self.actions_menu = QMenu()
        self.actions_button = QToolButton()

        self.actions_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.actions_button.setMenu(self.actions_menu)
        self.actions_button.setArrowType(Qt.ArrowType.DownArrow)
        self.actions_button.setStyleSheet('QToolButton::menu-indicator { image: none; }')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.reconstruct_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.actions_button)
        self.setLayout(layout)


class ProcessingStatusView(QSplitter):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Vertical, parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)

        self.text_edit = QPlainTextEdit()
        self.progress_bar = QProgressBar()
        self.stop_button = QPushButton('Stop')

        upper_layout = QVBoxLayout()
        upper_layout.addWidget(self.navigation_toolbar)
        upper_layout.addWidget(self.figure_canvas)

        upper_widget = QWidget()
        upper_widget.setLayout(upper_layout)
        self.addWidget(upper_widget)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stop_button)

        lower_layout = QVBoxLayout()
        lower_layout.addWidget(self.text_edit)
        lower_layout.addLayout(progress_layout)

        lower_widget = QWidget()
        lower_widget.setLayout(lower_layout)
        self.addWidget(lower_widget)

        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 0)
