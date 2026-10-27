from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
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


class TrainDialog(QDialog):
    """Where a training run reads its data from and writes its model to.

    The two are separate directories because training writes checkpoints, a
    loss log and whatever else a backend emits, none of which belongs beside
    the data it learned from.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Train')
        self.data_line_edit = QLineEdit()
        self.data_browse_button = QPushButton('Browse...')
        self.output_line_edit = QLineEdit()
        self.output_browse_button = QPushButton('Browse...')
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        data_layout = QHBoxLayout()
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self.data_line_edit)
        data_layout.addWidget(self.data_browse_button)

        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(self.output_browse_button)

        form_layout = QFormLayout()
        form_layout.addRow('Training Data:', data_layout)
        form_layout.addRow('Model Output:', output_layout)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.button_box)
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
