from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ProcessingParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.algorithm_combo_box = QComboBox()
        self.product_combo_box = QComboBox()
        self.compute_local_radio_button = QRadioButton('Local')
        self.compute_remote_radio_button = QRadioButton('Remote')
        self.reconstruct_button = QPushButton()
        self.reconstruct_tools_menu = QMenu()
        self.reconstruct_tools_button = QToolButton()
        self.train_button = QPushButton('Train')
        self.train_tools_menu = QMenu()
        self.train_tools_button = QToolButton()

        self.reconstruct_tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.reconstruct_tools_button.setMenu(self.reconstruct_tools_menu)
        self.reconstruct_tools_button.setArrowType(Qt.ArrowType.DownArrow)

        self.train_tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.train_tools_button.setMenu(self.train_tools_menu)
        self.train_tools_button.setArrowType(Qt.ArrowType.DownArrow)

        compute_layout = QHBoxLayout()
        compute_layout.addWidget(self.compute_local_radio_button)
        compute_layout.addWidget(self.compute_remote_radio_button)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self.reconstruct_button)
        action_layout.addWidget(self.reconstruct_tools_button)
        action_layout.addWidget(self.train_button)
        action_layout.addWidget(self.train_tools_button)

        layout = QFormLayout()
        layout.addRow('Algorithm:', self.algorithm_combo_box)
        layout.addRow('Product:', self.product_combo_box)
        layout.addRow('Compute:', compute_layout)
        layout.addRow('Action:', action_layout)
        self.setLayout(layout)


class ProcessingProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text_edit = QPlainTextEdit()
        self.progress_bar = QProgressBar()
        self.button_box = QDialogButtonBox()

        self.setWindowTitle('Processing Progress')
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class ProcessingView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = ProcessingParametersView()
        self.stacked_widget = QStackedWidget()

        stacked_widget_layout = self.stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.stacked_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        self.progress_dialog = ProcessingProgressDialog()


class ProcessingStatusView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.navigation_toolbar)
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)
