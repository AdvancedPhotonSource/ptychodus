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
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ReconstructorParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.algorithm_combo_box = QComboBox()
        self.product_combo_box = QComboBox()

        self.reconstructor_menu = QMenu()
        self.reconstructor_button = QPushButton('Reconstructor')
        self.reconstructor_button.setMenu(self.reconstructor_menu)

        self.trainer_menu = QMenu()
        self.trainer_button = QPushButton('Trainer')
        self.trainer_button.setMenu(self.trainer_menu)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self.reconstructor_button)
        action_layout.addWidget(self.trainer_button)

        layout = QFormLayout()
        layout.addRow('Algorithm:', self.algorithm_combo_box)
        layout.addRow('Product:', self.product_combo_box)
        layout.addRow('Action:', action_layout)
        self.setLayout(layout)


class ReconstructorProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text_edit = QPlainTextEdit()
        self.progress_bar = QProgressBar()
        self.button_box = QDialogButtonBox()

        self.setWindowTitle('Reconstruction Progress')
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class ReconstructorView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = ReconstructorParametersView()

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

        self.progress_dialog = ReconstructorProgressDialog()


class ReconstructorPlotView(QWidget):
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
