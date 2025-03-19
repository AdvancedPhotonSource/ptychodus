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
        self.reconstructorButton = QPushButton('Reconstructor')
        self.reconstructorButton.setMenu(self.reconstructor_menu)

        self.trainer_menu = QMenu()
        self.trainer_button = QPushButton('Trainer')
        self.trainer_button.setMenu(self.trainer_menu)

        actionLayout = QHBoxLayout()
        actionLayout.setContentsMargins(0, 0, 0, 0)
        actionLayout.addWidget(self.reconstructorButton)
        actionLayout.addWidget(self.trainer_button)

        layout = QFormLayout()
        layout.addRow('Algorithm:', self.algorithm_combo_box)
        layout.addRow('Product:', self.product_combo_box)
        layout.addRow('Action:', actionLayout)
        self.setLayout(layout)


class ReconstructorProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text_edit = QPlainTextEdit()
        self.progressBar = QProgressBar()
        self.button_box = QDialogButtonBox()

        self.setWindowTitle('Reconstruction Progress')
        self.button_box.addButton(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.Cancel)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class ReconstructorView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = ReconstructorParametersView()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.stacked_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

        self.progress_dialog = ReconstructorProgressDialog()


class ReconstructorPlotView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.navigationToolbar)
        layout.addWidget(self.figureCanvas)
        self.setLayout(layout)
