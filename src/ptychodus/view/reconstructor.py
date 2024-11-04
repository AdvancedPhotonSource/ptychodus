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
        self.algorithmComboBox = QComboBox()
        self.productComboBox = QComboBox()

        self.modelMenu = QMenu()
        self.modelButton = QPushButton('Model')
        self.modelButton.setMenu(self.modelMenu)

        self.trainerMenu = QMenu()
        self.trainerButton = QPushButton('Trainer')
        self.trainerButton.setMenu(self.trainerMenu)

        self.reconstructorMenu = QMenu()
        self.reconstructorButton = QPushButton('Reconstructor')
        self.reconstructorButton.setMenu(self.reconstructorMenu)

        actionLayout = QHBoxLayout()
        actionLayout.setContentsMargins(0, 0, 0, 0)
        actionLayout.addWidget(self.modelButton)
        actionLayout.addWidget(self.trainerButton)
        actionLayout.addWidget(self.reconstructorButton)

        layout = QFormLayout()
        layout.addRow('Algorithm:', self.algorithmComboBox)
        layout.addRow('Product:', self.productComboBox)
        layout.addRow('Action:', actionLayout)
        self.setLayout(layout)


class ReconstructorProgressDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.textEdit = QPlainTextEdit()
        self.progressBar = QProgressBar()
        self.buttonBox = QDialogButtonBox()

        self.setWindowTitle('Reconstruction Progress')
        self.buttonBox.addButton(QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.addButton(QDialogButtonBox.Cancel)
        self.buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.textEdit)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class ReconstructorView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parametersView = ReconstructorParametersView()

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.layout().setContentsMargins(0, 0, 0, 0)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.stackedWidget)

        layout = QVBoxLayout()
        layout.addWidget(self.parametersView)
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

        self.progressDialog = ReconstructorProgressDialog()


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
