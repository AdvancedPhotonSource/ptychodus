from __future__ import annotations

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


class ReconstructorView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__("Parameters", parent)
        self.algorithmComboBox = QComboBox()
        self.productComboBox = QComboBox()
        self.modelButton = QPushButton("Model")
        self.modelMenu = QMenu()
        self.trainerButton = QPushButton("Trainer")
        self.trainerMenu = QMenu()
        self.reconstructorButton = QPushButton("Reconstructor")
        self.reconstructorMenu = QMenu()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ReconstructorView:
        view = cls(parent)

        view.modelButton.setMenu(view.modelMenu)
        view.trainerButton.setMenu(view.trainerMenu)
        view.reconstructorButton.setMenu(view.reconstructorMenu)

        actionLayout = QHBoxLayout()
        actionLayout.setContentsMargins(0, 0, 0, 0)
        actionLayout.addWidget(view.modelButton)
        actionLayout.addWidget(view.trainerButton)
        actionLayout.addWidget(view.reconstructorButton)

        layout = QFormLayout()
        layout.addRow("Algorithm:", view.algorithmComboBox)
        layout.addRow("Product:", view.productComboBox)
        layout.addRow("Action:", actionLayout)
        view.setLayout(layout)

        return view


class ReconstructorProgressDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.textEdit = QPlainTextEdit()
        self.progressBar = QProgressBar()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ReconstructorProgressDialog:
        dialog = cls(parent)
        dialog.setWindowTitle("Reconstruction Progress")
        dialog.buttonBox.addButton(QDialogButtonBox.Ok)
        dialog.buttonBox.accepted.connect(dialog.accept)
        dialog.buttonBox.addButton(QDialogButtonBox.Cancel)
        dialog.buttonBox.rejected.connect(dialog.reject)

        layout = QVBoxLayout()
        layout.addWidget(dialog.textEdit)
        layout.addWidget(dialog.progressBar)
        layout.addWidget(dialog.buttonBox)
        dialog.setLayout(layout)

        return dialog


class ReconstructorParametersView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.reconstructorView = ReconstructorView.createInstance()
        self.stackedWidget = QStackedWidget()
        self.scrollArea = QScrollArea()
        self.progressDialog = ReconstructorProgressDialog.createInstance()  # TODO use this

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ReconstructorParametersView:
        view = cls(parent)

        view.scrollArea.setWidgetResizable(True)
        view.scrollArea.setWidget(view.stackedWidget)

        view.stackedWidget.layout().setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.addWidget(view.reconstructorView)
        layout.addWidget(view.scrollArea)
        view.setLayout(layout)

        return view


class ReconstructorPlotView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ReconstructorPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
