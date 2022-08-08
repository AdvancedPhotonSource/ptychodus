from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
                             QStackedWidget, QVBoxLayout, QWidget)

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WorkflowAuthorizeDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.lineEdit = QLineEdit()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowAuthorizeDialog:
        view = cls(parent)

        view.setWindowTitle('Authorize Workflow')
        view.label.setTextFormat(Qt.RichText)
        view.label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        view.label.setOpenExternalLinks(True)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.addButton(QDialogButtonBox.Cancel)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.label)
        layout.addWidget(view.lineEdit)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class WorkflowDataView(QGroupBox):

    def __init__(self, title: str, parent: Optional[QWidget]) -> None:
        super().__init__(title, parent)
        self.endpointIDLineEdit = QLineEdit()
        self.pathLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, title: str, parent: Optional[QWidget] = None) -> WorkflowDataView:
        view = cls(title, parent)

        layout = QFormLayout()
        layout.addRow('Endpoint ID:', view.endpointIDLineEdit)
        layout.addRow('Data Path:', view.pathLineEdit)
        view.setLayout(layout)

        return view


class WorkflowComputeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Compute', parent)
        self.endpointIDLineEdit = QLineEdit()
        self.flowIDLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowComputeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Endpoint ID:', view.endpointIDLineEdit)
        layout.addRow('Flow ID:', view.flowIDLineEdit)
        view.setLayout(layout)

        return view


class WorkflowButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.authorizeButton = QPushButton('Authorize')
        self.listFlowsButton = QPushButton('List Flows')
        self.listFlowRunsButton = QPushButton('List Flow Runs')
        self.deployFlowButton = QPushButton('Deploy Flow')
        self.runFlowButton = QPushButton('Run Flow')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.authorizeButton)
        layout.addWidget(view.listFlowsButton)
        layout.addWidget(view.listFlowRunsButton)
        layout.addWidget(view.deployFlowButton)
        layout.addWidget(view.runFlowButton)
        view.setLayout(layout)

        return view


class WorkflowParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.dataSourceView = WorkflowDataView.createInstance('Data Source')
        self.dataDestinationView = WorkflowDataView.createInstance('Data Destination')
        self.computeView = WorkflowComputeView.createInstance()
        self.buttonBox = WorkflowButtonBox.createInstance()
        self.authorizeDialog = WorkflowAuthorizeDialog.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.dataSourceView)
        layout.addWidget(view.dataDestinationView)
        layout.addWidget(view.computeView)
        layout.addWidget(view.buttonBox)
        layout.addStretch()
        view.setLayout(layout)

        return view
