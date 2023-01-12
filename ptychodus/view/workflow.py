from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import UUIDLineEdit


class WorkflowAuthorizationDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.lineEdit = QLineEdit()
        self.buttonBox = QDialogButtonBox()
        self.okButton = self.buttonBox.addButton(QDialogButtonBox.Ok)
        self.cancelButton = self.buttonBox.addButton(QDialogButtonBox.Cancel)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowAuthorizationDialog:
        view = cls(parent)

        view.setWindowTitle('Authorize Workflow')
        view.label.setTextFormat(Qt.RichText)
        view.label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        view.label.setOpenExternalLinks(True)

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
        self.endpointIDLineEdit = UUIDLineEdit()
        self.globusPathLineEdit = QLineEdit()
        self.posixPathLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, title: str, parent: Optional[QWidget] = None) -> WorkflowDataView:
        view = cls(title, parent)

        layout = QFormLayout()
        layout.addRow('Endpoint ID:', view.endpointIDLineEdit)
        layout.addRow('Globus Path:', view.globusPathLineEdit)
        layout.addRow('POSIX Path:', view.posixPathLineEdit)
        view.setLayout(layout)

        return view


class WorkflowComputeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Compute', parent)
        self.funcXEndpointIDLineEdit = UUIDLineEdit()
        self.dataEndpointIDLineEdit = UUIDLineEdit()
        self.dataGlobusPathLineEdit = QLineEdit()
        self.dataPosixPathLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowComputeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('FuncX Endpoint ID:', view.funcXEndpointIDLineEdit)
        layout.addRow('Data Endpoint ID:', view.dataEndpointIDLineEdit)
        layout.addRow('Data Globus Path:', view.dataGlobusPathLineEdit)
        layout.addRow('Data POSIX Path:', view.dataPosixPathLineEdit)
        view.setLayout(layout)

        return view


class WorkflowExecutionView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Execution', parent)
        self.labelLineEdit = QLineEdit('Ptychodus')
        self.inputDataView = WorkflowDataView.createInstance('Input Data')
        self.computeView = WorkflowComputeView.createInstance()
        self.outputDataView = WorkflowDataView.createInstance('Output Data')
        self.executeButton = QPushButton('Execute')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowExecutionView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Label:', view.labelLineEdit)
        layout.addRow(view.inputDataView)
        layout.addRow(view.computeView)
        layout.addRow(view.outputDataView)
        layout.addRow(view.executeButton)
        view.setLayout(layout)

        return view


class WorkflowStatusView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Status', parent)
        self.autoRefreshCheckBox = QCheckBox('Auto Refresh [sec]:')
        self.autoRefreshSpinBox = QSpinBox()
        self.refreshButton = QPushButton('Refresh')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowStatusView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.autoRefreshCheckBox, view.autoRefreshSpinBox)
        layout.addRow(view.refreshButton)
        view.setLayout(layout)

        return view


class WorkflowParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.executionView = WorkflowExecutionView.createInstance()
        self.statusView = WorkflowStatusView.createInstance()
        self.authorizationDialog = WorkflowAuthorizationDialog.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.executionView)
        layout.addWidget(view.statusView)
        layout.addStretch()
        view.setLayout(layout)

        return view
