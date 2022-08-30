from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
                             QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import UUIDLineEdit


class WorkflowAuthorizeDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.lineEdit = QLineEdit()
        self.buttonBox = QDialogButtonBox()
        self.okButton = self.buttonBox.addButton(QDialogButtonBox.Ok)
        self.cancelButton = self.buttonBox.addButton(QDialogButtonBox.Cancel)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowAuthorizeDialog:
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
        self.endpointIDLineEdit = UUIDLineEdit()
        self.dataEndpointIDLineEdit = UUIDLineEdit()
        self.pathLineEdit = QLineEdit()
        self.flowIDLineEdit = UUIDLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowComputeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Endpoint ID:', view.endpointIDLineEdit)
        layout.addRow('Data Endpoint ID:', view.dataEndpointIDLineEdit)
        layout.addRow('Data Path:', view.pathLineEdit)
        layout.addRow('Flow ID:', view.flowIDLineEdit)
        view.setLayout(layout)

        return view


class WorkflowStatusView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Status', parent)
        self.refreshIntervalSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowStatusView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Refresh Interval [sec]:', view.refreshIntervalSpinBox)
        view.setLayout(layout)

        return view


class WorkflowButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.authorizeButton = QPushButton('Authorize')
        self.executeButton = QPushButton('Execute')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.authorizeButton)
        layout.addWidget(view.executeButton)
        view.setLayout(layout)

        return view


class WorkflowDeveloperButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.deployFlowButton = QPushButton('Deploy')
        self.listFlowsButton = QPushButton('List')
        self.updateFlowButton = QPushButton('Update')
        self.deleteFlowButton = QPushButton('Delete')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowDeveloperButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.deployFlowButton)
        layout.addWidget(view.listFlowsButton)
        layout.addWidget(view.updateFlowButton)
        layout.addWidget(view.deleteFlowButton)
        view.setLayout(layout)

        return view


class WorkflowDeveloperView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Developer', parent)
        self.flowIDLineEdit = UUIDLineEdit()
        self.buttonBox = WorkflowDeveloperButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowDeveloperView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.flowIDLineEdit)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class WorkflowParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.inputDataView = WorkflowDataView.createInstance('Input Data')
        self.computeView = WorkflowComputeView.createInstance()
        self.outputDataView = WorkflowDataView.createInstance('Output Data')
        self.statusView = WorkflowStatusView.createInstance()
        self.buttonBox = WorkflowButtonBox.createInstance()
        self.developerView = WorkflowDeveloperView.createInstance()
        self.authorizeDialog = WorkflowAuthorizeDialog.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.inputDataView)
        layout.addWidget(view.computeView)
        layout.addWidget(view.outputDataView)
        layout.addWidget(view.statusView)
        layout.addWidget(view.buttonBox)
        layout.addStretch()
        layout.addWidget(view.developerView)
        view.setLayout(layout)

        return view
