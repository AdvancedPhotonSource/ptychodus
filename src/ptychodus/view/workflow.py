from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .widgets import UUIDLineEdit


class WorkflowAuthorizationDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.line_edit = QLineEdit()
        self.button_box = QDialogButtonBox()
        self.ok_button = self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowAuthorizationDialog:
        view = cls(parent)

        view.setWindowTitle('Authorize Workflow')
        view.label.setTextFormat(Qt.TextFormat.RichText)
        view.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        view.label.setOpenExternalLinks(True)

        view.button_box.clicked.connect(view._handle_button_box_clicked)

        layout = QVBoxLayout()
        layout.addWidget(view.label)
        layout.addWidget(view.line_edit)
        layout.addWidget(view.button_box)
        view.setLayout(layout)

        return view

    def _handle_button_box_clicked(self, button: QAbstractButton) -> None:
        if self.button_box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class WorkflowInputDataView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Input Data', parent)
        self.endpoint_id_line_edit = UUIDLineEdit()
        self.globus_path_line_edit = QLineEdit()
        self.posix_path_line_edit = QLineEdit()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowInputDataView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Endpoint ID:', view.endpoint_id_line_edit)
        layout.addRow('Globus Path:', view.globus_path_line_edit)
        layout.addRow('POSIX Path:', view.posix_path_line_edit)
        view.setLayout(layout)

        return view


class WorkflowOutputDataView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Output Data', parent)
        self.round_trip_check_box = QCheckBox('Round Trip')
        self.endpoint_id_line_edit = UUIDLineEdit()
        self.globus_path_line_edit = QLineEdit()
        self.posix_path_line_edit = QLineEdit()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowOutputDataView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.round_trip_check_box)
        layout.addRow('Endpoint ID:', view.endpoint_id_line_edit)
        layout.addRow('Globus Path:', view.globus_path_line_edit)
        layout.addRow('POSIX Path:', view.posix_path_line_edit)
        view.setLayout(layout)

        return view


class WorkflowComputeView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Compute', parent)
        self.compute_endpoint_id_line_edit = UUIDLineEdit()
        self.data_endpoint_id_line_edit = UUIDLineEdit()
        self.data_globus_path_line_edit = QLineEdit()
        self.data_posix_path_line_edit = QLineEdit()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowComputeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Compute Endpoint ID:', view.compute_endpoint_id_line_edit)
        layout.addRow('Data Endpoint ID:', view.data_endpoint_id_line_edit)
        layout.addRow('Data Globus Path:', view.data_globus_path_line_edit)
        layout.addRow('Data POSIX Path:', view.data_posix_path_line_edit)
        view.setLayout(layout)

        return view


class WorkflowExecutionView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Execution', parent)
        self.product_combo_box = QComboBox()
        self.input_data_view = WorkflowInputDataView.create_instance()
        self.compute_view = WorkflowComputeView.create_instance()
        self.output_data_view = WorkflowOutputDataView.create_instance()
        self.execute_button = QPushButton('Execute')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowExecutionView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Product:', view.product_combo_box)
        layout.addRow(view.input_data_view)
        layout.addRow(view.compute_view)
        layout.addRow(view.output_data_view)
        layout.addRow(view.execute_button)
        view.setLayout(layout)

        return view


class WorkflowStatusView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Status', parent)
        self.auto_refresh_check_box = QCheckBox('Auto Refresh [sec]:')
        self.auto_refresh_spin_box = QSpinBox()
        self.refresh_button = QPushButton('Refresh')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowStatusView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.auto_refresh_check_box, view.auto_refresh_spin_box)
        layout.addRow(view.refresh_button)
        view.setLayout(layout)

        return view


class WorkflowParametersView(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.execution_view = WorkflowExecutionView.create_instance()
        self.status_view = WorkflowStatusView.create_instance()
        self.authorization_dialog = WorkflowAuthorizationDialog.create_instance(self)

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> WorkflowParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.execution_view)
        layout.addWidget(view.status_view)
        layout.addStretch()
        view.setLayout(layout)

        return view
