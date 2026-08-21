from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .image import ImageView
from .repository import InsertSaveEditRemoveButtonBox
from .widgets import TaskStatusView


class ProductEditorPropertiesView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Properties')
        self.table_view = QTableView()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)


class ProductEditorActionsView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Actions')
        self.estimate_probe_photon_count_button = QPushButton('Estimate Probe Photon Count')

        layout = QVBoxLayout()
        layout.addWidget(self.estimate_probe_photon_count_button)
        layout.addStretch()
        self.setLayout(layout)


class ProductEditorCommentsView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Comments')
        self.text_edit = QPlainTextEdit()

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)


class ProductEditorDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.properties_view = ProductEditorPropertiesView()
        self.actions_view = ProductEditorActionsView()
        self.comments_view = ProductEditorCommentsView()
        self.button_box = QDialogButtonBox()

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.properties_view)
        top_layout.addWidget(self.actions_view)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.comments_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    @property
    def table_view(self) -> QTableView:
        return self.properties_view.table_view

    @property
    def text_edit(self) -> QPlainTextEdit:
        return self.comments_view.text_edit


class ProductView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()
        self.info_label = QLabel()
        self.button_box = InsertSaveEditRemoveButtonBox()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class ProductVisualizationView(QWidget):
    """Right-panel tabbed visualization of reconstruction residuals for the active product."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.real_space_image_view = ImageView()
        self.reciprocal_space_image_view = ImageView()
        self.compute_button = QPushButton('Compute Residuals')

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.real_space_image_view, 'Real-Space Error Map')
        self.tab_widget.addTab(self.reciprocal_space_image_view, 'Reciprocal-Space Error Map')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.compute_button)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)


class ProductRightPanelView(QWidget):
    """Right-hand Products panel: optional residual visualization over a status strip.

    The visualization is developer-mode only, but the status strip is not — queued
    product construction has to be visible in either mode, so this wrapper is always
    the panel's right widget and only its upper half is conditional.
    """

    def __init__(
        self,
        visualization_view: ProductVisualizationView | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.visualization_view = visualization_view
        self.status_view = TaskStatusView()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if visualization_view is None:
            layout.addStretch()
        else:
            layout.addWidget(visualization_view)

        layout.addWidget(self.status_view)
        self.setLayout(layout)
