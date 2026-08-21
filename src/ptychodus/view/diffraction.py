from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from .image import ImageView
from .repository import InsertSaveEditRemoveButtonBox
from .widgets import TaskStatusView


class OpenDatasetWizardPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_complete = False

    def isComplete(self) -> bool:  # noqa: N802
        """Overrides QWizardPage.isComplete()"""
        return self._is_complete

    def _set_complete(self, complete: bool) -> None:
        if self._is_complete != complete:
            self._is_complete = complete
            self.completeChanged.emit()


class OpenDatasetWizardBadPixelsPage(OpenDatasetWizardPage):
    """Bad-pixels chooser page — always complete; layout populated by the controller."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._set_complete(True)


class OpenDatasetWizardMetadataPage(OpenDatasetWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()

        self.setTitle('Import Metadata')

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)
        self._set_complete(True)


class DatasetEditorLayoutView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Layout')
        self.tree_view = QTreeView()

        tree_header = self.tree_view.header()

        if tree_header is not None:
            tree_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            tree_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        self.setLayout(layout)


class DatasetEditorPropertiesView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Properties')
        self.table_view = QTableView()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)


class DatasetEditorDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout_view = DatasetEditorLayoutView()
        self.properties_view = DatasetEditorPropertiesView()
        self.button_box = QDialogButtonBox()

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.layout_view)
        top_layout.addWidget(self.properties_view)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    @property
    def tree_view(self) -> QTreeView:
        return self.layout_view.tree_view

    @property
    def table_view(self) -> QTableView:
        return self.properties_view.table_view


class SimulateDiffractionDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product_combo_box = QComboBox()
        self.button_box = QDialogButtonBox()

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        self.form_layout = QFormLayout()
        self.form_layout.addRow('Product:', self.product_combo_box)
        self.form_layout.addRow(self.button_box)
        self.setLayout(self.form_layout)

        self.setWindowTitle('Simulate Diffraction Patterns')


class DiffractionView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree_view = QTreeView()
        self.info_label = QLabel()
        self.button_box = InsertSaveEditRemoveButtonBox()
        self.simulate_dialog = SimulateDiffractionDialog(self)

        tree_view_header = self.tree_view.header()

        if tree_view_header is not None:
            tree_view_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class DiffractionImageView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_view = ImageView()
        self.status_view = TaskStatusView()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)
        layout.addWidget(self.status_view)
        self.setLayout(layout)
