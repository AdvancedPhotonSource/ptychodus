from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractButton,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


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
        self.button_box.clicked.connect(self._handle_button_box_clicked)

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

    def _handle_button_box_clicked(self, button: QAbstractButton) -> None:
        if self.button_box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ProductButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.insert_menu = QMenu()
        self.insert_button = QPushButton('Insert')
        self.save_menu = QMenu()
        self.save_button = QPushButton('Save')
        self.edit_button = QPushButton('Edit')
        self.remove_button = QPushButton('Remove')

        self.insert_button.setMenu(self.insert_menu)
        self.save_button.setMenu(self.save_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.insert_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)


class ProductView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()
        self.info_label = QLabel()
        self.button_box = ProductButtonBox()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
