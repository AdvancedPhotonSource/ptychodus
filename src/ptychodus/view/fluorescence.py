from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class FluorescenceEnhanceParametersView(QGroupBox):
    """Lean enhancement parameter form: algorithm chooser + per-algorithm parameter stack.

    Compared with the older ``FluorescenceParametersView`` this drops the
    Open/Save buttons — measured-dataset loading and enhanced-dataset saving
    now live in the top-level fluorescence panel; this widget is only shown
    inside the modal enhance dialog.
    """

    def __init__(self, algorithm_widget: QWidget, parent: QWidget | None = None) -> None:
        super().__init__('Enhancement Strategy', parent)
        self.stacked_widget = QStackedWidget()

        stacked_widget_layout = self.stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        layout = QFormLayout()
        layout.addRow('Algorithm:', algorithm_widget)
        layout.addRow(self.stacked_widget)
        self.setLayout(layout)


class FluorescenceStatusView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Status', parent)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.stop_button = QPushButton('Stop')

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addLayout(progress_layout)
        self.setLayout(layout)


class FluorescenceEnhanceDialog(QDialog):
    """Modal enhancement dialog: algorithm + params + status + Run/Close.

    Visualization, element selection, and save affordances live in the parent
    panel; this dialog is intentionally minimal.
    """

    def __init__(self, algorithm_widget: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = FluorescenceEnhanceParametersView(algorithm_widget)
        self.status_view = FluorescenceStatusView()
        self.run_button = QPushButton('Run')
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)

        run_layout = QHBoxLayout()
        run_layout.addStretch()
        run_layout.addWidget(self.run_button)

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)
        layout.addLayout(run_layout)
        layout.addWidget(self.status_view, 1)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class FluorescenceEditorDialog(QDialog):
    """Property editor for one fluorescence repository item.

    Follows ProductEditorDialog: a property table plus a close button. Name and the
    bound product are editable; everything else is provenance shown read-only.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()
        self.button_box = QDialogButtonBox()

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class FluorescenceButtonBox(QWidget):
    """Insert / Enhance / Save / Edit / Remove strip for the fluorescence panel.

    Matches InsertSaveEditRemoveButtonBox, with Enhance added between Insert and
    Save. Enhance is a plain push button: it acts immediately rather than opening
    a menu, so attaching one would suppress its clicked signal.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.insert_menu = QMenu()
        self.insert_button = QPushButton('Insert')
        self.enhance_button = QPushButton('Enhance')
        self.save_menu = QMenu()
        self.save_button = QPushButton('Save')
        self.edit_button = QPushButton('Edit')
        self.remove_button = QPushButton('Remove')

        self.insert_button.setMenu(self.insert_menu)
        self.save_button.setMenu(self.save_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.insert_button)
        layout.addWidget(self.enhance_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)


class FluorescenceView(QWidget):
    """Left-pane dataset browser for the top-level Fluorescence subview.

    Layout (top → bottom): dataset tree (expandable to element leaves), a
    measured/enhanced variant selector, and the button box. Each fluorescence
    dataset is bound to a target product at load time, so the panel carries
    no global product picker — the tree's Product column shows the bound
    product per item.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree_view = QTreeView()
        self.tree_view.setRootIsDecorated(True)
        self.tree_view.setUniformRowHeights(True)
        self.measured_radio_button = QRadioButton('Measured')
        self.enhanced_radio_button = QRadioButton('Enhanced')
        self.measured_radio_button.setChecked(True)
        self.variant_button_group = QButtonGroup(self)
        self.variant_button_group.setExclusive(True)
        self.variant_button_group.addButton(self.measured_radio_button)
        self.variant_button_group.addButton(self.enhanced_radio_button)
        self.info_label = QLabel()
        self.button_box = FluorescenceButtonBox()

        variant_group = QGroupBox('Variant')
        variant_layout = QHBoxLayout()
        variant_layout.addWidget(self.measured_radio_button)
        variant_layout.addWidget(self.enhanced_radio_button)
        variant_group.setLayout(variant_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view, 1)
        layout.addWidget(self.info_label)
        layout.addWidget(variant_group)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
