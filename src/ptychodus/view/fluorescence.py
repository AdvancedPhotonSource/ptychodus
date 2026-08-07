from PyQt5.QtWidgets import (
    QButtonGroup,
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
    QRadioButton,
    QStackedWidget,
    QStatusBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .visualization import VisualizationParametersView, VisualizationWidget


class FluorescenceEnhanceParametersView(QGroupBox):
    """Lean enhancement parameter form: algorithm chooser + per-algorithm parameter stack.

    Compared with the older ``FluorescenceParametersView`` this drops the
    Open/Save buttons — measured-dataset loading and enhanced-dataset saving
    now live in the top-level fluorescence panel; this widget is only shown
    inside the modal enhance dialog.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Enhancement Strategy', parent)
        self.algorithm_combo_box = QComboBox()
        self.stacked_widget = QStackedWidget()

        stacked_widget_layout = self.stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        layout = QFormLayout()
        layout.addRow('Algorithm:', self.algorithm_combo_box)
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = FluorescenceEnhanceParametersView()
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


class FluorescenceButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.load_button = QPushButton('Load')
        self.enhance_menu = QMenu()
        self.enhance_button = QPushButton('Enhance')
        self.save_button = QPushButton('Save')
        self.remove_button = QPushButton('Remove')

        self.enhance_button.setMenu(self.enhance_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.load_button)
        layout.addWidget(self.enhance_button)
        layout.addWidget(self.save_button)
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
        self.button_box = FluorescenceButtonBox()

        variant_group = QGroupBox('Variant')
        variant_layout = QHBoxLayout()
        variant_layout.addWidget(self.measured_radio_button)
        variant_layout.addWidget(self.enhanced_radio_button)
        variant_group.setLayout(variant_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view, 1)
        layout.addWidget(variant_group)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class FluorescenceImageView(QWidget):
    """Right-pane element-map viewer for the top-level Fluorescence subview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualization_widget = VisualizationWidget('Element Map')
        self.visualization_parameters_view = VisualizationParametersView()
        self.status_bar = QStatusBar()

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(self.visualization_widget, 1)
        contents_layout.addWidget(self.visualization_parameters_view)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
