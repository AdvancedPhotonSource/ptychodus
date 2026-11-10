from __future__ import annotations
from collections.abc import Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

PLACEHOLDER = '—'


class PlotView(QWidget):
    """A matplotlib figure with its navigation toolbar.

    Starts with a single ``axes``; a view needing a grid of plots clears the figure and
    builds its own with ``figure.subplots``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.navigation_toolbar)
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)


class MetricsGroupView(QGroupBox):
    """A titled column of labelled values."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._value_labels: list[QLabel] = []
        self._layout = QFormLayout()
        self.setLayout(self._layout)

    def add_row(self, label: str, tooltip: str = '') -> QLabel:
        """Append a row and return the label that shows its value."""
        value_label = QLabel(PLACEHOLDER)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        value_label.setWordWrap(True)

        if tooltip:
            value_label.setToolTip(tooltip)

        self._layout.addRow(f'{label}:', value_label)
        self._value_labels.append(value_label)
        return value_label

    def reset(self) -> None:
        """Show every value as unknown."""
        for value_label in self._value_labels:
            value_label.setText(PLACEHOLDER)


class MetricsTableView(QTableWidget):
    """A read-only table of values, one row per quantity and one column per subject."""

    def __init__(self, column_labels: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(0, len(column_labels), parent)
        self.setHorizontalHeaderLabels(list(column_labels))
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        horizontal_header = self.horizontalHeader()

        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.fit_to_contents()

    def fit_to_contents(self) -> None:
        """Size the table to show every row without scrolling."""
        horizontal_header = self.horizontalHeader()
        header_height = 0 if horizontal_header is None else horizontal_header.height()
        rows_height = sum(self.rowHeight(row) for row in range(self.rowCount()))
        self.setFixedHeight(header_height + rows_height + 2 * self.frameWidth())


class DashboardDialog(QDialog):
    """A one-page analysis dashboard.

    Inputs run across the top; numbers fill a scrolling column on the left; images sit
    in a grid on the right above any plots. A view adds its widgets to
    ``inputs_layout``, ``numbers_layout``, ``images_layout`` and ``plots_layout``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.inputs_layout = QFormLayout()
        self.numbers_layout = QVBoxLayout()
        self.images_layout = QGridLayout()
        self.plots_layout = QVBoxLayout()
        self.status_bar = QStatusBar()

        inputs_group = QGroupBox('Inputs')
        inputs_group.setLayout(self.inputs_layout)

        numbers_widget = QWidget()
        numbers_widget.setLayout(self.numbers_layout)
        numbers_scroll_area = QScrollArea()
        numbers_scroll_area.setWidgetResizable(True)
        numbers_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        numbers_scroll_area.setWidget(numbers_widget)

        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.addLayout(self.images_layout, 3)
        display_layout.addLayout(self.plots_layout, 2)
        display_widget = QWidget()
        display_widget.setLayout(display_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(numbers_scroll_area)
        splitter.addWidget(display_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([580, 820])

        layout = QVBoxLayout()
        layout.addWidget(inputs_group)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
        self.resize(1400, 900)

    def finish_numbers(self) -> None:
        """Push the numbers column to the top once every group has been added."""
        self.numbers_layout.addStretch()
