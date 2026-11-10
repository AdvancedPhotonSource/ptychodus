from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from .dashboard import DashboardDialog, MetricsGroupView, MetricsTableView, PlotView


class ProbePositionsMetricsDialog(DashboardDialog):
    """What one scan's positions say about its indexing, extent and step sizes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product_combo_box = QComboBox()
        self.inputs_layout.addRow('Product:', self.product_combo_box)

        self.indexes_group = MetricsGroupView('Scan Indexes')
        self.num_positions_label = self.indexes_group.add_row('Positions')
        self.num_distinct_label = self.indexes_group.add_row(
            'Distinct Indexes', 'Positions sharing an index are averaged into one'
        )
        self.index_range_label = self.indexes_group.add_row('Index Range')
        self.duplicated_label = self.indexes_group.add_row('Duplicated Indexes')
        self.missing_label = self.indexes_group.add_row('Missing Indexes')

        self.extent_group = MetricsGroupView('Extent')
        self.width_label = self.extent_group.add_row('Width [µm]')
        self.height_label = self.extent_group.add_row('Height [µm]')
        self.center_x_label = self.extent_group.add_row('Center X [µm]')
        self.center_y_label = self.extent_group.add_row('Center Y [µm]')
        self.hull_area_label = self.extent_group.add_row(
            'Convex Hull Area [µm²]',
            'Zero for fewer than three positions or a one-dimensional scan',
        )
        self.density_label = self.extent_group.add_row('Density [1/µm²]')

        self.step_group = MetricsGroupView('Nearest-Neighbor Distance')
        self.step_mean_label = self.step_group.add_row('Mean [nm]')
        self.step_median_label = self.step_group.add_row('Median [nm]')
        self.step_minimum_label = self.step_group.add_row('Minimum [nm]')
        self.step_maximum_label = self.step_group.add_row('Maximum [nm]')

        self.i0_group = MetricsGroupView('I0')
        self.i0_mean_label = self.i0_group.add_row('Mean')
        self.i0_std_label = self.i0_group.add_row('Std Dev')
        self.i0_minimum_label = self.i0_group.add_row('Minimum')
        self.i0_maximum_label = self.i0_group.add_row('Maximum')

        for group in (self.indexes_group, self.extent_group, self.step_group, self.i0_group):
            self.numbers_layout.addWidget(group)

        self.finish_numbers()

        self.scan_plot_view = PlotView()
        self.images_layout.addWidget(self.scan_plot_view, 0, 0)

        self.plot_view = PlotView()
        self.plot_view.figure.clear()
        self.step_histogram_axes, self.i0_axes = self.plot_view.figure.subplots(1, 2)
        self.plots_layout.addWidget(self.plot_view)


class ProbePositionsCompareDialog(DashboardDialog):
    """How a test scan's positions are displaced from a reference scan's."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reference_combo_box = QComboBox()
        self.test_combo_box = QComboBox()
        self.inputs_layout.addRow('Reference:', self.reference_combo_box)
        self.inputs_layout.addRow('Test:', self.test_combo_box)

        self.matching_group = MetricsGroupView('Matching')
        self.matching_group.setToolTip(
            'Positions pair by scan index; nothing is interpolated between indexes'
        )
        self.num_matched_label = self.matching_group.add_row('Matched Indexes')
        self.num_reference_only_label = self.matching_group.add_row('Reference Only')
        self.num_test_only_label = self.matching_group.add_row('Test Only')

        self.displacement_group = MetricsGroupView('Displacement (Test − Reference)')
        self.mean_x_label = self.displacement_group.add_row('Mean X [nm]')
        self.mean_y_label = self.displacement_group.add_row('Mean Y [nm]')
        self.rms_label = self.displacement_group.add_row('RMS Magnitude [nm]')
        self.median_label = self.displacement_group.add_row('Median Magnitude [nm]')
        self.maximum_label = self.displacement_group.add_row('Maximum Magnitude [nm]')

        self.numbers_layout.addWidget(self.matching_group)
        self.numbers_layout.addWidget(self.displacement_group)
        self.finish_numbers()

        self.field_plot_view = PlotView()
        self.images_layout.addWidget(self.field_plot_view, 0, 0)

        self.plot_view = PlotView()
        self.plot_view.figure.clear()
        self.magnitude_histogram_axes = self.plot_view.figure.add_subplot(1, 3, 1)
        self.direction_histogram_axes = self.plot_view.figure.add_subplot(
            1, 3, 2, projection='polar'
        )
        self.index_axes = self.plot_view.figure.add_subplot(1, 3, 3)
        self.plots_layout.addWidget(self.plot_view)


class AffineTransformDialog(DashboardDialog):
    """Fit one affine transform jointly across measured/corrected position pairs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pairs_table_widget = QTableWidget(0, 2)
        self.pairs_table_widget.setHorizontalHeaderLabels(['Measured', 'Corrected'])
        self.pairs_table_widget.setMaximumHeight(160)
        self.pairs_table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pairs_table_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        horizontal_header = self.pairs_table_widget.horizontalHeader()

        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.add_pair_button = QPushButton('Add Pair')
        self.remove_pair_button = QPushButton('Remove Pair')
        self.estimate_button = QPushButton('Estimate')

        pair_buttons_layout = QVBoxLayout()
        pair_buttons_layout.addWidget(self.add_pair_button)
        pair_buttons_layout.addWidget(self.remove_pair_button)
        pair_buttons_layout.addStretch()

        pairs_layout = QHBoxLayout()
        pairs_layout.addWidget(self.pairs_table_widget, 1)
        pairs_layout.addLayout(pair_buttons_layout)
        self.inputs_layout.addRow(pairs_layout)

        self.settings_group = QGroupBox('Estimator')
        self.settings_layout = QFormLayout()
        self.settings_group.setLayout(self.settings_layout)
        self.inputs_layout.addRow(self.settings_group)
        self.inputs_layout.addRow(self.estimate_button)

        self.transform_group = MetricsGroupView('Shared Linear Part')
        self.scale_label = self.transform_group.add_row('Scale')
        self.asymmetry_label = self.transform_group.add_row('Asymmetry')
        self.rotation_label = self.transform_group.add_row('Rotation [deg]')
        self.shear_label = self.transform_group.add_row('Shear [deg]')
        self.rms_residual_label = self.transform_group.add_row('RMS Residual [nm]')

        self.pairs_group = QGroupBox('Pairs')
        self.pairs_group.setToolTip(
            'Each pair keeps its own translation, since a separate scan has its own stage origin'
        )
        self.pairs_result_table_view = MetricsTableView(
            ['Shift X [nm]', 'Shift Y [nm]', 'Inliers', 'RMS [nm]']
        )
        pairs_result_layout = QVBoxLayout()
        pairs_result_layout.addWidget(self.pairs_result_table_view)
        self.pairs_group.setLayout(pairs_result_layout)

        self.numbers_layout.addWidget(self.transform_group)
        self.numbers_layout.addWidget(self.pairs_group)
        self.finish_numbers()

        self.residual_pair_combo_box = QComboBox()
        self.residual_plot_view = PlotView()
        residual_layout = QVBoxLayout()
        residual_layout.addWidget(self.residual_pair_combo_box)
        residual_layout.addWidget(self.residual_plot_view)
        residual_group = QGroupBox('Residual After the Fitted Transform')
        residual_group.setLayout(residual_layout)
        self.images_layout.addWidget(residual_group, 0, 0)
