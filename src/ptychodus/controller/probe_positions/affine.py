from __future__ import annotations
import logging
import math

from PyQt5.QtCore import QAbstractItemModel
from PyQt5.QtWidgets import QComboBox

from matplotlib.lines import Line2D

from ptychodus.api.affine import AffineFitResult
from ptychodus.api.constants import LengthUnit

from ...model.analysis import AffineTransformEstimator, AffineTransformEstimatorSettings
from ...view.probe_positions import AffineTransformDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import format_scalar, set_table_rows, wait_cursor
from ..parameters import LengthParameterViewController, SpinBoxParameterViewController

logger = logging.getLogger(__name__)


class AffineTransformViewController:
    """Pairs measured with corrected positions and fits one affine transform across them."""

    def __init__(
        self,
        estimator: AffineTransformEstimator,
        settings: AffineTransformEstimatorSettings,
        table_model: QAbstractItemModel,
    ) -> None:
        self._estimator = estimator
        self._table_model = table_model
        self._result: AffineFitResult | None = None
        self._pair_names: list[str] = []
        self._dialog = AffineTransformDialog()
        self._dialog.setWindowTitle('Estimate Affine Transform')

        self._num_iterations_controller = SpinBoxParameterViewController(
            settings.num_iterations, tool_tip='Consensus samples drawn per pair'
        )
        self._inlier_threshold_controller = LengthParameterViewController(
            settings.inlier_threshold_m,
            default_unit=LengthUnit.NANOMETER,
            tool_tip='Distance within which the consensus pass counts a point as agreeing',
        )
        self._min_inliers_controller = SpinBoxParameterViewController(
            settings.min_inliers, tool_tip='Inliers a pair must reach for its fit to be trusted'
        )
        settings_layout = self._dialog.settings_layout
        settings_layout.addRow('Iterations:', self._num_iterations_controller.get_widget())
        settings_layout.addRow('Inlier Threshold:', self._inlier_threshold_controller.get_widget())
        settings_layout.addRow('Minimum Inliers:', self._min_inliers_controller.get_widget())

        self._dialog.add_pair_button.clicked.connect(self._add_default_pair)
        self._dialog.remove_pair_button.clicked.connect(self._remove_selected_pair)
        self._dialog.estimate_button.clicked.connect(self._estimate)
        self._dialog.residual_pair_combo_box.activated.connect(self._show_residuals)

    def launch(self, product_index: int) -> None:
        if self._dialog.pairs_table_widget.rowCount() == 0:
            self._add_pair(product_index, product_index + 1)

        self._dialog.open()

    def _add_pair(self, measured_index: int, corrected_index: int) -> None:
        table = self._dialog.pairs_table_widget
        row = table.rowCount()
        table.insertRow(row)
        count = self._table_model.rowCount()

        for column, index in enumerate((measured_index, corrected_index)):
            combo_box = QComboBox()
            combo_box.setModel(self._table_model)
            combo_box.setCurrentIndex(min(max(index, 0), count - 1))
            table.setCellWidget(row, column, combo_box)

    def _add_default_pair(self) -> None:
        count = self._table_model.rowCount()
        used = len(self._pair_indexes())
        self._add_pair(min(2 * used, count - 1), min(2 * used + 1, count - 1))

    def _remove_selected_pair(self) -> None:
        table = self._dialog.pairs_table_widget
        row = table.currentRow()

        if row < 0:
            row = table.rowCount() - 1

        if row >= 0:
            table.removeRow(row)

    def _pair_indexes(self) -> list[tuple[int, int]]:
        table = self._dialog.pairs_table_widget
        pairs: list[tuple[int, int]] = []

        for row in range(table.rowCount()):
            measured = table.cellWidget(row, 0)
            corrected = table.cellWidget(row, 1)

            if isinstance(measured, QComboBox) and isinstance(corrected, QComboBox):
                pairs.append((measured.currentIndex(), corrected.currentIndex()))

        return pairs

    def _estimate(self) -> None:
        pairs = self._pair_indexes()
        self._result = None

        try:
            if not pairs:
                raise ValueError('Add at least one measured/corrected pair.')

            with wait_cursor():
                self._result = self._estimator.estimate(
                    [measured for measured, _ in pairs], [corrected for _, corrected in pairs]
                )
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Estimate Affine Transform', err)

        model = self._table_model
        self._pair_names = [
            f'{model.index(measured, 0).data()} → {model.index(corrected, 0).data()}'
            for measured, corrected in pairs
        ]
        self._show_result()

    def _show_result(self) -> None:
        dialog = self._dialog
        result = self._result
        dialog.residual_pair_combo_box.clear()

        if result is None:
            dialog.transform_group.reset()
            set_table_rows(dialog.pairs_result_table_view, [])
            self._show_residuals()
            return

        nanometer = LengthUnit.NANOMETER
        components = result.components
        dialog.scale_label.setText(format_scalar(components.scale, precision=6))
        dialog.asymmetry_label.setText(format_scalar(components.asymmetry, precision=4))
        dialog.rotation_label.setText(
            format_scalar(math.degrees(components.rotation_rad), precision=4)
        )
        dialog.shear_label.setText(format_scalar(math.degrees(components.shear_rad), precision=4))
        dialog.rms_residual_label.setText(
            format_scalar(nanometer.convert(result.rms_residual_m), precision=4)
        )
        set_table_rows(
            dialog.pairs_result_table_view,
            [
                (
                    name,
                    [
                        format_scalar(nanometer.convert(pair.translation_x_m), precision=4),
                        format_scalar(nanometer.convert(pair.translation_y_m), precision=4),
                        f'{pair.num_inliers}/{pair.num_points}',
                        format_scalar(nanometer.convert(pair.rms_residual_m), precision=4),
                    ],
                )
                for name, pair in zip(self._pair_names, result.pairs)
            ],
        )
        dialog.residual_pair_combo_box.addItems(self._pair_names)
        self._show_residuals()

    def _show_residuals(self) -> None:
        plot_view = self._dialog.residual_plot_view
        figure = plot_view.figure
        figure.clear()
        result = self._result
        pair_index = self._dialog.residual_pair_combo_box.currentIndex()

        if result is not None and 0 <= pair_index < len(result.pairs):
            pair = result.pairs[pair_index]
            micrometer = LengthUnit.MICROMETER
            nanometer = LengthUnit.NANOMETER
            coordinates_um = pair.coordinates_xy_m / micrometer.meters_per_unit
            residual_nm = pair.residual_xy_m / nanometer.meters_per_unit
            axes = figure.add_subplot(111)
            inlier_color, outlier_color = 'tab:blue', 'tab:red'
            # One quiver, so inliers and outliers share an arrow scale.
            axes.quiver(
                coordinates_um[:, 0],
                coordinates_um[:, 1],
                residual_nm[:, 0],
                residual_nm[:, 1],
                color=[inlier_color if inlier else outlier_color for inlier in pair.is_inlier],
                angles='xy',
            )
            axes.legend(
                handles=[
                    Line2D([], [], color=inlier_color, label='Inlier'),
                    Line2D([], [], color=outlier_color, label='Outlier'),
                ],
                loc='best',
                fontsize='small',
            )

            axes.invert_yaxis()
            axes.set_aspect('equal')
            axes.grid(True)
            axes.set_xlabel('X [µm]')
            axes.set_ylabel('Y [µm]')
            axes.set_title('Arrow lengths are scaled for visibility', fontsize='small')
            figure.tight_layout()

        plot_view.figure_canvas.draw()
