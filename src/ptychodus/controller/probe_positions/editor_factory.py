from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QWidget,
)

from ...model.product.probe_positions import (
    CartesianProbePositionsBuilder,
    ConcentricProbePositionsBuilder,
    FromFileProbePositionsBuilder,
    FromMemoryProbePositionsBuilder,
    LissajousProbePositionsBuilder,
    ProbePositionsBuilder,
    ProbePositionsRepositoryItem,
    SpiralProbePositionsBuilder,
)
from ..parametric import (
    CheckableGroupBoxParameterViewController,
    DecimalLineEditParameterViewController,
    LengthWidgetParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
)
from ...view.widgets import GroupBoxWithPresets

__all__ = [
    'ProbePositionsEditorViewControllerFactory',
]


class ProbePositionsTransformViewController(ParameterViewController):
    def __init__(self, builder: ProbePositionsBuilder) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets('Transformation')

        for index, presets_label in enumerate(builder.labels_for_preset_transforms()):
            action = self._widget.presets_menu.addAction(presets_label)
            action.triggered.connect(lambda _, index=index: builder.assign_preset_transform(index))

        self._label_xe = QLabel('x\u2032 =')
        self._label_xe.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._affine00_view_controller = DecimalLineEditParameterViewController(
            builder.affine00, is_signed=True
        )
        self._label_xp0 = QLabel('x +')
        self._affine01_view_controller = DecimalLineEditParameterViewController(
            builder.affine01, is_signed=True
        )
        self._label_yp0 = QLabel('y +')
        self._affine02_view_controller = LengthWidgetParameterViewController(
            builder.affine02, is_signed=True
        )

        self._label_ye = QLabel('y\u2032 =')
        self._label_ye.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._affine10_view_controller = DecimalLineEditParameterViewController(
            builder.affine10, is_signed=True
        )
        self._label_xp1 = QLabel('x +')
        self._affine11_view_controller = DecimalLineEditParameterViewController(
            builder.affine11, is_signed=True
        )
        self._label_yp1 = QLabel('y +')
        self._affine12_view_controller = LengthWidgetParameterViewController(
            builder.affine12, is_signed=True
        )

        self._jitter_radius_label = QLabel('Jitter Radius:')
        self._jitter_radius_view_controller = LengthWidgetParameterViewController(
            builder.jitter_radius_m, is_signed=False
        )

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._label_xe, 0, 0)
        layout.addWidget(self._affine00_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._label_xp0, 0, 2)
        layout.addWidget(self._affine01_view_controller.get_widget(), 0, 3)
        layout.addWidget(self._label_yp0, 0, 4)
        layout.addWidget(self._affine02_view_controller.get_widget(), 0, 5)

        layout.addWidget(self._label_ye, 1, 0)
        layout.addWidget(self._affine10_view_controller.get_widget(), 1, 1)
        layout.addWidget(self._label_xp1, 1, 2)
        layout.addWidget(self._affine11_view_controller.get_widget(), 1, 3)
        layout.addWidget(self._label_yp1, 1, 4)
        layout.addWidget(self._affine12_view_controller.get_widget(), 1, 5)

        layout.addWidget(self._jitter_radius_label, 2, 0)
        layout.addWidget(self._jitter_radius_view_controller.get_widget(), 2, 1, 1, 5)
        self._widget.contents.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class ScanBoundingBoxViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, item: ProbePositionsRepositoryItem) -> None:
        super().__init__(item.expand_bbox, 'Expand Bounding Box')
        self._xmin_controller = LengthWidgetParameterViewController(
            item.expand_bbox_xmin_m, is_signed=True
        )
        self._xmax_controller = LengthWidgetParameterViewController(
            item.expand_bbox_xmax_m, is_signed=True
        )
        self._ymin_controller = LengthWidgetParameterViewController(
            item.expand_bbox_ymin_m, is_signed=True
        )
        self._ymax_controller = LengthWidgetParameterViewController(
            item.expand_bbox_ymax_m, is_signed=True
        )

        layout = QFormLayout()
        layout.addRow('Minimum X:', self._xmin_controller.get_widget())
        layout.addRow('Maximum X:', self._xmax_controller.get_widget())
        layout.addRow('Minimum Y:', self._ymin_controller.get_widget())
        layout.addRow('Maximum Y:', self._ymax_controller.get_widget())
        self.get_widget().setLayout(layout)


class ProbePositionsEditorViewControllerFactory:
    def _append_common_controls(
        self, dialog_builder: ParameterViewBuilder, item: ProbePositionsRepositoryItem
    ) -> None:
        builder = item.get_builder()

        # A from-memory builder holds already-conditioned positions (reconstruction
        # output, or a product loaded from file), so its trim and transform
        # parameters are deliberately inert; do not offer controls that would do
        # nothing. The bounding box lives on the item and only widens the object
        # canvas, so it stays available for every builder.
        if not isinstance(builder, FromMemoryProbePositionsBuilder):
            trim_group = 'Trim Probe Positions'
            dialog_builder.add_spin_box(
                builder.num_discard_at_start,
                'Discard at Start:',
                tool_tip='Number of probe positions to discard from the beginning'
                ' of the scan, in acquisition order.',
                group=trim_group,
            )
            dialog_builder.add_spin_box(
                builder.num_discard_at_end,
                'Discard at End:',
                tool_tip='Number of probe positions to discard from the end'
                ' of the scan, in acquisition order.',
                group=trim_group,
            )
            dialog_builder.add_view_controller_to_bottom(
                ProbePositionsTransformViewController(builder)
            )

        dialog_builder.add_view_controller_to_bottom(ScanBoundingBoxViewController(item))

    def create_editor_dialog(
        self, item_name: str, item: ProbePositionsRepositoryItem, parent: QWidget
    ) -> QDialog:
        probe_positions_builder = item.get_builder()
        builder_name = probe_positions_builder.get_name()
        base_probe_positions_group = 'Base Probe Positions'
        title = f'{item_name} [{builder_name}]'

        if isinstance(probe_positions_builder, CartesianProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_spin_box(
                probe_positions_builder.num_points_x,
                'Number of Points X:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_spin_box(
                probe_positions_builder.num_points_y,
                'Number of Points Y:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_length_widget(
                probe_positions_builder.step_size_x_m,
                'Step Size X:',
                group=base_probe_positions_group,
            )

            if not probe_positions_builder.is_equilateral:
                dialog_builder.add_length_widget(
                    probe_positions_builder.step_size_y_m,
                    'Step Size Y:',
                    group=base_probe_positions_group,
                )

            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_positions_builder, ConcentricProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_spin_box(
                probe_positions_builder.num_shells,
                'Number of Shells:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_spin_box(
                probe_positions_builder.num_points_1st_shell,
                'Number of Points in First Shell:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_length_widget(
                probe_positions_builder.radial_step_size_m,
                'Radial Step Size:',
                group=base_probe_positions_group,
            )
            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_positions_builder, FromFileProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_positions_builder, FromMemoryProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_positions_builder, SpiralProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_spin_box(
                probe_positions_builder.num_points,
                'Number of Points:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_length_widget(
                probe_positions_builder.radius_scalar_m,
                'Radius Scalar:',
                group=base_probe_positions_group,
            )
            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_positions_builder, LissajousProbePositionsBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_spin_box(
                probe_positions_builder.num_points,
                'Number of Points:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_length_widget(
                probe_positions_builder.amplitude_x_m,
                'Amplitude X:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_length_widget(
                probe_positions_builder.amplitude_y_m,
                'Amplitude Y:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_angle_widget(
                probe_positions_builder.angular_step_x_turns,
                'Angular Step X:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_angle_widget(
                probe_positions_builder.angular_step_y_turns,
                'Angular Step Y:',
                group=base_probe_positions_group,
            )
            dialog_builder.add_angle_widget(
                probe_positions_builder.angular_shift_turns,
                'Angular Shift:',
                group=base_probe_positions_group,
            )
            self._append_common_controls(dialog_builder, item)
            return dialog_builder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builder_name}" has no editable parameters!',
            QMessageBox.StandardButton.Ok,
            parent,
        )
