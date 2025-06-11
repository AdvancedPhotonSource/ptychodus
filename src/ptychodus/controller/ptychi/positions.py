from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtWidgets import QFormLayout, QFrame, QListView, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import (
    PtyChiAffineDegreesOfFreedomBitField,
    PtyChiEnumerators,
    PtyChiProbePositionSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    IntegerLineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbePositionsViewController']


class PtyChiCrossCorrelationViewController(ParameterViewController, Observer):
    def __init__(
        self,
        algorithm: StringParameter,
        scale: IntegerParameter,
        real_space_width: RealParameter,
        probe_threshold: RealParameter,
    ) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._scale_view_controller = SpinBoxParameterViewController(
            scale, tool_tip='Upsampling factor of the cross-correlation in real space'
        )
        self._real_space_width_view_controller = DecimalLineEditParameterViewController(
            real_space_width, tool_tip='Width of the cross-correlation in real-space'
        )
        self._probe_threshold_view_controller = DecimalSliderParameterViewController(
            probe_threshold, tool_tip='Probe intensity threshold used to calculate the probe mask'
        )
        self._widget = QFrame()
        self._widget.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QFormLayout()
        layout.addRow('Scale:', self._scale_view_controller.get_widget())
        layout.addRow('Real Space Width:', self._real_space_width_view_controller.get_widget())
        layout.addRow('Probe Threshold:', self._probe_threshold_view_controller.get_widget())
        self._widget.setLayout(layout)

        algorithm.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._widget.setVisible(self._algorithm.get_value().upper() == 'CROSS_CORRELATION')

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._sync_model_to_view()


class PtyChiUpdateMagnitudeLimitViewController(ParameterViewController, Observer):
    def __init__(
        self,
        limit_update_magnitude: BooleanParameter,
        update_magnitude_limit: RealParameter,
    ) -> None:
        self._limit_update_magnitude = limit_update_magnitude
        self._limit_update_magnitude_view_controller = CheckBoxParameterViewController(
            limit_update_magnitude,
            'Limit Update Magnitude:',
            tool_tip='Whether to limit the update magnitude',
        )
        self._update_magnitude_limit_view_controller = DecimalLineEditParameterViewController(
            update_magnitude_limit,
            tool_tip='Maximum allowed magnitude of position update in each axis',
        )

        limit_update_magnitude.add_observer(self)
        self._sync_model_to_view()

    def get_label(self) -> QWidget:
        return self._limit_update_magnitude_view_controller.get_widget()

    def get_widget(self) -> QWidget:
        return self._update_magnitude_limit_view_controller.get_widget()

    def _sync_model_to_view(self) -> None:
        self._update_magnitude_limit_view_controller.get_widget().setEnabled(
            self._limit_update_magnitude.get_value()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._limit_update_magnitude:
            self._sync_model_to_view()


class PtyChiSliceForCorrectionViewController(ParameterViewController, Observer):
    def __init__(
        self,
        choose_slice_for_correction: BooleanParameter,
        slice_for_correction: IntegerParameter,
    ) -> None:
        self._choose_slice_for_correction = choose_slice_for_correction
        self._choose_slice_for_correction_view_controller = CheckBoxParameterViewController(
            choose_slice_for_correction,
            'Slice For Correction:',
            tool_tip='Whether to specify the slice that is used for position correction',
        )
        self._slice_for_correction_view_controller = IntegerLineEditParameterViewController(
            slice_for_correction,
            tool_tip='Slice that is used for position correction (0-based index)',
        )

        choose_slice_for_correction.add_observer(self)
        self._sync_model_to_view()

    def get_label(self) -> QWidget:
        return self._choose_slice_for_correction_view_controller.get_widget()

    def get_widget(self) -> QWidget:
        return self._slice_for_correction_view_controller.get_widget()

    def _sync_model_to_view(self) -> None:
        self._slice_for_correction_view_controller.get_widget().setEnabled(
            self._choose_slice_for_correction.get_value()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._choose_slice_for_correction:
            self._sync_model_to_view()


class PtyChiAffineDegreesOfFreedomListModel(QAbstractListModel):
    def __init__(self, parameter: IntegerParameter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._dof = PtyChiAffineDegreesOfFreedomBitField(parameter)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return self._dof[index.row()]
            elif role == Qt.ItemDataRole.CheckStateRole:
                return (
                    Qt.CheckState.Checked
                    if self._dof.is_bit_set(index.row())
                    else Qt.CheckState.Unchecked
                )

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.CheckStateRole:
            self._dof.set_bit(index.row(), value == Qt.CheckState.Checked)
            self.dataChanged.emit(index, index)
            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._dof)


class PtyChiAffineDegreesOfFreedomViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._list_model = PtyChiAffineDegreesOfFreedomListModel(parameter)
        self._widget = QListView()
        self._widget.setModel(self._list_model)

        parameter.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._list_model.beginResetModel()
        self._list_model.endResetModel()

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class PtyChiConstrainAffineTransformViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        is_optimizable: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        degrees_of_freedom: IntegerParameter,
        position_weight_update_interval: IntegerParameter,
        apply_constraint: BooleanParameter,
        max_expected_error_px: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            is_optimizable,
            'Constrain Affine Transform',
            tool_tip='Constrain the affine transform during position correction',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._degrees_of_freedom_view_controller = PtyChiAffineDegreesOfFreedomViewController(
            degrees_of_freedom
        )
        self._weight_update_interval_view_controller = SpinBoxParameterViewController(
            position_weight_update_interval,
            tool_tip='Interval for updating the position weight',
        )
        self._apply_constraint_view_controller = CheckBoxParameterViewController(
            apply_constraint, 'Apply Constraint', tool_tip='Whether to apply the constraint'
        )
        self._max_expected_error_px_view_controller = DecimalLineEditParameterViewController(
            max_expected_error_px, tool_tip='Maximum expected error in pixels'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Degrees of Freedom:', self._degrees_of_freedom_view_controller.get_widget())
        layout.addRow(
            'Weight Update Interval:', self._weight_update_interval_view_controller.get_widget()
        )
        layout.addRow(self._apply_constraint_view_controller.get_widget())
        layout.addRow(
            'Max Expected Error [px]:', self._max_expected_error_px_view_controller.get_widget()
        )
        self.get_widget().setLayout(layout)


class PtyChiProbePositionsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbePositionSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.is_optimizable,
            'Optimize Probe Positions',
            tool_tip='Whether the probe positions are optimizable',
        )
        self._optimization_plan_view_controller = PtyChiOptimizationPlanViewController(
            settings.optimization_plan_start,
            settings.optimization_plan_stop,
            settings.optimization_plan_stride,
            num_epochs,
        )
        self._optimizer_view_controller = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._step_size_view_controller = DecimalLineEditParameterViewController(
            settings.step_size, tool_tip='Optimizer step size'
        )
        self._constrain_centroid_view_controller = CheckBoxParameterViewController(
            settings.constrain_centroid,
            'Constrain Centroid',
            tool_tip='Whether to subtract the mean from positions after updating positions',
        )
        self._correction_type_view_controller = ComboBoxParameterViewController(
            settings.correction_type,
            enumerators.position_correction_types(),
            tool_tip='Algorithm used to calculate the position correction update',
        )
        self._differentiation_method_view_controller = ComboBoxParameterViewController(
            settings.differentiation_method,
            enumerators.image_gradient_methods(),
            tool_tip='Method for calculating the object gradient',
        )
        self._cross_correlation_view_controller = PtyChiCrossCorrelationViewController(
            settings.correction_type,
            settings.cross_correlation_scale,
            settings.cross_correlation_real_space_width,
            settings.cross_correlation_probe_threshold,
        )
        self._slice_for_correction_view_controller = PtyChiSliceForCorrectionViewController(
            settings.choose_slice_for_correction,
            settings.slice_for_correction,
        )
        self._clip_update_magnitude_by_mad_view_controller = CheckBoxParameterViewController(
            settings.clip_update_magnitude_by_mad,
            'Clip Update Magnitude by MAD',
            tool_tip='Whether to clip the update magnitude by the median absolute deviation',
        )
        self._update_magnitude_limit_view_controller = PtyChiUpdateMagnitudeLimitViewController(
            settings.limit_update_magnitude,
            settings.update_magnitude_limit,
        )
        self._constrain_affine_transform_view_controller = (
            PtyChiConstrainAffineTransformViewController(
                settings.constrain_affine_transform,
                settings.constrain_affine_transform_start,
                settings.constrain_affine_transform_stop,
                settings.constrain_affine_transform_stride,
                settings.constrain_affine_transform_degrees_of_freedom,
                settings.constrain_affine_transform_position_weight_update_interval,
                settings.constrain_affine_transform_apply_constraint,
                settings.constrain_affine_transform_max_expected_error_px,
                num_epochs,
            )
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimization_plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        layout.addRow(self._constrain_centroid_view_controller.get_widget())
        layout.addRow('Correction Type:', self._correction_type_view_controller.get_widget())
        layout.addRow(
            'Differentiation Method:', self._differentiation_method_view_controller.get_widget()
        )
        layout.addRow(self._cross_correlation_view_controller.get_widget())
        layout.addRow(
            self._slice_for_correction_view_controller.get_label(),
            self._slice_for_correction_view_controller.get_widget(),
        )
        layout.addRow(self._clip_update_magnitude_by_mad_view_controller.get_widget())
        layout.addRow(
            self._update_magnitude_limit_view_controller.get_label(),
            self._update_magnitude_limit_view_controller.get_widget(),
        )
        layout.addRow(self._constrain_affine_transform_view_controller.get_widget())
        self.get_widget().setLayout(layout)
