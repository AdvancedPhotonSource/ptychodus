from typing import Any

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerSequenceParameter

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListView,
    QRadioButton,
    QWidget,
)

from ...model.pty_chi import (
    PtyChiDeviceRepository,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)
from ..parametric import (
    ParameterViewController,
    SpinBoxParameterViewController,
)


class PtyChiDeviceListModel(QAbstractListModel):
    def __init__(self, repository: PtyChiDeviceRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
        self._checkedRows: set[int] = set()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return self._repository[index.row()]
            elif role == Qt.ItemDataRole.CheckStateRole:
                return (
                    Qt.CheckState.Checked
                    if index.row() in self._checkedRows
                    else Qt.CheckState.Unchecked
                )

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.CheckStateRole:
            if value == Qt.CheckState.Checked:
                self._checkedRows.add(index.row())
            else:
                self._checkedRows.discard(index.row())

            self.dataChanged.emit(index, index)

            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)


class PtyChiDeviceParameterViewController(ParameterViewController, Observer):
    def __init__(
        self,
        useDevices: BooleanParameter,
        devices: IntegerSequenceParameter,
        repository: PtyChiDeviceRepository,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._useDevices = useDevices
        self._devices = devices
        self._repository = repository
        self._listModel = PtyChiDeviceListModel(repository)
        self._widget = QListView()
        self._widget.setModel(self._listModel)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._syncModelToView()
        useDevices.addObserver(self)
        devices.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME update checked indexes, etc.

    def update(self, observable: Observable) -> None:
        if observable in (self._useDevices, self._devices):
            self._syncModelToView()


class PtyChiPrecisionParameterViewController(ParameterViewController, Observer):
    def __init__(self, useDoublePrecision: BooleanParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._useDoublePrecision = useDoublePrecision
        self._singlePrecisionButton = QRadioButton('Single')
        self._doublePrecisionButton = QRadioButton('Double')
        self._buttonGroup = QButtonGroup()
        self._widget = QWidget()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._buttonGroup.addButton(self._singlePrecisionButton, 1)
        self._buttonGroup.addButton(self._doublePrecisionButton, 2)
        self._buttonGroup.setExclusive(True)
        self._buttonGroup.idToggled.connect(self._syncViewToModel)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._singlePrecisionButton)
        layout.addWidget(self._doublePrecisionButton)
        layout.addStretch()
        self._widget.setLayout(layout)

        self._syncModelToView()
        useDoublePrecision.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, toolId: int, checked: bool) -> None:
        if toolId == 2:
            self._useDoublePrecision.setValue(checked)

    def _syncModelToView(self) -> None:
        button = self._buttonGroup.button(2 if self._useDoublePrecision.getValue() else 1)
        button.setChecked(True)

    def update(self, observable: Observable) -> None:
        if observable is self._useDoublePrecision:
            self._syncModelToView()


class PtyChiReconstructorViewController(ParameterViewController):
    def __init__(
        self, settings: PtyChiReconstructorSettings, repository: PtyChiDeviceRepository
    ) -> None:
        super().__init__()
        self._numEpochsViewController = SpinBoxParameterViewController(
            settings.numEpochs, tool_tip='The number of epochs to run.'
        )
        self._batchSizeViewController = SpinBoxParameterViewController(
            settings.batchSize,
            tool_tip='The number of probe positions to process in each minibatch.',
        )
        self._deviceViewController = PtyChiDeviceParameterViewController(
            settings.useDevices,
            settings.devices,
            repository,
            tool_tip='The devices to use for computation.',
        )
        self._precisionViewController = PtyChiPrecisionParameterViewController(
            settings.useDoublePrecision,
            tool_tip='The floating point precision to use for computation.',
        )
        # TODO random_seed
        # TODO displayed_loss_function
        # TODO log_level
        self._widget = QGroupBox('Reconstructor')

        layout = QFormLayout()
        layout.addRow('Number of Epochs:', self._numEpochsViewController.getWidget())
        layout.addRow('Batch Size:', self._batchSizeViewController.getWidget())
        layout.addRow('Devices:', self._deviceViewController.getWidget())
        layout.addRow('Precision:', self._precisionViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtyChiObjectViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiObjectSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('Object')

        layout = QFormLayout()
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._settings.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncModelToView()


# # optimization plan: start, stop, stride; optimizable, optimizer, step_size
# object_options = PIEObjectOptions(
#     optimizable=self._objectSettings.isOptimizable.getValue(),  # TODO optimizer_params
#     optimization_plan=object_optimization_plan,
#     optimizer=object_optimizer,
#     step_size=self._objectSettings.stepSize.getValue(),
#     initial_guess=object_in.array,
#     slice_spacings_m=None,  # TODO Optional[ndarray]
#     pixel_size_m=pixel_size_m,
#     l1_norm_constraint_weight=self._objectSettings.l1NormConstraintWeight.getValue(),
#     l1_norm_constraint_stride=self._objectSettings.l1NormConstraintStride.getValue(),
#     smoothness_constraint_alpha=self._objectSettings.smoothnessConstraintAlpha.getValue(),
#     smoothness_constraint_stride=self._objectSettings.smoothnessConstraintStride.getValue(),
#     total_variation_weight=self._objectSettings.totalVariationWeight.getValue(),
#     total_variation_stride=self._objectSettings.totalVaritionStride.getValue(),
# )


class PtyChiProbeViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiProbeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('Probe')

        layout = QFormLayout()
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._settings.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncModelToView()


# probe_optimization_plan = self._create_optimization_plan(
#     self._probeSettings.optimizationPlanStart.getValue(),
#     self._probeSettings.optimizationPlanStop.getValue(),
#     self._probeSettings.optimizationPlanStride.getValue(),
# )
# probe_optimizer = self._create_optimizer(self._probeSettings.optimizer.getValue())
# probe_orthogonalize_incoherent_modes_method = self._create_orthogonalization_method(
#     self._probeSettings.orthogonalizeIncoherentModesMethod.getValue()
# )
# probe_options = PIEProbeOptions(
#     optimizable=self._probeSettings.isOptimizable.getValue(),  # TODO optimizer_params
#     optimization_plan=probe_optimization_plan,
#     optimizer=probe_optimizer,
#     step_size=self._probeSettings.stepSize.getValue(),
#     initial_guess=probe_in.array,
#     probe_power=self._probeSettings.probePower.getValue(),
#     probe_power_constraint_stride=self._probeSettings.probePowerConstraintStride.getValue(),
#     orthogonalize_incoherent_modes=self._probeSettings.orthogonalizeIncoherentModes.getValue(),
#     orthogonalize_incoherent_modes_stride=self._probeSettings.orthogonalizeIncoherentModesStride.getValue(),
#     orthogonalize_incoherent_modes_method=probe_orthogonalize_incoherent_modes_method,
#     orthogonalize_opr_modes=self._probeSettings.orthogonalizeOPRModes.getValue(),
#     orthogonalize_opr_modes_stride=self._probeSettings.orthogonalizeOPRModesStride.getValue(),
# )


class PtyChiProbePositionViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiProbePositionSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('Probe Positions')

        layout = QFormLayout()
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._settings.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncModelToView()


# probe_position_optimization_plan = self._create_optimization_plan(
#     self._probePositionSettings.optimizationPlanStart.getValue(),
#     self._probePositionSettings.optimizationPlanStop.getValue(),
#     self._probePositionSettings.optimizationPlanStride.getValue(),
# )
# probe_position_optimizer = self._create_optimizer(
#     self._probePositionSettings.optimizer.getValue()
# )
# probe_position_options = PIEProbePositionOptions(
#     optimizable=self._probePositionSettings.isOptimizable.getValue(),
#     optimization_plan=probe_position_optimization_plan,
#     optimizer=probe_position_optimizer,
#     step_size=self._probePositionSettings.stepSize.getValue(),
#     position_x_px=position_in_px[:, -1],
#     position_y_px=position_in_px[:, -2],
#     update_magnitude_limit=None,  # TODO Optional[float]
# )


class PtyChiOPRViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiOPRSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('OPR')

        layout = QFormLayout()
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._settings.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncModelToView()


# opr_optimization_plan = self._create_optimization_plan(
#     self._oprSettings.optimizationPlanStart.getValue(),
#     self._oprSettings.optimizationPlanStop.getValue(),
#     self._oprSettings.optimizationPlanStride.getValue(),
# )
# opr_optimizer = self._create_optimizer(self._oprSettings.optimizer.getValue())
# opr_weights = numpy.array([0.0])  # FIXME
# opr_mode_weight_options = PIEOPRModeWeightsOptions(
#     optimizable=self._oprSettings.isOptimizable.getValue(),
#     optimization_plan=opr_optimization_plan,
#     optimizer=opr_optimizer,
#     step_size=self._oprSettings.stepSize.getValue(),
#     initial_weights=opr_weights,
#     optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
#     optimize_intensity_variation=self._oprSettings.optimizeIntensityVariation.getValue(),
# )
