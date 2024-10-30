from ptychodus.api.observer import Observable, Observer

from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ...model.pty_chi import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)
from ..parametric import (
    ParameterViewController,
)


class PtyChiReconstructorViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiReconstructorSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('FIXME')

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


# reconstructor_options = PIEReconstructorOptions(
#     num_epochs=self._settings.numEpochs.getValue(),
#     batch_size=self._settings.batchSize.getValue(),
#     default_device=default_device,
#     gpu_indices=(),  # TODO Sequence[int]
#     default_dtype=default_dtype,
#     random_seed=None,  # TODO
#     displayed_loss_function=None,  # TODO
#     log_level=logging.INFO,  # TODO
# )


class PtyChiObjectViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiObjectSettings) -> None:
        super().__init__()
        self._settings = settings
        self._widget = QGroupBox('FIXME')

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
        self._widget = QGroupBox('FIXME')

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
        self._widget = QGroupBox('FIXME')

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
        self._widget = QGroupBox('FIXME')

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
