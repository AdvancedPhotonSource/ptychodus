"""Parent-safe pty-chi algorithm layer: shared kwarg builders and per-algorithm specs.

This module is the single source of truth for the pty-chi algorithm registry.
Both the parent-side reconstructor factory (:func:`build_reconstructor_list`)
and the child-side task-options loader (:func:`ptychodus.model.ptychi.task.load_task_options`)
look up algorithms through :data:`ALGORITHMS`, keyed by pty-chi's own
``Reconstructors`` enum. Adding a new pty-chi engine means writing one
``PtyChiAlgorithm`` subclass and adding one entry to :data:`ALGORITHMS`.

The module imports ``ptychi.api`` (for the Options dataclasses and enums),
which pulls torch in for type annotations only -- no CUDA runtime is acquired.
That happens child-side when ``PtychographyTask`` is instantiated in
:mod:`ptychodus.model.ptychi.task`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, TypeVar

from ptychi.api import (
    AffineDegreesOfFreedom,
    AutodiffPtychographyOPRModeWeightsOptions,
    AutodiffPtychographyObjectOptions,
    AutodiffPtychographyOptions,
    AutodiffPtychographyProbeOptions,
    AutodiffPtychographyProbePositionOptions,
    AutodiffPtychographyReconstructorOptions,
    BHOPRModeWeightsOptions,
    BHObjectOptions,
    BHOptions,
    BHProbeOptions,
    BHProbePositionOptions,
    BHReconstructorOptions,
    BatchingModes,
    DMOPRModeWeightsOptions,
    DMObjectOptions,
    DMOptions,
    DMProbeOptions,
    DMProbePositionOptions,
    DMReconstructorOptions,
    Devices,
    Directions,
    Dtypes,
    EPIEOptions,
    EPIEReconstructorOptions,
    ForwardModels,
    ImageGradientMethods,
    ImageIntegrationMethods,
    LSQMLOPRModeWeightsOptions,
    LSQMLObjectOptions,
    LSQMLOptions,
    LSQMLProbeOptions,
    LSQMLProbePositionOptions,
    LSQMLReconstructorOptions,
    LossFunctions,
    MagPhaseComponents,
    NoiseModels,
    OPRWeightSmoothingMethods,
    ObjectPosOriginCoordsMethods,
    Optimizers,
    OrthogonalizationMethods,
    PIEOPRModeWeightsOptions,
    PIEObjectOptions,
    PIEOptions,
    PIEProbeOptions,
    PIEProbePositionOptions,
    PIEReconstructorOptions,
    PatchInterpolationMethods,
    PositionCorrectionTypes,
    ProbeSupportMethods,
    PtychographyDataOptions,
    RAAROPRModeWeightsOptions,
    RAARObjectOptions,
    RAAROptions,
    RAARProbeOptions,
    RAARProbePositionOptions,
    RAARReconstructorOptions,
    RPIEOptions,
    RPIEReconstructorOptions,
    Reconstructors,
)
from ptychi.api.options.base import (
    ForwardModelOptions,
    OPRModeWeightsSmoothingOptions,
    ObjectHardLimitsMagnitudePhase,
    ObjectL1NormConstraintOptions,
    ObjectL2NormConstraintOptions,
    ObjectMultisliceRegularizationOptions,
    ObjectSmoothnessConstraintOptions,
    ObjectTotalVariationOptions,
    OptimizationPlan,
    PositionAffineTransformConstraintOptions,
    PositionCorrectionOptions,
    ProbeCenterConstraintOptions,
    ProbeOrthogonalizeIncoherentModesOptions,
    ProbeOrthogonalizeOPRModesOptions,
    ProbePowerConstraintOptions,
    ProbeSupportConstraintOptions,
    RemoveGridArtifactsOptions,
    RemoveObjectProbeAmbiguityOptions,
    SliceSpacingOptions,
)
from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.object import Object
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import PtyChiPayload
from .affine import PtyChiAffineDegreesOfFreedom, PtyChiAffineDegreesOfFreedomBitField
from .settings import (
    PtyChiAutodiffSettings,
    PtyChiBHSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiRAARSettings,
    PtyChiSettings,
)

__all__ = ['ALGORITHMS', 'PtyChiAlgorithm', 'PtyChiCommon', 'build_reconstructor_list']

logger = logging.getLogger(__name__)


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychi._subprocess:run_reconstruct'


_E = TypeVar('_E', bound=Enum)


def _optimization_plan(start: int, stop: int, stride: int) -> OptimizationPlan:
    return OptimizationPlan(start, None if stop < 0 else stop, stride)


def _parse_enum(enum_cls: type[_E], text: str, default: _E) -> _E:
    try:
        return enum_cls[text.upper()]
    except KeyError:
        logger.warning(f'Failed to parse {enum_cls.__name__} "{text}"!')
        return default


def _parse_optimizer(text: str) -> Optimizers:
    return _parse_enum(Optimizers, text, Optimizers.SGD)


def _parse_smoothing_method(text: str) -> OPRWeightSmoothingMethods:
    # An unset OPR smoothing method is the legitimate 'disabled' default, not
    # a config error, so this stays quieter than the general _parse_enum path.
    try:
        return OPRWeightSmoothingMethods[text.upper()]
    except KeyError:
        logger.debug('OPR weight smoothing method is None.')
        return OPRWeightSmoothingMethods.MEDIAN


class PtyChiCommon:
    """Reads the five shared settings groups and returns kwarg dicts for the
    corresponding pty-chi ``*Options`` sub-object constructors.

    One instance per :class:`PtyChiReconstructorLibrary`, handed to every
    :class:`PtyChiAlgorithm`. The algorithm's ``build_task_options`` folds these
    common kwargs with per-algorithm extras via kwargs unpacking.
    """

    def __init__(
        self,
        reconstructor_settings: PtyChiSettings,
        object_settings: PtyChiObjectSettings,
        probe_settings: PtyChiProbeSettings,
        probe_position_settings: PtyChiProbePositionSettings,
        opr_settings: PtyChiOPRSettings,
    ) -> None:
        self._reconstructor = reconstructor_settings
        self._object = object_settings
        self._probe = probe_settings
        self._probe_position = probe_position_settings
        self._opr = opr_settings
        self._affine_dof = PtyChiAffineDegreesOfFreedomBitField(
            probe_position_settings.constrain_affine_transform_degrees_of_freedom
        )

    @property
    def num_epochs(self) -> int:
        return self._reconstructor.num_epochs.get_value()

    @property
    def num_sync_epochs(self) -> int:
        return self._reconstructor.num_sync_epochs.get_value()

    def data_options(self, metadata: ProductMetadata) -> PtychographyDataOptions:
        free_space_propagation_distance_m = (
            math.inf
            if self._reconstructor.use_far_field_propagation.get_value()
            else metadata.detector_distance_m
        )
        return PtychographyDataOptions(
            free_space_propagation_distance_m=free_space_propagation_distance_m,
            wavelength_m=metadata.probe_wavelength_m,
            fft_shift=self._reconstructor.fft_shift_diffraction_patterns.get_value(),
            save_data_on_device=self._reconstructor.save_data_on_device.get_value(),
        )

    def reconstructor_kwargs(self) -> dict[str, Any]:
        s = self._reconstructor
        blur_sigma = (
            s.diffraction_pattern_blur_sigma.get_value()
            if s.enable_diffraction_pattern_blur.get_value()
            else None
        )
        # pty-chi requires clustering stride >= 1 even when clustering is off
        # (the disabled sentinel in ptychodus settings is 0).
        clustering_raw = s.compact_mode_update_clustering.get_value()
        exclude_below = (
            s.exclude_measured_pixels_below.get_value()
            if s.enable_exclude_measured_pixels_below.get_value()
            else None
        )
        return {
            'num_epochs': s.num_epochs.get_value(),
            'batch_size': s.batch_size.get_value(),
            'batching_mode': _parse_enum(
                BatchingModes, s.batching_mode.get_value(), BatchingModes.RANDOM
            ),
            'compact_mode_update_clustering': clustering_raw > 0,
            'compact_mode_update_clustering_stride': max(1, clustering_raw),
            'default_device': Devices.GPU if s.use_devices.get_value() else Devices.CPU,
            'default_dtype': (
                Dtypes.FLOAT64 if s.use_double_precision.get_value() else Dtypes.FLOAT32
            ),
            'use_double_precision_for_fft': s.use_double_precision_for_fft.get_value(),
            'allow_nondeterministic_algorithms': s.allow_nondeterministic_algorithms.get_value(),
            'random_seed': None,
            'displayed_loss_function': LossFunctions.MSE_SQRT,
            'exclude_measured_pixels_below': exclude_below,
            'forward_model_options': ForwardModelOptions(
                low_memory_mode=s.use_low_memory_mode.get_value(),
                pad_for_shift=s.pad_for_shift.get_value(),
                diffraction_pattern_blur_sigma=blur_sigma,
            ),
        }

    def object_kwargs(self, object_: Object) -> dict[str, Any]:
        s = self._object
        # pty-chi types slice spacings + hard limits + origin as serializable
        # list/tuple, not ndarray. Test length rather than truthiness: a product
        # read back from HDF5 carries an ndarray, and `if array` raises for
        # every length but one.
        slice_spacings_m = object_.layer_spacing_m
        slice_spacings_out = list(slice_spacings_m) if len(slice_spacings_m) > 0 else None
        pixel_geometry = object_.get_pixel_geometry()
        return {
            'optimizable': s.is_optimizable.get_value(),
            'optimization_plan': _optimization_plan(
                s.optimization_plan_start.get_value(),
                s.optimization_plan_stop.get_value(),
                s.optimization_plan_stride.get_value(),
            ),
            'optimizer': _parse_optimizer(s.optimizer.get_value()),
            'step_size': s.step_size.get_value(),
            'optimizer_params': {},
            'slice_spacings_m': slice_spacings_out,
            'slice_spacing_options': SliceSpacingOptions(
                optimizable=s.optimize_slice_spacing.get_value(),
                optimization_plan=_optimization_plan(
                    s.optimize_slice_spacing_start.get_value(),
                    s.optimize_slice_spacing_stop.get_value(),
                    s.optimize_slice_spacing_stride.get_value(),
                ),
                optimizer=_parse_optimizer(s.optimize_slice_spacing_optimizer.get_value()),
                step_size=s.optimize_slice_spacing_step_size.get_value(),
            ),
            'pixel_size_m': pixel_geometry.width_m,
            'pixel_size_aspect_ratio': pixel_geometry.get_aspect_ratio(),
            'l1_norm_constraint': ObjectL1NormConstraintOptions(
                enabled=s.constrain_l1_norm.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_l1_norm_start.get_value(),
                    s.constrain_l1_norm_stop.get_value(),
                    s.constrain_l1_norm_stride.get_value(),
                ),
                weight=s.constrain_l1_norm_weight.get_value(),
            ),
            'l2_norm_constraint': ObjectL2NormConstraintOptions(
                enabled=s.constrain_l2_norm.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_l2_norm_start.get_value(),
                    s.constrain_l2_norm_stop.get_value(),
                    s.constrain_l2_norm_stride.get_value(),
                ),
                weight=s.constrain_l2_norm_weight.get_value(),
            ),
            'smoothness_constraint': ObjectSmoothnessConstraintOptions(
                enabled=s.constrain_smoothness.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_smoothness_start.get_value(),
                    s.constrain_smoothness_stop.get_value(),
                    s.constrain_smoothness_stride.get_value(),
                ),
                alpha=s.constrain_smoothness_alpha.get_value(),
            ),
            'total_variation': ObjectTotalVariationOptions(
                enabled=s.constrain_total_variation.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_total_variation_start.get_value(),
                    s.constrain_total_variation_stop.get_value(),
                    s.constrain_total_variation_stride.get_value(),
                ),
                weight=s.constrain_total_variation_weight.get_value(),
            ),
            'remove_grid_artifacts': RemoveGridArtifactsOptions(
                enabled=s.remove_grid_artifacts.get_value(),
                optimization_plan=_optimization_plan(
                    s.remove_grid_artifacts_start.get_value(),
                    s.remove_grid_artifacts_stop.get_value(),
                    s.remove_grid_artifacts_stride.get_value(),
                ),
                period_x_m=s.remove_grid_artifacts_period_x_m.get_value(),
                period_y_m=s.remove_grid_artifacts_period_y_m.get_value(),
                window_size=s.remove_grid_artifacts_window_size_px.get_value(),
                direction=_parse_enum(
                    Directions, s.remove_grid_artifacts_direction.get_value(), Directions.XY
                ),
                component=_parse_enum(
                    MagPhaseComponents,
                    s.remove_grid_artifacts_component.get_value(),
                    MagPhaseComponents.PHASE,
                ),
            ),
            'multislice_regularization': ObjectMultisliceRegularizationOptions(
                enabled=s.regularize_multislice.get_value(),
                optimization_plan=_optimization_plan(
                    s.regularize_multislice_start.get_value(),
                    s.regularize_multislice_stop.get_value(),
                    s.regularize_multislice_stride.get_value(),
                ),
                weight=s.regularize_multislice_weight.get_value(),
                unwrap_phase=s.regularize_multislice_unwrap_phase.get_value(),
                unwrap_image_grad_method=_parse_enum(
                    ImageGradientMethods,
                    s.regularize_multislice_unwrap_phase_image_gradient_method.get_value(),
                    ImageGradientMethods.FOURIER_SHIFT,
                ),
                unwrap_image_integration_method=_parse_enum(
                    ImageIntegrationMethods,
                    s.regularize_multislice_unwrap_phase_image_integration_method.get_value(),
                    ImageIntegrationMethods.DECONVOLUTION,
                ),
            ),
            'patch_interpolation_method': _parse_enum(
                PatchInterpolationMethods,
                s.patch_interpolation_method.get_value(),
                PatchInterpolationMethods.FOURIER,
            ),
            'remove_object_probe_ambiguity': RemoveObjectProbeAmbiguityOptions(
                enabled=s.remove_object_probe_ambiguity.get_value(),
                optimization_plan=_optimization_plan(
                    s.remove_object_probe_ambiguity_start.get_value(),
                    s.remove_object_probe_ambiguity_stop.get_value(),
                    s.remove_object_probe_ambiguity_stride.get_value(),
                ),
            ),
            'build_preconditioner_with_all_modes': s.build_preconditioner_with_all_modes.get_value(),
            'determine_position_origin_coords_by': ObjectPosOriginCoordsMethods.SPECIFIED,
            'position_origin_coords': [0.0, 0.0],
            'hard_limits_magnitude_phase': ObjectHardLimitsMagnitudePhase(
                enabled=s.constrain_hard_limits.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_hard_limits_start.get_value(),
                    s.constrain_hard_limits_stop.get_value(),
                    s.constrain_hard_limits_stride.get_value(),
                ),
                abs_lim=(
                    [
                        s.constrain_hard_limits_abs_min.get_value(),
                        s.constrain_hard_limits_abs_max.get_value(),
                    ]
                    if s.constrain_hard_limits_enable_abs.get_value()
                    else None
                ),
                phase_lim=(
                    [
                        math.radians(s.constrain_hard_limits_phase_min_deg.get_value()),
                        math.radians(s.constrain_hard_limits_phase_max_deg.get_value()),
                    ]
                    if s.constrain_hard_limits_enable_phase.get_value()
                    else None
                ),
            ),
        }

    def probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        s = self._probe
        return {
            'optimizable': s.is_optimizable.get_value(),
            'optimization_plan': _optimization_plan(
                s.optimization_plan_start.get_value(),
                s.optimization_plan_stop.get_value(),
                s.optimization_plan_stride.get_value(),
            ),
            'optimizer': _parse_optimizer(s.optimizer.get_value()),
            'step_size': s.step_size.get_value(),
            'optimizer_params': {},
            'power_constraint': ProbePowerConstraintOptions(
                enabled=s.constrain_probe_power.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_probe_power_start.get_value(),
                    s.constrain_probe_power_stop.get_value(),
                    s.constrain_probe_power_stride.get_value(),
                ),
                probe_power=metadata.probe_photon_count,
                scale_object=s.constrain_probe_power_scale_object.get_value(),
            ),
            'orthogonalize_incoherent_modes': ProbeOrthogonalizeIncoherentModesOptions(
                enabled=s.orthogonalize_incoherent_modes.get_value(),
                optimization_plan=_optimization_plan(
                    s.orthogonalize_incoherent_modes_start.get_value(),
                    s.orthogonalize_incoherent_modes_stop.get_value(),
                    s.orthogonalize_incoherent_modes_stride.get_value(),
                ),
                method=_parse_enum(
                    OrthogonalizationMethods,
                    s.orthogonalize_incoherent_modes_method.get_value(),
                    OrthogonalizationMethods.GS,
                ),
                sort_by_occupancy=s.orthogonalize_incoherent_modes_sort_by_occupancy.get_value(),
            ),
            'orthogonalize_opr_modes': ProbeOrthogonalizeOPRModesOptions(
                enabled=s.orthogonalize_opr_modes.get_value(),
                optimization_plan=_optimization_plan(
                    s.orthogonalize_opr_modes_start.get_value(),
                    s.orthogonalize_opr_modes_stop.get_value(),
                    s.orthogonalize_opr_modes_stride.get_value(),
                ),
            ),
            'support_constraint': ProbeSupportConstraintOptions(
                enabled=s.constrain_support.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_support_start.get_value(),
                    s.constrain_support_stop.get_value(),
                    s.constrain_support_stride.get_value(),
                ),
                threshold=s.constrain_support_threshold.get_value(),
                fixed_probe_support=_parse_enum(
                    ProbeSupportMethods,
                    s.constrain_support_method.get_value(),
                    ProbeSupportMethods.NONE,
                ),
            ),
            'center_constraint': ProbeCenterConstraintOptions(
                enabled=s.constrain_center.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_center_start.get_value(),
                    s.constrain_center_stop.get_value(),
                    s.constrain_center_stride.get_value(),
                ),
                use_total_intensity_for_com=s.use_total_intensity_for_com.get_value(),
                center_modes_individually=s.constrain_center_modes_individually.get_value(),
            ),
            'eigenmode_update_relaxation': s.eigenmode_update_relaxation.get_value(),
        }

    def probe_position_kwargs(self) -> dict[str, Any]:
        s = self._probe_position
        slice_for_correction = (
            s.slice_for_correction.get_value()
            if s.choose_slice_for_correction.get_value()
            else None
        )
        update_magnitude_limit = (
            s.update_magnitude_limit.get_value() if s.limit_update_magnitude.get_value() else None
        )
        degrees_of_freedom: list[AffineDegreesOfFreedom] = []
        for bit, member in (
            (PtyChiAffineDegreesOfFreedom.TRANSLATION, AffineDegreesOfFreedom.TRANSLATION),
            (PtyChiAffineDegreesOfFreedom.ROTATION, AffineDegreesOfFreedom.ROTATION),
            (PtyChiAffineDegreesOfFreedom.SCALING, AffineDegreesOfFreedom.SCALE),
            (PtyChiAffineDegreesOfFreedom.SHEARING, AffineDegreesOfFreedom.SHEAR),
            (PtyChiAffineDegreesOfFreedom.ASYMMETRY, AffineDegreesOfFreedom.ASYMMETRY),
        ):
            if self._affine_dof.is_bit_set(bit):
                degrees_of_freedom.append(member)
        override_update_flexibility = (
            s.constrain_affine_transform_override_update_flexibility.get_value()
            if s.override_affine_transform_update_flexibility.get_value()
            else None
        )
        return {
            'optimizable': s.is_optimizable.get_value(),
            'optimization_plan': _optimization_plan(
                s.optimization_plan_start.get_value(),
                s.optimization_plan_stop.get_value(),
                s.optimization_plan_stride.get_value(),
            ),
            'optimizer': _parse_optimizer(s.optimizer.get_value()),
            'step_size': s.step_size.get_value(),
            'optimizer_params': {},
            'constrain_position_mean': s.constrain_position_mean.get_value(),
            'correction_options': PositionCorrectionOptions(
                correction_type=_parse_enum(
                    PositionCorrectionTypes,
                    s.correction_type.get_value(),
                    PositionCorrectionTypes.GRADIENT,
                ),
                differentiation_method=_parse_enum(
                    ImageGradientMethods,
                    s.differentiation_method.get_value(),
                    ImageGradientMethods.FOURIER_DIFFERENTIATION,
                ),
                cross_correlation_scale=s.cross_correlation_scale.get_value(),
                cross_correlation_real_space_width=s.cross_correlation_real_space_width.get_value(),
                cross_correlation_probe_threshold=s.cross_correlation_probe_threshold.get_value(),
                slice_for_correction=slice_for_correction,  # type: ignore
                clip_update_magnitude_by_mad=s.clip_update_magnitude_by_mad.get_value(),
                update_magnitude_limit=update_magnitude_limit,
            ),
            'affine_transform_constraint': PositionAffineTransformConstraintOptions(
                enabled=s.constrain_affine_transform.get_value(),
                optimization_plan=_optimization_plan(
                    s.constrain_affine_transform_start.get_value(),
                    s.constrain_affine_transform_stop.get_value(),
                    s.constrain_affine_transform_stride.get_value(),
                ),
                degrees_of_freedom=degrees_of_freedom,
                position_weight_update_interval=s.constrain_affine_transform_position_weight_update_interval.get_value(),
                apply_constraint=s.constrain_affine_transform_apply_constraint.get_value(),
                max_expected_error=s.constrain_affine_transform_max_expected_error_px.get_value(),
                override_update_flexibility=override_update_flexibility,
            ),
        }

    def opr_kwargs(self) -> dict[str, Any]:
        s = self._opr
        primary_floor = (
            s.primary_mode_weight_floor.get_value()
            if s.enable_primary_mode_weight_floor.get_value()
            else None
        )
        return {
            'optimizable': s.is_optimizable.get_value(),
            'optimization_plan': _optimization_plan(
                s.optimization_plan_start.get_value(),
                s.optimization_plan_stop.get_value(),
                s.optimization_plan_stride.get_value(),
            ),
            'optimizer': _parse_optimizer(s.optimizer.get_value()),
            'step_size': s.step_size.get_value(),
            'optimizer_params': {},
            'primary_mode_weight_floor': primary_floor,
            'optimize_eigenmode_weights': s.optimize_eigenmode_weights.get_value(),
            'optimize_intensity_variation': s.optimize_intensity_variation.get_value(),
            'smoothing': OPRModeWeightsSmoothingOptions(
                enabled=s.smooth_mode_weights.get_value(),
                optimization_plan=_optimization_plan(
                    s.smooth_mode_weights_start.get_value(),
                    s.smooth_mode_weights_stop.get_value(),
                    s.smooth_mode_weights_stride.get_value(),
                ),
                method=_parse_smoothing_method(s.smoothing_method.get_value()),
                polynomial_degree=s.polynomial_smoothing_degree.get_value(),
            ),
            'update_relaxation': s.update_relaxation.get_value(),
        }


@dataclass(frozen=True)
class _Spec:
    """Names the six pty-chi ``*Options`` classes plus enum + display name."""

    reconstructor: Reconstructors
    display_name: str
    task_options_cls: type[PtychographyTaskOptions]
    reconstructor_options_cls: type
    object_options_cls: type
    probe_options_cls: type
    probe_position_options_cls: type
    opr_options_cls: type


class PtyChiAlgorithm:
    """One pty-chi algorithm's parent-side view.

    Subclasses supply ``spec`` (the six Options classes to instantiate) and, if
    the algorithm has fields on top of the base ``*Options``, override the
    matching ``_extra_*_kwargs`` hook. Base hooks return empty dicts, so an
    algorithm that only reuses the base fields is a two-line subclass.
    """

    spec: ClassVar[_Spec]

    def __init__(self, common: PtyChiCommon, algorithm_settings: Any) -> None:
        self._common = common
        self._settings = algorithm_settings

    def build_task_options(self, product: Product) -> PtychographyTaskOptions:
        s = self.spec
        return s.task_options_cls(
            data_options=self._common.data_options(product.metadata),
            reconstructor_options=s.reconstructor_options_cls(
                **self._common.reconstructor_kwargs(),
                **self._extra_reconstructor_kwargs(),
            ),
            object_options=s.object_options_cls(
                **self._common.object_kwargs(product.object_),
                **self._extra_object_kwargs(product.object_),
            ),
            probe_options=s.probe_options_cls(
                **self._common.probe_kwargs(product.metadata),
                **self._extra_probe_kwargs(product.metadata),
            ),
            probe_position_options=s.probe_position_options_cls(
                **self._common.probe_position_kwargs(),
                **self._extra_probe_position_kwargs(),
            ),
            opr_mode_weight_options=s.opr_options_cls(**self._common.opr_kwargs()),
        )

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        return {}

    def _extra_object_kwargs(self, object_: Object) -> dict[str, Any]:
        return {}

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {}

    def _extra_probe_position_kwargs(self) -> dict[str, Any]:
        return {}


class DMAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.DM,
        'DM',
        DMOptions,
        DMReconstructorOptions,
        DMObjectOptions,
        DMProbeOptions,
        DMProbePositionOptions,
        DMOPRModeWeightsOptions,
    )
    _settings: PtyChiDMSettings

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        return {
            'exit_wave_update_relaxation': self._settings.exit_wave_update_relaxation.get_value(),
            'chunk_length': self._settings.chunk_length.get_value(),
        }

    def _extra_object_kwargs(self, object_: Object) -> dict[str, Any]:
        return {
            'amplitude_clamp_limit': self._settings.object_amplitude_clamp_limit.get_value(),
            'inertia': self._settings.object_inertia.get_value(),
        }

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {'inertia': self._settings.probe_inertia.get_value()}


class RAARAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.RAAR,
        'RAAR',
        RAAROptions,
        RAARReconstructorOptions,
        RAARObjectOptions,
        RAARProbeOptions,
        RAARProbePositionOptions,
        RAAROPRModeWeightsOptions,
    )
    _settings: PtyChiRAARSettings

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        return {
            'beta': self._settings.beta.get_value(),
            'chunk_length': self._settings.chunk_length.get_value(),
        }

    def _extra_object_kwargs(self, object_: Object) -> dict[str, Any]:
        return {
            'amplitude_clamp_limit': self._settings.object_amplitude_clamp_limit.get_value(),
            'inertia': self._settings.object_inertia.get_value(),
        }

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {'inertia': self._settings.probe_inertia.get_value()}


class PIEAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.PIE,
        'PIE',
        PIEOptions,
        PIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )
    _settings: PtyChiPIESettings

    def _extra_object_kwargs(self, object_: Object) -> dict[str, Any]:
        return {'alpha': self._settings.object_alpha.get_value()}

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {'alpha': self._settings.probe_alpha.get_value()}


class EPIEAlgorithm(PIEAlgorithm):
    spec = _Spec(
        Reconstructors.EPIE,
        'ePIE',
        EPIEOptions,
        EPIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )


class RPIEAlgorithm(PIEAlgorithm):
    spec = _Spec(
        Reconstructors.RPIE,
        'rPIE',
        RPIEOptions,
        RPIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )


class LSQMLAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.LSQML,
        'LSQML',
        LSQMLOptions,
        LSQMLReconstructorOptions,
        LSQMLObjectOptions,
        LSQMLProbeOptions,
        LSQMLProbePositionOptions,
        LSQMLOPRModeWeightsOptions,
    )
    _settings: PtyChiLSQMLSettings

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        s = self._settings
        mixing_factor = (
            None
            if s.auto_momentum_acceleration_gradient_mixing_factor.get_value()
            else s.momentum_acceleration_gradient_mixing_factor.get_value()
        )
        return {
            'noise_model': _parse_enum(
                NoiseModels, s.noise_model.get_value(), NoiseModels.GAUSSIAN
            ),
            'gaussian_noise_std': s.gaussian_noise_std.get_value(),
            'single_slice_solve_obj_prb_step_size_jointly': s.single_slice_solve_object_probe_step_size_jointly.get_value(),
            'multislice_solve_obj_prb_step_size_jointly': s.multislice_solve_object_probe_step_size_jointly.get_value(),
            'solve_step_sizes_only_using_first_probe_mode': s.solve_step_sizes_only_using_first_probe_mode.get_value(),
            'momentum_acceleration_gain': s.momentum_acceleration_gain.get_value(),
            'momentum_acceleration_gradient_mixing_factor': mixing_factor,
            'rescale_probe_intensity_in_first_epoch': s.rescale_probe_intensity_in_first_epoch.get_value(),
            'preconditioning_damping_factor': s.preconditioning_damping_factor.get_value(),
        }

    def _extra_object_kwargs(self, object_: Object) -> dict[str, Any]:
        return {
            'optimal_step_size_scaler': self._settings.object_optimal_step_size_scaler.get_value(),
            'multimodal_update': self._settings.object_multimodal_update.get_value(),
        }

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {
            'optimal_step_size_scaler': self._settings.probe_optimal_step_size_scaler.get_value()
        }

    def _extra_probe_position_kwargs(self) -> dict[str, Any]:
        s = self._settings
        mixing_factor = (
            None
            if s.probe_position_auto_momentum_gradient_mixing_factor.get_value()
            else s.probe_position_momentum_acceleration_gradient_mixing_factor.get_value()
        )
        return {
            'momentum_acceleration_gain': s.probe_position_momentum_acceleration_gain.get_value(),
            'momentum_acceleration_gradient_mixing_factor': mixing_factor,
            'momentum_acceleration_memory': s.probe_position_momentum_acceleration_memory.get_value(),
        }


class AutodiffAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.AD_PTYCHO,
        'Autodiff',
        AutodiffPtychographyOptions,
        AutodiffPtychographyReconstructorOptions,
        AutodiffPtychographyObjectOptions,
        AutodiffPtychographyProbeOptions,
        AutodiffPtychographyProbePositionOptions,
        AutodiffPtychographyOPRModeWeightsOptions,
    )
    _settings: PtyChiAutodiffSettings

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        s = self._settings
        return {
            'loss_function': _parse_enum(
                LossFunctions, s.loss_function.get_value(), LossFunctions.MSE_SQRT
            ),
            'forward_model_class': _parse_enum(
                ForwardModels,
                s.forward_model_class.get_value(),
                ForwardModels.PLANAR_PTYCHOGRAPHY,
            ),
            'forward_model_params': None,
        }


class BHAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        Reconstructors.BH,
        'BH',
        BHOptions,
        BHReconstructorOptions,
        BHObjectOptions,
        BHProbeOptions,
        BHProbePositionOptions,
        BHOPRModeWeightsOptions,
    )
    _settings: PtyChiBHSettings

    def _extra_reconstructor_kwargs(self) -> dict[str, Any]:
        return {'method': self._settings.method.get_value()}

    def _extra_probe_kwargs(self, metadata: ProductMetadata) -> dict[str, Any]:
        return {'rho': self._settings.probe_rho.get_value()}

    def _extra_probe_position_kwargs(self) -> dict[str, Any]:
        return {'rho': self._settings.probe_position_rho.get_value()}


ALGORITHMS: Final[Mapping[Reconstructors, type[PtyChiAlgorithm]]] = {
    Reconstructors.DM: DMAlgorithm,
    Reconstructors.RAAR: RAARAlgorithm,
    Reconstructors.PIE: PIEAlgorithm,
    Reconstructors.EPIE: EPIEAlgorithm,
    Reconstructors.RPIE: RPIEAlgorithm,
    Reconstructors.LSQML: LSQMLAlgorithm,
    Reconstructors.AD_PTYCHO: AutodiffAlgorithm,
    Reconstructors.BH: BHAlgorithm,
}


def build_reconstructor_list(
    common: PtyChiCommon,
    settings_by_reconstructor: Mapping[Reconstructors, Any],
) -> list[SubprocessReconstructor]:
    """Build one :class:`SubprocessReconstructor` per pty-chi algorithm."""

    reconstructors: list[SubprocessReconstructor] = []

    def progress_goal() -> int:
        return common.num_epochs

    for reconstructor, algo_cls in ALGORITHMS.items():
        algorithm = algo_cls(common, settings_by_reconstructor[reconstructor])

        def build_payload(
            parameters: ReconstructInput,
            _loaded_model_path: Path | None,
            _algorithm: PtyChiAlgorithm = algorithm,
            _common: PtyChiCommon = common,
        ) -> PtyChiPayload:
            return PtyChiPayload(
                reconstruct_input=parameters,
                task_options=_algorithm.build_task_options(parameters.product),
                num_sync_epochs=_common.num_sync_epochs,
            )

        reconstructors.append(
            SubprocessReconstructor(
                name=algorithm.spec.display_name,
                reconstruct_entry_point=_RECONSTRUCT_ENTRY,
                progress_goal_fn=progress_goal,
                build_reconstruct_payload=build_payload,
            )
        )

    return reconstructors
