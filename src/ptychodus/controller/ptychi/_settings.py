"""The pty-chi panels' view of the settings generated from the backend's declaration.

**Transitional, with the panels it serves.** pty-chi's panel is laid out by
hand, one widget per named setting; the values behind those names are now
generated from what the backend declares and arrive as a flat mapping keyed by
persisted name. This pairs the two, so the panels bind to the same attributes
they always did, and it is deleted when they are.

Every attribute below was generated from the hand-written settings classes it
replaces, so no name was transcribed. What is checked rather than assumed is
the kind: asking for a key the declaration now holds as a different type fails
here, at construction, instead of rendering the wrong editor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from ptychodus.api.parameters import (
    BooleanParameter,
    IntegerParameter,
    Parameter,
    RealParameter,
    StringParameter,
)
from ptychodus.model.processing import DescribedField, ProcessingParameterStore

__all__ = [
    'PtyChiAlgorithmSettings',
    'PtyChiSettingsSource',
]


class _Group:
    """One settings group, read by persisted key and narrowed to a kind."""

    def __init__(self, name: str, fields: Mapping[str, DescribedField]) -> None:
        self._name = name
        self._fields = fields

    def _parameter(self, key: str, kind: type) -> Any:
        try:
            field = self._fields[key]
        except KeyError:
            raise KeyError(f'{self._name} has no setting "{key}".') from None

        if not isinstance(field.parameter, kind):
            raise TypeError(
                f'{self._name}/{key} is a {type(field.parameter).__name__}, not a {kind.__name__}.'
            )

        return field.parameter

    def boolean(self, key: str) -> BooleanParameter:
        return self._parameter(key, BooleanParameter)

    def integer(self, key: str) -> IntegerParameter:
        return self._parameter(key, IntegerParameter)

    def real(self, key: str) -> RealParameter:
        return self._parameter(key, RealParameter)

    def string(self, key: str) -> StringParameter:
        return self._parameter(key, StringParameter)


class PtyChiSettingsSource:
    """Every live setting behind one pty-chi algorithm, and what each accepts.

    A group a given algorithm does not have -- another algorithm's own -- reads
    as empty rather than missing, so a panel asking for it fails on the key it
    actually wanted.
    """

    def __init__(self, store: ProcessingParameterStore, library: str, algorithm: str) -> None:
        self._groups: dict[str, Mapping[str, DescribedField]] = {}
        self._described: dict[int, DescribedField] = {}

        for group in store.describe(library, algorithm):
            fields = {field.described.ini_key: field for field in group.fields}
            self._groups[group.name] = fields
            self._described.update({id(field.parameter): field for field in group.fields})

    def group(self, name: str) -> _Group:
        return _Group(name, self._groups.get(name, {}))

    def choices_of(self, parameter: Parameter[Any]) -> Sequence[str]:
        """The values *parameter* accepts, as the settings file spells them.

        Taken from the declaration rather than from a list kept beside it: a
        field typed as an enum already names its members, and the key records
        whether they are written by name or by value.
        """
        field = self._described.get(id(parameter))

        if field is None:
            return ()

        described = field.described

        if described.choices:
            return described.choices

        annotation = described.annotation

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return [
                member.name if described.enum_by_name else str(member.value)
                for member in annotation
            ]

        return ()


class PtyChiSettings:
    """The ``PtyChi`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChi')
        self.num_epochs = group.integer('NumEpochs')
        self.num_sync_epochs = group.integer('NumSyncEpochs')
        self.batch_size = group.integer('BatchSize')
        self.batching_mode = group.string('BatchingMode')
        self.compact_mode_update_clustering = group.integer('CompactModeUpdateClustering')
        self.use_double_precision = group.boolean('UseDoublePrecision')
        self.use_double_precision_for_fft = group.boolean('UseDoublePrecisionForFFT')
        self.allow_nondeterministic_algorithms = group.boolean('AllowNondeterministicAlgorithms')
        self.use_low_memory_mode = group.boolean('UseLowMemoryMode')
        self.pad_for_shift = group.integer('PadForShift')
        self.enable_diffraction_pattern_blur = group.boolean('EnableDiffractionPatternBlur')
        self.diffraction_pattern_blur_sigma = group.real('DiffractionPatternBlurSigma')
        self.enable_exclude_measured_pixels_below = group.boolean(
            'EnableExcludeMeasuredPixelsBelow'
        )
        self.exclude_measured_pixels_below = group.real('ExcludeMeasuredPixelsBelow')
        self.fft_shift_diffraction_patterns = group.boolean('FFTShiftDiffractionPatterns')
        self.save_data_on_device = group.boolean('SaveDataOnDevice')


class PtyChiObjectSettings:
    """The ``PtyChiObject`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiObject')
        self.is_optimizable = group.boolean('IsOptimizable')
        self.optimization_plan_start = group.integer('OptimizationPlanStart')
        self.optimization_plan_stop = group.integer('OptimizationPlanStop')
        self.optimization_plan_stride = group.integer('OptimizationPlanStride')
        self.optimizer = group.string('Optimizer')
        self.step_size = group.real('StepSize')
        self.optimize_slice_spacing = group.boolean('OptimizeSliceSpacing')
        self.optimize_slice_spacing_start = group.integer('OptimizeSliceSpacingStart')
        self.optimize_slice_spacing_stop = group.integer('OptimizeSliceSpacingStop')
        self.optimize_slice_spacing_stride = group.integer('OptimizeSliceSpacingStride')
        self.optimize_slice_spacing_optimizer = group.string('OptimizeSliceSpacingOptimizer')
        self.optimize_slice_spacing_step_size = group.real('OptimizeSliceSpacingStepSize')
        self.constrain_l1_norm = group.boolean('ConstrainL1Norm')
        self.constrain_l1_norm_start = group.integer('ConstrainL1NormStart')
        self.constrain_l1_norm_stop = group.integer('ConstrainL1NormStop')
        self.constrain_l1_norm_stride = group.integer('ConstrainL1NormStride')
        self.constrain_l1_norm_weight = group.real('ConstrainL1NormWeight')
        self.constrain_l2_norm = group.boolean('ConstrainL2Norm')
        self.constrain_l2_norm_start = group.integer('ConstrainL2NormStart')
        self.constrain_l2_norm_stop = group.integer('ConstrainL2NormStop')
        self.constrain_l2_norm_stride = group.integer('ConstrainL2NormStride')
        self.constrain_l2_norm_weight = group.real('ConstrainL2NormWeight')
        self.constrain_smoothness = group.boolean('ConstrainSmoothness')
        self.constrain_smoothness_start = group.integer('ConstrainSmoothnessStart')
        self.constrain_smoothness_stop = group.integer('ConstrainSmoothnessStop')
        self.constrain_smoothness_stride = group.integer('ConstrainSmoothnessStride')
        self.constrain_smoothness_alpha = group.real('ConstrainSmoothnessAlpha')
        self.constrain_total_variation = group.boolean('ConstrainTotalVariation')
        self.constrain_total_variation_start = group.integer('ConstrainTotalVariationStart')
        self.constrain_total_variation_stop = group.integer('ConstrainTotalVariationStop')
        self.constrain_total_variation_stride = group.integer('ConstrainTotalVariationStride')
        self.constrain_total_variation_weight = group.real('ConstrainTotalVariationWeight')
        self.remove_grid_artifacts = group.boolean('RemoveGridArtifacts')
        self.remove_grid_artifacts_start = group.integer('RemoveGridArtifactsStart')
        self.remove_grid_artifacts_stop = group.integer('RemoveGridArtifactsStop')
        self.remove_grid_artifacts_stride = group.integer('RemoveGridArtifactsStride')
        self.remove_grid_artifacts_period_x_m = group.real('RemoveGridArtifactsPeriodXInMeters')
        self.remove_grid_artifacts_period_y_m = group.real('RemoveGridArtifactsPeriodYInMeters')
        self.remove_grid_artifacts_window_size_px = group.integer(
            'RemoveGridArtifactsWindowSizeInPixels'
        )
        self.remove_grid_artifacts_direction = group.string('RemoveGridArtifactsDirection')
        self.remove_grid_artifacts_component = group.string('RemoveGridArtifactsComponent')
        self.regularize_multislice = group.boolean('RegularizeMultislice')
        self.regularize_multislice_start = group.integer('RegularizeMultisliceStart')
        self.regularize_multislice_stop = group.integer('RegularizeMultisliceStop')
        self.regularize_multislice_stride = group.integer('RegularizeMultisliceStride')
        self.regularize_multislice_weight = group.real('RegularizeMultisliceWeight')
        self.regularize_multislice_unwrap_phase = group.boolean('RegularizeMultisliceUnwrapPhase')
        self.regularize_multislice_unwrap_phase_image_gradient_method = group.string(
            'RegularizeMultisliceUnwrapPhaseImageGradientMethod'
        )
        self.regularize_multislice_unwrap_phase_image_integration_method = group.string(
            'RegularizeMultisliceUnwrapPhaseImageIntegrationMethod'
        )
        self.patch_interpolation_method = group.string('PatchInterpolationMethod')
        self.remove_object_probe_ambiguity = group.boolean('RemoveObjectProbeAmbiguity')
        self.remove_object_probe_ambiguity_start = group.integer('RemoveObjectProbeAmbiguityStart')
        self.remove_object_probe_ambiguity_stop = group.integer('RemoveObjectProbeAmbiguityStop')
        self.remove_object_probe_ambiguity_stride = group.integer(
            'RemoveObjectProbeAmbiguityStride'
        )
        self.build_preconditioner_with_all_modes = group.boolean('BuildPreconditionerWithAllModes')
        self.constrain_hard_limits = group.boolean('ConstrainHardLimits')
        self.constrain_hard_limits_start = group.integer('ConstrainHardLimitsStart')
        self.constrain_hard_limits_stop = group.integer('ConstrainHardLimitsStop')
        self.constrain_hard_limits_stride = group.integer('ConstrainHardLimitsStride')
        self.constrain_hard_limits_enable_abs = group.boolean('ConstrainHardLimitsEnableAbs')
        self.constrain_hard_limits_abs_min = group.real('ConstrainHardLimitsAbsMin')
        self.constrain_hard_limits_abs_max = group.real('ConstrainHardLimitsAbsMax')
        self.constrain_hard_limits_enable_phase = group.boolean('ConstrainHardLimitsEnablePhase')
        self.constrain_hard_limits_phase_min_deg = group.real('ConstrainHardLimitsPhaseMinDeg')
        self.constrain_hard_limits_phase_max_deg = group.real('ConstrainHardLimitsPhaseMaxDeg')


class PtyChiProbeSettings:
    """The ``PtyChiProbe`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiProbe')
        self.is_optimizable = group.boolean('IsOptimizable')
        self.optimization_plan_start = group.integer('OptimizationPlanStart')
        self.optimization_plan_stop = group.integer('OptimizationPlanStop')
        self.optimization_plan_stride = group.integer('OptimizationPlanStride')
        self.optimizer = group.string('Optimizer')
        self.step_size = group.real('StepSize')
        self.constrain_probe_power = group.boolean('ConstrainProbePower')
        self.constrain_probe_power_start = group.integer('ConstrainProbePowerStart')
        self.constrain_probe_power_stop = group.integer('ConstrainProbePowerStop')
        self.constrain_probe_power_stride = group.integer('ConstrainProbePowerStride')
        self.constrain_probe_power_scale_object = group.boolean('ConstrainProbePowerScaleObject')
        self.orthogonalize_incoherent_modes = group.boolean('OrthogonalizeIncoherentModes')
        self.orthogonalize_incoherent_modes_start = group.integer(
            'OrthogonalizeIncoherentModesStart'
        )
        self.orthogonalize_incoherent_modes_stop = group.integer('OrthogonalizeIncoherentModesStop')
        self.orthogonalize_incoherent_modes_stride = group.integer(
            'OrthogonalizeIncoherentModesStride'
        )
        self.orthogonalize_incoherent_modes_method = group.string(
            'OrthogonalizeIncoherentModesMethod'
        )
        self.orthogonalize_incoherent_modes_sort_by_occupancy = group.boolean(
            'OrthogonalizeIncoherentModesSortByOccupancy'
        )
        self.orthogonalize_opr_modes = group.boolean('OrthogonalizeOPRModes')
        self.orthogonalize_opr_modes_start = group.integer('OrthogonalizeOPRModesStart')
        self.orthogonalize_opr_modes_stop = group.integer('OrthogonalizeOPRModesStop')
        self.orthogonalize_opr_modes_stride = group.integer('OrthogonalizeOPRModesStride')
        self.constrain_support = group.boolean('ConstrainSupport')
        self.constrain_support_start = group.integer('ConstrainSupportStart')
        self.constrain_support_stop = group.integer('ConstrainSupportStop')
        self.constrain_support_stride = group.integer('ConstrainSupportStride')
        self.constrain_support_threshold = group.real('ConstrainSupportThreshold')
        self.constrain_support_method = group.string('ConstrainSupportMethod')
        self.constrain_center = group.boolean('ConstrainCenter')
        self.constrain_center_start = group.integer('ConstrainCenterStart')
        self.constrain_center_stop = group.integer('ConstrainCenterStop')
        self.constrain_center_stride = group.integer('ConstrainCenterStride')
        self.use_total_intensity_for_com = group.boolean('UseTotalIntensityForCOM')
        self.constrain_center_modes_individually = group.boolean('ConstrainCenterModesIndividually')
        self.eigenmode_update_relaxation = group.real('EigenmodeUpdateRelaxation')


class PtyChiProbePositionSettings:
    """The ``PtyChiProbePosition`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiProbePosition')
        self.is_optimizable = group.boolean('IsOptimizable')
        self.optimization_plan_start = group.integer('OptimizationPlanStart')
        self.optimization_plan_stop = group.integer('OptimizationPlanStop')
        self.optimization_plan_stride = group.integer('OptimizationPlanStride')
        self.optimizer = group.string('Optimizer')
        self.step_size = group.real('StepSize')
        self.constrain_position_mean = group.boolean('ConstrainPositionMean')
        self.correction_type = group.string('CorrectionType')
        self.differentiation_method = group.string('DifferentiationMethod')
        self.cross_correlation_scale = group.integer('CrossCorrelationScale')
        self.cross_correlation_real_space_width = group.real('CrossCorrelationRealSpaceWidth')
        self.cross_correlation_probe_threshold = group.real('CrossCorrelationProbeThreshold')
        self.choose_slice_for_correction = group.boolean('ChooseSliceForCorrection')
        self.slice_for_correction = group.integer('SliceForCorrection')
        self.clip_update_magnitude_by_mad = group.boolean('ClipUpdateMagnitudeByMAD')
        self.limit_update_magnitude = group.boolean('LimitUpdateMagnitude')
        self.update_magnitude_limit = group.real('UpdateMagnitudeLimit')
        self.constrain_affine_transform = group.boolean('ConstrainAffineTransform')
        self.constrain_affine_transform_start = group.integer('ConstrainAffineTransformStart')
        self.constrain_affine_transform_stop = group.integer('ConstrainAffineTransformStop')
        self.constrain_affine_transform_stride = group.integer('ConstrainAffineTransformStride')
        self.constrain_affine_transform_degrees_of_freedom = group.integer(
            'ConstrainAffineTransformDegreesOfFreedom'
        )
        self.constrain_affine_transform_position_weight_update_interval = group.integer(
            'ConstrainAffineTransformPositionWeightUpdateInterval'
        )
        self.constrain_affine_transform_apply_constraint = group.boolean(
            'ConstrainAffineTransformApplyConstraint'
        )
        self.constrain_affine_transform_max_expected_error_px = group.real(
            'ConstrainAffineTransformMaxExpectedErrorInPixels'
        )
        self.override_affine_transform_update_flexibility = group.boolean(
            'OverrideAffineTransformUpdateFlexibility'
        )
        self.constrain_affine_transform_override_update_flexibility = group.real(
            'ConstrainAffineTransformOverrideUpdateFlexibility'
        )


class PtyChiOPRSettings:
    """The ``PtyChiOPR`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiOPR')
        self.is_optimizable = group.boolean('IsOptimizable')
        self.optimization_plan_start = group.integer('OptimizationPlanStart')
        self.optimization_plan_stop = group.integer('OptimizationPlanStop')
        self.optimization_plan_stride = group.integer('OptimizationPlanStride')
        self.optimizer = group.string('Optimizer')
        self.step_size = group.real('StepSize')
        self.optimize_eigenmode_weights = group.boolean('OptimizeEigenmodeWeights')
        self.optimize_intensity_variation = group.boolean('OptimizeIntensityVariation')
        self.enable_primary_mode_weight_floor = group.boolean('EnablePrimaryModeWeightFloor')
        self.primary_mode_weight_floor = group.real('PrimaryModeWeightFloor')
        self.smooth_mode_weights = group.boolean('SmoothModeWeights')
        self.smooth_mode_weights_start = group.integer('SmoothModeWeightsStart')
        self.smooth_mode_weights_stop = group.integer('SmoothModeWeightsStop')
        self.smooth_mode_weights_stride = group.integer('SmoothModeWeightsStride')
        self.smoothing_method = group.string('SmoothingMethod')
        self.polynomial_smoothing_degree = group.integer('PolynomialSmoothingDegree')
        self.update_relaxation = group.real('UpdateRelaxation')


class PtyChiAutodiffSettings:
    """The ``PtyChiAutodiff`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiAutodiff')
        self.loss_function = group.string('LossFunction')
        self.forward_model_class = group.string('ForwardModelClass')


class PtyChiDMSettings:
    """The ``PtyChiDM`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiDM')
        self.exit_wave_update_relaxation = group.real('ExitWaveUpdateRelaxation')
        self.chunk_length = group.integer('ChunkLength')
        self.object_amplitude_clamp_limit = group.real('ObjectAmplitudeClampLimit')
        self.object_inertia = group.real('ObjectInertia')
        self.probe_inertia = group.real('ProbeInertia')


class PtyChiRAARSettings:
    """The ``PtyChiRAAR`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiRAAR')
        self.beta = group.real('Beta')
        self.chunk_length = group.integer('ChunkLength')
        self.object_amplitude_clamp_limit = group.real('ObjectAmplitudeClampLimit')
        self.object_inertia = group.real('ObjectInertia')
        self.probe_inertia = group.real('ProbeInertia')


class PtyChiLSQMLSettings:
    """The ``PtyChiLSQML`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiLSQML')
        self.noise_model = group.string('NoiseModel')
        self.gaussian_noise_std = group.real('GaussianNoiseStd')
        self.single_slice_solve_object_probe_step_size_jointly = group.boolean(
            'SingleSliceSolveObjectProbeStepSizeJointly'
        )
        self.multislice_solve_object_probe_step_size_jointly = group.boolean(
            'MultisliceSolveObjectProbeStepSizeJointly'
        )
        self.solve_step_sizes_only_using_first_probe_mode = group.boolean(
            'SolveStepSizesOnlyUsingFirstProbeMode'
        )
        self.momentum_acceleration_gain = group.real('MomentumAccelerationGain')
        self.auto_momentum_acceleration_gradient_mixing_factor = group.boolean(
            'AutoMomentumAccelerationGradientMixingFactor'
        )
        self.momentum_acceleration_gradient_mixing_factor = group.real(
            'MomentumAccelerationGradientMixingFactor'
        )
        self.rescale_probe_intensity_in_first_epoch = group.boolean(
            'RescaleProbeIntensityInFirstEpoch'
        )
        self.preconditioning_damping_factor = group.real('PreconditioningDampingFactor')
        self.object_optimal_step_size_scaler = group.real('ObjectOptimalStepSizeScaler')
        self.object_multimodal_update = group.boolean('ObjectMultimodalUpdate')
        self.probe_optimal_step_size_scaler = group.real('ProbeOptimalStepSizeScaler')
        self.probe_position_momentum_acceleration_gain = group.real(
            'ProbePositionMomentumAccelerationGain'
        )
        self.probe_position_auto_momentum_gradient_mixing_factor = group.boolean(
            'ProbePositionAutoMomentumGradientMixingFactor'
        )
        self.probe_position_momentum_acceleration_gradient_mixing_factor = group.real(
            'ProbePositionMomentumAccelerationGradientMixingFactor'
        )
        self.probe_position_momentum_acceleration_memory = group.integer(
            'ProbePositionMomentumAccelerationMemory'
        )


class PtyChiBHSettings:
    """The ``PtyChiBH`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiBH')
        self.method = group.string('Method')
        self.probe_rho = group.real('ProbeRho')
        self.probe_position_rho = group.real('ProbePositionRho')


class PtyChiPIESettings:
    """The ``PtyChiPIE`` settings group, under the names the panels use."""

    def __init__(self, source: PtyChiSettingsSource) -> None:
        group = source.group('PtyChiPIE')
        self.probe_alpha = group.real('ProbeAlpha')
        self.object_alpha = group.real('ObjectAlpha')


# What a panel is handed for the algorithm it is showing. Each algorithm has
# exactly one, and the panel code narrows on which.
PtyChiAlgorithmSettings = (
    PtyChiAutodiffSettings
    | PtyChiBHSettings
    | PtyChiDMSettings
    | PtyChiLSQMLSettings
    | PtyChiPIESettings
    | PtyChiRAARSettings
)
