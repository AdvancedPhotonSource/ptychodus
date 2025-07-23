from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtyChiSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChi')
        self._group.add_observer(self)

        # ReconstructorOptions
        self.num_epochs = self._group.create_integer_parameter('NumEpochs', 100, minimum=1)
        self.batch_size = self._group.create_integer_parameter('BatchSize', 100, minimum=1)
        self.batching_mode = self._group.create_string_parameter('BatchingMode', 'random')
        self.compact_mode_update_clustering = self._group.create_integer_parameter(
            'CompactModeUpdateClustering', 1, minimum=0
        )
        self.use_devices = self._group.create_boolean_parameter('UseDevices', True)
        self.use_double_precision = self._group.create_boolean_parameter(
            'UseDoublePrecision', False
        )
        self.use_double_precision_for_fft = self._group.create_boolean_parameter(
            'UseDoublePrecisionForFFT', False
        )
        self.allow_nondeterministic_algorithms = self._group.create_boolean_parameter(
            'AllowNondeterministicAlgorithms', True
        )

        # ForwardModelOptions
        self.use_low_memory_mode = self._group.create_boolean_parameter('UseLowMemoryMode', False)
        self.pad_for_shift = self._group.create_integer_parameter('PadForShift', 0, minimum=0)

        # PtychographyDataOptions
        self.use_far_field_propagation = self._group.create_boolean_parameter(
            'UseFarFieldPropagation', True
        )
        self.fft_shift_diffraction_patterns = self._group.create_boolean_parameter(
            'FFTShiftDiffractionPatterns', True
        )
        self.save_data_on_device = self._group.create_boolean_parameter('SaveDataOnDevice', False)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiObject')
        self._group.add_observer(self)

        self.is_optimizable = self._group.create_boolean_parameter('IsOptimizable', True)
        self.optimization_plan_start = self._group.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimization_plan_stop = self._group.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimization_plan_stride = self._group.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._group.create_string_parameter('Optimizer', 'SGD')
        self.step_size = self._group.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.optimize_slice_spacing = self._group.create_boolean_parameter(
            'OptimizeSliceSpacing', False
        )
        self.optimize_slice_spacing_start = self._group.create_integer_parameter(
            'OptimizeSliceSpacingStart', 0, minimum=0
        )
        self.optimize_slice_spacing_stop = self._group.create_integer_parameter(
            'OptimizeSliceSpacingStop', -1
        )
        self.optimize_slice_spacing_stride = self._group.create_integer_parameter(
            'OptimizeSliceSpacingStride', 1, minimum=1
        )
        self.optimize_slice_spacing_optimizer = self._group.create_string_parameter(
            'OptimizeSliceSpacingOptimizer', 'SGD'
        )
        self.optimize_slice_spacing_step_size = self._group.create_real_parameter(
            'OptimizeSliceSpacingStepSize', 1.0e-10, minimum=0.0
        )

        self.constrain_l1_norm = self._group.create_boolean_parameter('ConstrainL1Norm', False)
        self.constrain_l1_norm_start = self._group.create_integer_parameter(
            'ConstrainL1NormStart', 0, minimum=0
        )
        self.constrain_l1_norm_stop = self._group.create_integer_parameter(
            'ConstrainL1NormStop', -1
        )
        self.constrain_l1_norm_stride = self._group.create_integer_parameter(
            'ConstrainL1NormStride', 1, minimum=1
        )
        self.constrain_l1_norm_weight = self._group.create_real_parameter(
            'ConstrainL1NormWeight', 0.0, minimum=0.0
        )

        self.constrain_l2_norm = self._group.create_boolean_parameter('ConstrainL2Norm', False)
        self.constrain_l2_norm_start = self._group.create_integer_parameter(
            'ConstrainL2NormStart', 0, minimum=0
        )
        self.constrain_l2_norm_stop = self._group.create_integer_parameter(
            'ConstrainL2NormStop', -1
        )
        self.constrain_l2_norm_stride = self._group.create_integer_parameter(
            'ConstrainL2NormStride', 1, minimum=1
        )
        self.constrain_l2_norm_weight = self._group.create_real_parameter(
            'ConstrainL2NormWeight', 0.0, minimum=0.0
        )

        self.constrain_smoothness = self._group.create_boolean_parameter(
            'ConstrainSmoothness', False
        )
        self.constrain_smoothness_start = self._group.create_integer_parameter(
            'ConstrainSmoothnessStart', 0, minimum=0
        )
        self.constrain_smoothness_stop = self._group.create_integer_parameter(
            'ConstrainSmoothnessStop', -1
        )
        self.constrain_smoothness_stride = self._group.create_integer_parameter(
            'ConstrainSmoothnessStride', 1, minimum=1
        )
        self.constrain_smoothness_alpha = self._group.create_real_parameter(
            'ConstrainSmoothnessAlpha', 0.0, minimum=0.0, maximum=1.0 / 8
        )

        self.constrain_total_variation = self._group.create_boolean_parameter(
            'ConstrainTotalVariation', False
        )
        self.constrain_total_variation_start = self._group.create_integer_parameter(
            'ConstrainTotalVariationStart', 0, minimum=0
        )
        self.constrain_total_variation_stop = self._group.create_integer_parameter(
            'ConstrainTotalVariationStop', -1
        )
        self.constrain_total_variation_stride = self._group.create_integer_parameter(
            'ConstrainTotalVariationStride', 1, minimum=1
        )
        self.constrain_total_variation_weight = self._group.create_real_parameter(
            'ConstrainTotalVariationWeight', 0.0, minimum=0.0
        )

        self.remove_grid_artifacts = self._group.create_boolean_parameter(
            'RemoveGridArtifacts', False
        )
        self.remove_grid_artifacts_start = self._group.create_integer_parameter(
            'RemoveGridArtifactsStart', 0, minimum=0
        )
        self.remove_grid_artifacts_stop = self._group.create_integer_parameter(
            'RemoveGridArtifactsStop', -1
        )
        self.remove_grid_artifacts_stride = self._group.create_integer_parameter(
            'RemoveGridArtifactsStride', 1, minimum=1
        )
        self.remove_grid_artifacts_period_x_m = self._group.create_real_parameter(
            'RemoveGridArtifactsPeriodXInMeters', 1e-7, minimum=0.0
        )
        self.remove_grid_artifacts_period_y_m = self._group.create_real_parameter(
            'RemoveGridArtifactsPeriodYInMeters', 1e-7, minimum=0.0
        )
        self.remove_grid_artifacts_window_size_px = self._group.create_integer_parameter(
            'RemoveGridArtifactsWindowSizeInPixels',
            5,
            minimum=1,
        )
        self.remove_grid_artifacts_direction = self._group.create_string_parameter(
            'RemoveGridArtifactsDirection', 'XY'
        )

        self.regularize_multislice = self._group.create_boolean_parameter(
            'RegularizeMultislice', False
        )
        self.regularize_multislice_start = self._group.create_integer_parameter(
            'RegularizeMultisliceStart', 0, minimum=0
        )
        self.regularize_multislice_stop = self._group.create_integer_parameter(
            'RegularizeMultisliceStop', -1
        )
        self.regularize_multislice_stride = self._group.create_integer_parameter(
            'RegularizeMultisliceStride', 1, minimum=1
        )
        self.regularize_multislice_weight = self._group.create_real_parameter(
            'RegularizeMultisliceWeight', 0.0, minimum=0.0
        )
        self.regularize_multislice_unwrap_phase = self._group.create_boolean_parameter(
            'RegularizeMultisliceUnwrapPhase', True
        )
        self.regularize_multislice_unwrap_phase_image_gradient_method = (
            self._group.create_string_parameter(
                'RegularizeMultisliceUnwrapPhaseImageGradientMethod', 'FOURIER_DIFFERENTIATION'
            )
        )
        self.regularize_multislice_unwrap_phase_image_integration_method = (
            self._group.create_string_parameter(
                'RegularizeMultisliceUnwrapPhaseImageIntegrationMethod', 'FOURIER'
            )
        )

        self.patch_interpolator = self._group.create_string_parameter(
            'PatchInterpolator', 'FOURIER'
        )

        self.remove_object_probe_ambiguity = self._group.create_boolean_parameter(
            'RemoveObjectProbeAmbiguity', True
        )
        self.remove_object_probe_ambiguity_start = self._group.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStart', 0, minimum=0
        )
        self.remove_object_probe_ambiguity_stop = self._group.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStop', -1
        )
        self.remove_object_probe_ambiguity_stride = self._group.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStride', 10, minimum=1
        )

        self.build_preconditioner_with_all_modes = self._group.create_boolean_parameter(
            'BuildPreconditionerWithAllModes', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiProbe')
        self._group.add_observer(self)

        self.is_optimizable = self._group.create_boolean_parameter('IsOptimizable', True)
        self.optimization_plan_start = self._group.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimization_plan_stop = self._group.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimization_plan_stride = self._group.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._group.create_string_parameter('Optimizer', 'SGD')
        self.step_size = self._group.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.constrain_probe_power = self._group.create_boolean_parameter(
            'ConstrainProbePower', False
        )
        self.constrain_probe_power_start = self._group.create_integer_parameter(
            'ConstrainProbePowerStart', 0, minimum=0
        )
        self.constrain_probe_power_stop = self._group.create_integer_parameter(
            'ConstrainProbePowerStop', -1
        )
        self.constrain_probe_power_stride = self._group.create_integer_parameter(
            'ConstrainProbePowerStride', 1, minimum=1
        )

        self.orthogonalize_incoherent_modes = self._group.create_boolean_parameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.orthogonalize_incoherent_modes_start = self._group.create_integer_parameter(
            'OrthogonalizeIncoherentModesStart', 0, minimum=0
        )
        self.orthogonalize_incoherent_modes_stop = self._group.create_integer_parameter(
            'OrthogonalizeIncoherentModesStop', -1
        )
        self.orthogonalize_incoherent_modes_stride = self._group.create_integer_parameter(
            'OrthogonalizeIncoherentModesStride', 1, minimum=1
        )
        self.orthogonalize_incoherent_modes_method = self._group.create_string_parameter(
            'OrthogonalizeIncoherentModesMethod', 'SVD'
        )

        self.orthogonalize_opr_modes = self._group.create_boolean_parameter(
            'OrthogonalizeOPRModes', True
        )
        self.orthogonalize_opr_modes_start = self._group.create_integer_parameter(
            'OrthogonalizeOPRModesStart', 0, minimum=0
        )
        self.orthogonalize_opr_modes_stop = self._group.create_integer_parameter(
            'OrthogonalizeOPRModesStop', -1
        )
        self.orthogonalize_opr_modes_stride = self._group.create_integer_parameter(
            'OrthogonalizeOPRModesStride', 1, minimum=1
        )

        self.constrain_support = self._group.create_boolean_parameter('ConstrainSupport', False)
        self.constrain_support_start = self._group.create_integer_parameter(
            'ConstrainSupportStart', 0, minimum=0
        )
        self.constrain_support_stop = self._group.create_integer_parameter(
            'ConstrainSupportStop', -1
        )
        self.constrain_support_stride = self._group.create_integer_parameter(
            'ConstrainSupportStride', 1, minimum=1
        )
        self.constrain_support_threshold = self._group.create_real_parameter(
            'ConstrainSupportThreshold', 0.005, minimum=0.0
        )

        self.constrain_center = self._group.create_boolean_parameter('ConstrainCenter', False)
        self.constrain_center_start = self._group.create_integer_parameter(
            'ConstrainCenterStart', 0, minimum=0
        )
        self.constrain_center_stop = self._group.create_integer_parameter('ConstrainCenterStop', -1)
        self.constrain_center_stride = self._group.create_integer_parameter(
            'ConstrainCenterStride', 1, minimum=1
        )
        self.use_intensity_for_mass_centroid = self._group.create_boolean_parameter(
            'UseIntensityForMassCentroid', False
        )

        self.relax_eigenmode_update = self._group.create_real_parameter(
            'RelaxEigenmodeUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiProbePositionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiProbePosition')
        self._group.add_observer(self)

        self.is_optimizable = self._group.create_boolean_parameter('IsOptimizable', False)
        self.optimization_plan_start = self._group.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimization_plan_stop = self._group.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimization_plan_stride = self._group.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._group.create_string_parameter('Optimizer', 'SGD')
        self.step_size = self._group.create_real_parameter('StepSize', 0.3, minimum=0.0)

        self.constrain_centroid = self._group.create_boolean_parameter('ConstrainCentroid', False)

        # correction_options
        self.correction_type = self._group.create_string_parameter('CorrectionType', 'GRADIENT')
        self.differentiation_method = self._group.create_string_parameter(
            'DifferentiationMethod', 'FOURIER_DIFFERENTIATION'
        )
        self.cross_correlation_scale = self._group.create_integer_parameter(
            'CrossCorrelationScale', 20000, minimum=1
        )
        self.cross_correlation_real_space_width = self._group.create_real_parameter(
            'CrossCorrelationRealSpaceWidth', 0.01, minimum=0.0
        )
        self.cross_correlation_probe_threshold = self._group.create_real_parameter(
            'CrossCorrelationProbeThreshold', 0.1, minimum=0.0, maximum=1.0
        )
        self.choose_slice_for_correction = self._group.create_boolean_parameter(
            'ChooseSliceForCorrection', False
        )
        self.slice_for_correction = self._group.create_integer_parameter(
            'SliceForCorrection', 0, minimum=0
        )
        self.clip_update_magnitude_by_mad = self._group.create_boolean_parameter(
            'ClipUpdateMagnitudeByMAD', True
        )
        self.limit_update_magnitude = self._group.create_boolean_parameter(
            'LimitUpdateMagnitude', False
        )
        self.update_magnitude_limit = self._group.create_real_parameter(
            'UpdateMagnitudeLimit', 0.1, minimum=0.0
        )

        # affine_transform_constraint
        self.constrain_affine_transform = self._group.create_boolean_parameter(
            'ConstrainAffineTransform', False
        )
        self.constrain_affine_transform_start = self._group.create_integer_parameter(
            'ConstrainAffineTransformStart', 0, minimum=0
        )
        self.constrain_affine_transform_stop = self._group.create_integer_parameter(
            'ConstrainAffineTransformStop', -1
        )
        self.constrain_affine_transform_stride = self._group.create_integer_parameter(
            'ConstrainAffineTransformStride', 1, minimum=1
        )
        self.constrain_affine_transform_degrees_of_freedom = self._group.create_integer_parameter(
            'ConstrainAffineTransformDegreesOfFreedom', 0, minimum=0
        )
        self.constrain_affine_transform_position_weight_update_interval = (
            self._group.create_integer_parameter(
                'ConstrainAffineTransformPositionWeightUpdateInterval', 10, minimum=1
            )
        )
        self.constrain_affine_transform_apply_constraint = self._group.create_boolean_parameter(
            'ConstrainAffineTransformApplyConstraint', True
        )
        self.constrain_affine_transform_max_expected_error_px = self._group.create_real_parameter(
            'ConstrainAffineTransformMaxExpectedErrorInPixels', 1.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiOPRSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiOPR')
        self._group.add_observer(self)

        self.is_optimizable = self._group.create_boolean_parameter('IsOptimizable', False)
        self.optimization_plan_start = self._group.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimization_plan_stop = self._group.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimization_plan_stride = self._group.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._group.create_string_parameter('Optimizer', 'SGD')
        self.step_size = self._group.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.optimize_eigenmode_weights = self._group.create_boolean_parameter(
            'OptimizeEigenmodeWeigts', True
        )
        self.optimize_intensities = self._group.create_boolean_parameter(
            'OptimizeIntensities', False
        )

        self.smooth_mode_weights = self._group.create_boolean_parameter('SmoothModeWeights', False)
        self.smooth_mode_weights_start = self._group.create_integer_parameter(
            'SmoothModeWeightsStart', 0, minimum=0
        )
        self.smooth_mode_weights_stop = self._group.create_integer_parameter(
            'SmoothModeWeightsStop', -1
        )
        self.smooth_mode_weights_stride = self._group.create_integer_parameter(
            'SmoothModeWeightsStride', 1, minimum=1
        )
        self.smoothing_method = self._group.create_string_parameter('SmoothingMethod', '')
        self.polynomial_smoothing_degree = self._group.create_integer_parameter(
            'PolynomialSmoothingDegree', 4, minimum=0, maximum=10
        )

        self.relax_update = self._group.create_real_parameter(
            'RelaxUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiAutodiffSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiAutodiff')
        self._group.add_observer(self)

        self.loss_function = self._group.create_string_parameter('LossFunction', 'MSE_SQRT')
        self.forward_model_class = self._group.create_string_parameter(
            'ForwardModelClass', 'PLANAR_PTYCHOGRAPHY'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiDMSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiDM')
        self._group.add_observer(self)

        self.exit_wave_update_relaxation = self._group.create_real_parameter(
            'ExitWaveUpdateRelaxation', 1.0, minimum=0.0, maximum=1.0
        )
        self.chunk_length = self._group.create_integer_parameter('ChunkLength', 1, minimum=1)
        self.object_amplitude_clamp_limit = self._group.create_real_parameter(
            'ObjectAmplitudeClampLimit', 1000, minimum=0.0
        )
        self.object_inertia = self._group.create_real_parameter(
            'ObjectInertia', 0.0, minimum=0.0, maximum=1.0
        )
        self.probe_inertia = self._group.create_real_parameter(
            'ProbeInertia', 0.0, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiLSQMLSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiLSQML')
        self._group.add_observer(self)

        self.noise_model = self._group.create_string_parameter('NoiseModel', 'GAUSSIAN')
        self.gaussian_noise_deviation = self._group.create_real_parameter(
            'GaussianNoiseDeviation', 0.5
        )
        self.solve_object_probe_step_size_jointly_for_first_slice_in_multislice = (
            self._group.create_boolean_parameter(
                'SolveObjectProbeStepSizeJointlyForFirstSliceInMultislice', False
            )
        )
        self.solve_step_sizes_only_using_first_probe_mode = self._group.create_boolean_parameter(
            'SolveStepSizesOnlyUsingFirstProbeMode', True
        )
        self.momentum_acceleration_gain = self._group.create_real_parameter(
            'MomentumAccelerationGain', 0.0, minimum=0.0
        )
        self.use_momentum_acceleration_gradient_mixing_factor = (
            self._group.create_boolean_parameter(
                'UseMomentumAccelerationGradientMixingFactor', False
            )
        )
        self.momentum_acceleration_gradient_mixing_factor = self._group.create_real_parameter(
            'MomentumAccelerationGradientMixingFactor', 1.0
        )
        self.rescale_probe_intensity_in_first_epoch = self._group.create_boolean_parameter(
            'RescaleProbeIntensityInFirstEpoch', True
        )
        self.preconditioning_damping_factor = self._group.create_real_parameter(
            'PreconditioningDampingFactor', 0.1, minimum=0.0
        )

        self.object_optimal_step_size_scaler = self._group.create_real_parameter(
            'ObjectOptimalStepSizeScaler', 0.9, minimum=0.0
        )
        self.object_multimodal_update = self._group.create_boolean_parameter(
            'ObjectMultimodalUpdate', True
        )
        self.probe_optimal_step_size_scaler = self._group.create_real_parameter(
            'ProbeOptimalStepSizeScaler', 0.9, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtyChiPIESettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtyChiPIE')
        self._group.add_observer(self)

        self.probe_alpha = self._group.create_real_parameter(
            'ProbeAlpha', 0.1, minimum=0.0, maximum=1.0
        )
        self.object_alpha = self._group.create_real_parameter(
            'ObjectAlpha', 0.1, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
