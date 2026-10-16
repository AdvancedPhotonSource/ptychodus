"""Where each pty-chi option is persisted in ptychodus's flat settings layout.

pty-chi declares its options as a nested tree; ptychodus persists eleven flat
sections. This is the whole of the difference: which branch feeds which section,
what each key is called, which fields are not surfaced, and the handful that
change shape on the way out.

It lives here rather than beside the backend because a plugin never names an INI
key, and **it is transitional**: when the Qt panels retire and the nested
declaration goes straight to the web UI, this file goes with them.

A hidden field is hidden for one of two reasons, kept apart below because they
read alike and mean opposite things: ptychodus either fills it from the product
at reconstruct time, or does not offer it at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .flatten import (
    BoolEnum,
    Label,
    fields_of,
    FlatteningPolicy,
    OptionalGate,
    OptionalRange,
    Sentinel,
    StructuralAdapter,
)
from .parameter_store import OptionsSection

__all__ = ['sections_for']

# Every optimizable branch carries a plan whose three visible fields sit directly
# under that branch's prefix; its scheduler fields are not surfaced.
_PLAN = 'optimization_plan'
_PLAN_HIDDEN = ('step_size_scheduler_class', 'step_size_scheduler_options')


def _plans(*branches: str) -> dict[str, str]:
    """Make each branch's own plan transparent, so its keys take the branch prefix."""
    return {f'{b}.{_PLAN}': '' for b in branches}


def _plan_noise(*branches: str) -> set[str]:
    paths = {_PLAN, *(f'{b}.{_PLAN}' for b in branches)}
    return {f'{p}.{f}' for p in paths for f in _PLAN_HIDDEN}


def _optimizer_noise(*branches: str) -> set[str]:
    return {'optimizer_params', *(f'{b}.optimizer_params' for b in branches)}


def _extras(instance: Any, base: type) -> set[str]:
    """Fields an algorithm's subclass adds, which ptychodus files under that algorithm."""
    return set(fields_of(type(instance))) - set(fields_of(base))


def _stops(*branches: str) -> dict[str, StructuralAdapter]:
    """Every plan's ``stop`` is persisted as -1 when the plan runs to the end."""
    paths = {_PLAN, *(f'{b}.{_PLAN}' for b in branches)}
    return {f'{p}.stop': Sentinel('Stop', value_type=int, sentinel=-1) for p in paths}


_OBJECT_BRANCHES = (
    'slice_spacing_options',
    'l1_norm_constraint',
    'l2_norm_constraint',
    'smoothness_constraint',
    'total_variation',
    'hard_limits_magnitude_phase',
    'remove_grid_artifacts',
    'multislice_regularization',
    'remove_object_probe_ambiguity',
)

_PROBE_BRANCHES = (
    'power_constraint',
    'orthogonalize_incoherent_modes',
    'orthogonalize_opr_modes',
    'support_constraint',
    'center_constraint',
)

_POSITION_BRANCHES = ('correction_options', 'affine_transform_constraint')
_OPR_BRANCHES = ('smoothing',)


_OBJECT_LABELS: Mapping[str, Label] = {
    'optimizable': Label('Optimize Object', 'Whether the object is optimizable'),
    'optimizer': Label('Optimizer:'),
    'step_size': Label('Step Size:', 'Optimizer step size (learning rate).'),
    'patch_interpolation_method': Label(
        'Patch Interpolator:',
        'Interpolation method used for extracting and updating patches of the object',
    ),
    'build_preconditioner_with_all_modes': Label(
        'Build Preconditioner with All Modes',
        'Build the illumination preconditioner from the summed intensity of all probe modes.',
    ),
    'slice_spacing_options.optimizable': Label(
        'Optimize Slice Spacing', 'Whether to optimize the slice spacing'
    ),
    'slice_spacing_options.optimizer': Label('Optimizer:'),
    'slice_spacing_options.step_size': Label('Step Size:', 'Optimizer step size (learning rate).'),
    'l1_norm_constraint.enabled': Label(
        'Constrain L\u2081 Norm', 'Whether to constrain the L\u2081 norm'
    ),
    'l1_norm_constraint.weight': Label(
        'Weight:', 'Weight of the L\u2081 norm constraint (0 or less disables it).'
    ),
    'l2_norm_constraint.enabled': Label(
        'Constrain L\u2082 Norm', 'Whether to constrain the L\u2082 norm'
    ),
    'l2_norm_constraint.weight': Label(
        'Weight:', 'Weight of the L\u2082 norm constraint (0 or less disables it).'
    ),
    'smoothness_constraint.enabled': Label(
        'Constrain Smoothness',
        'Whether to constrain smoothness in the magnitude (but not phase) of the object',
    ),
    'smoothness_constraint.alpha': Label(
        'Alpha:',
        'Relaxation smoothing constant; must be in (0, 1/8], with maximal smoothing at 1/8.',
    ),
    'total_variation.enabled': Label(
        'Constrain Total Variation', 'Whether to constrain the total variation'
    ),
    'total_variation.weight': Label(
        'Weight:', 'Weight of the total variation constraint (0 or less disables it).'
    ),
    'remove_grid_artifacts.enabled': Label(
        'Remove Grid Artifacts',
        "Whether to remove grid artifacts in the object's phase at the end of an epoch.",
    ),
    'remove_grid_artifacts.period_x_m': Label(
        'Period X:', 'Horizontal period of grid artifacts in meters'
    ),
    'remove_grid_artifacts.period_y_m': Label(
        'Period Y:', 'Vertical period of grid artifacts in meters'
    ),
    'remove_grid_artifacts.window_size': Label(
        'Window Size [px]:', 'Window size for grid artifact removal in pixels'
    ),
    'remove_grid_artifacts.direction': Label('Direction:', 'Direction of grid artifact removal'),
    'remove_grid_artifacts.component': Label(
        'Component:', 'Component of the object to apply grid artifact removal to'
    ),
    'multislice_regularization.enabled': Label(
        'Regularize Multislice',
        'Whether to regularize multislice objects using cross-slice smoothing',
    ),
    'multislice_regularization.weight': Label(
        'Weight:',
        'Weight of the cross-slice smoothing regularization for multislice objects '
        '(0 disables it).',
    ),
    'multislice_regularization.unwrap_phase': Label(
        'Unwrap Phase',
        'Whether to unwrap the phase of the object during multislice regularization',
    ),
    'multislice_regularization.unwrap_image_grad_method': Label(
        'Gradient Method:', 'Method for calculating the phase gradient during phase unwrapping'
    ),
    'multislice_regularization.unwrap_image_integration_method': Label(
        'Integration Method:',
        'Method for integrating the phase gradient during phase unwrapping',
    ),
    'remove_object_probe_ambiguity.enabled': Label(
        'Remove Object Probe Ambiguity', 'Whether to remove object-probe ambiguity'
    ),
    'hard_limits_magnitude_phase.enabled': Label(
        'Constrain Hard Limits',
        'Whether to apply hard magnitude/phase limits to the object',
    ),
    'EnableAbs': Label('Limit Magnitude', 'Whether to apply magnitude limits'),
    'AbsMin': Label('Magnitude Min:', 'Minimum magnitude value'),
    'AbsMax': Label('Magnitude Max:', 'Maximum magnitude value'),
    'EnablePhase': Label('Limit Phase', 'Whether to apply phase limits'),
    'PhaseMinDeg': Label('Phase Min [deg]:', 'Minimum phase value in degrees'),
    'PhaseMaxDeg': Label('Phase Max [deg]:', 'Maximum phase value in degrees'),
}


def _object_policy(extras: set[str]) -> FlatteningPolicy:
    return FlatteningPolicy(
        hidden=frozenset(
            # Filled from the product at reconstruct time.
            {
                'initial_guess',
                'slice_spacings_m',
                'pixel_size_m',
                'pixel_size_aspect_ratio',
                'position_origin_coords',
            }
            # Not surfaced.
            | {'determine_position_origin_coords_by', 'power_constraint.probe_power'}
            | _plan_noise(*_OBJECT_BRANCHES)
            | _optimizer_noise(*_OBJECT_BRANCHES)
            | extras
        ),
        prefixes={
            _PLAN: 'OptimizationPlan',
            'slice_spacing_options': 'OptimizeSliceSpacing',
            'l1_norm_constraint': 'ConstrainL1Norm',
            'l2_norm_constraint': 'ConstrainL2Norm',
            'smoothness_constraint': 'ConstrainSmoothness',
            'total_variation': 'ConstrainTotalVariation',
            'hard_limits_magnitude_phase': 'ConstrainHardLimits',
            'remove_grid_artifacts': 'RemoveGridArtifacts',
            'multislice_regularization': 'RegularizeMultislice',
            'remove_object_probe_ambiguity': 'RemoveObjectProbeAmbiguity',
            **_plans(*_OBJECT_BRANCHES),
        },
        renames={
            'optimizable': 'IsOptimizable',
            'remove_grid_artifacts.window_size': 'WindowSizeInPixels',
            'multislice_regularization.unwrap_image_grad_method': 'UnwrapPhaseImageGradientMethod',
            'multislice_regularization.unwrap_image_integration_method': 'UnwrapPhaseImageIntegrationMethod',
        },
        structural={
            'hard_limits_magnitude_phase.abs_lim': OptionalRange('EnableAbs', 'AbsMin', 'AbsMax'),
            'hard_limits_magnitude_phase.phase_lim': OptionalRange(
                'EnablePhase', 'PhaseMinDeg', 'PhaseMaxDeg'
            ),
            **_stops(*_OBJECT_BRANCHES),
        },
        labels=_OBJECT_LABELS,
    )


_PROBE_LABELS: Mapping[str, Label] = {
    'optimizable': Label('Optimize Probe', 'Whether the probe is optimizable'),
    'optimizer': Label('Optimizer:'),
    'step_size': Label('Step Size:', 'Optimizer step size (learning rate).'),
    'eigenmode_update_relaxation': Label(
        'Relax Eigenmode Update:', 'Separate step size (relaxation) for the eigenmode update.'
    ),
    'power_constraint.enabled': Label('Constrain Power', 'Whether to constrain probe power'),
    'power_constraint.scale_object': Label(
        'Scale Object',
        'Scale the object inversely when the probe power is constrained; otherwise only '
        'the probe is rescaled.',
    ),
    'orthogonalize_incoherent_modes.enabled': Label(
        'Orthogonalize Incoherent Modes', 'Whether to orthogonalize incoherent probe modes'
    ),
    'orthogonalize_incoherent_modes.method': Label(
        'Method:', 'Method to use for incoherent mode orthogonalization'
    ),
    'orthogonalize_incoherent_modes.sort_by_occupancy': Label(
        'Sort By Occupancy',
        'Keep modes sorted so the highest-occupancy mode is the first shared mode.',
    ),
    'orthogonalize_opr_modes.enabled': Label(
        'Orthogonalize OPR Modes', 'Whether to orthogonalize OPR modes'
    ),
    'support_constraint.enabled': Label(
        'Constrain Support',
        'When enabled, the probe will be shrinkwrapped so that small values are set to zero',
    ),
    'support_constraint.threshold': Label(
        'Threshold:',
        'Shrinkwrap threshold: probe pixels below this fraction of the blurred maximum are '
        'set to zero.',
    ),
    'support_constraint.fixed_probe_support': Label(
        'Method:',
        'Optional fixed support mask (ellipse or rectangle) applied before shrinkwrapping.',
    ),
    'center_constraint.enabled': Label(
        'Constrain Center',
        'When enabled, the probe center of mass will be constrained to the center of the '
        'probe array',
    ),
    'center_constraint.use_total_intensity_for_com': Label(
        'Use Intensity for Mass Centroid',
        'Use the total probe intensity for the center-of-mass calculation instead of the '
        "dominant shared mode's magnitude.",
    ),
    'center_constraint.center_modes_individually': Label(
        'Center Modes Individually', 'When checked, each probe mode will be centered independently'
    ),
}


def _probe_policy(extras: set[str]) -> FlatteningPolicy:
    return FlatteningPolicy(
        hidden=frozenset(
            {
                'initial_guess',
                'pixel_size_m',
                'pixel_size_aspect_ratio',
                'power_constraint.probe_power',
            }
            | {
                'support_constraint.fixed_probe_support_params',
                'center_constraint.use_intensity_for_com',
            }
            | _plan_noise(*_PROBE_BRANCHES)
            | _optimizer_noise(*_PROBE_BRANCHES)
            | extras
        ),
        prefixes={
            _PLAN: 'OptimizationPlan',
            'power_constraint': 'ConstrainProbePower',
            'orthogonalize_incoherent_modes': 'OrthogonalizeIncoherentModes',
            'orthogonalize_opr_modes': 'OrthogonalizeOPRModes',
            'support_constraint': 'ConstrainSupport',
            'center_constraint': 'ConstrainCenter',
            **_plans(*_PROBE_BRANCHES),
        },
        renames={
            'optimizable': 'IsOptimizable',
            'support_constraint.fixed_probe_support': 'Method',
            'center_constraint.use_total_intensity_for_com': '/UseTotalIntensityForCOM',
            # The prefix already says Center.
            'center_constraint.center_modes_individually': 'ModesIndividually',
        },
        labels=_PROBE_LABELS,
        structural=dict(_stops(*_PROBE_BRANCHES)),
    )


_POSITION_LABELS: Mapping[str, Label] = {
    'optimizable': Label('Optimize Probe Positions', 'Whether the probe positions are optimizable'),
    'optimizer': Label('Optimizer:'),
    'step_size': Label('Step Size:', 'Optimizer step size (learning rate).'),
    'constrain_position_mean': Label(
        'Constrain Centroid',
        'Whether to subtract the mean from positions after updating positions',
    ),
    'correction_options.correction_type': Label(
        'Correction Type:', 'Algorithm used to calculate the position correction update'
    ),
    'correction_options.differentiation_method': Label(
        'Differentiation Method:',
        'Method for computing the object gradient (GRADIENT correction only); '
        'FOURIER_DIFFERENTIATION is usually fastest.',
    ),
    'correction_options.cross_correlation_scale': Label(
        'Scale:', 'Upsampling factor of the cross-correlation in real space'
    ),
    'correction_options.cross_correlation_real_space_width': Label(
        'Real Space Width:', 'Width of the cross-correlation in real-space'
    ),
    'correction_options.cross_correlation_probe_threshold': Label(
        'Probe Threshold:', 'Probe intensity threshold used to calculate the probe mask'
    ),
    'correction_options.clip_update_magnitude_by_mad': Label(
        'Clip Update Magnitude by MAD',
        'Whether to clip the update magnitude by the median absolute deviation',
    ),
    'ChooseSliceForCorrection': Label(
        'Slice For Correction:',
        'Specify the object slice used for position correction; otherwise the middle slice '
        'is used.',
    ),
    'SliceForCorrection': Label(
        'Slice:', 'Slice that is used for position correction (0-based index)'
    ),
    'LimitUpdateMagnitude': Label(
        'Limit Update Magnitude:', 'Whether to limit the update magnitude'
    ),
    'UpdateMagnitudeLimit': Label(
        'Limit:', 'Maximum allowed magnitude of position update in each axis'
    ),
    'affine_transform_constraint.enabled': Label(
        'Constrain Affine Transform',
        'Fit an affine transform from the initial to the current positions and penalize '
        'positions that deviate from it.',
    ),
    'affine_transform_constraint.degrees_of_freedom': Label('Degrees of Freedom:'),
    'affine_transform_constraint.position_weight_update_interval': Label(
        'Weight Update Interval:', 'Number of epochs between position weight updates.'
    ),
    'affine_transform_constraint.apply_constraint': Label(
        'Apply Constraint',
        'Apply the fitted transform to the positions; when off, weights are still computed '
        'for logging but positions are unchanged.',
    ),
    'affine_transform_constraint.max_expected_error': Label(
        'Max Expected Error [px]:',
        'Maximum expected position error in pixels; used only to estimate friction in the '
        'affine constraint.',
    ),
    '/OverrideAffineTransformUpdateFlexibility': Label(
        'Override Update Flexibility:',
        'When checked, the affine transform update flexibility will be overridden',
    ),
    'OverrideUpdateFlexibility': Label(
        'Flexibility:', 'Update flexibility value (0 = rigid, 1 = fully flexible)'
    ),
}


def _position_policy(extras: set[str]) -> FlatteningPolicy:
    return FlatteningPolicy(
        hidden=frozenset(
            {'position_x_px', 'position_y_px'}
            | _plan_noise('affine_transform_constraint')
            | _optimizer_noise()
            | extras
        ),
        prefixes={
            _PLAN: 'OptimizationPlan',
            'correction_options': '',
            'affine_transform_constraint': 'ConstrainAffineTransform',
            **_plans('affine_transform_constraint'),
        },
        renames={
            'optimizable': 'IsOptimizable',
            'correction_options.clip_update_magnitude_by_mad': 'ClipUpdateMagnitudeByMAD',
            'affine_transform_constraint.max_expected_error': 'MaxExpectedErrorInPixels',
        },
        structural={
            'correction_options.slice_for_correction': OptionalGate(
                'ChooseSliceForCorrection', 'SliceForCorrection', value_type=int, fallback=0
            ),
            'correction_options.update_magnitude_limit': OptionalGate(
                'LimitUpdateMagnitude', 'UpdateMagnitudeLimit', fallback=0.1
            ),
            'affine_transform_constraint.override_update_flexibility': OptionalGate(
                '/OverrideAffineTransformUpdateFlexibility', 'OverrideUpdateFlexibility'
            ),
            **_stops('affine_transform_constraint'),
        },
        labels=_POSITION_LABELS,
    )


# Ported from the panels, because an upstream declaration carries no text. The
# wording is deliberately short: each sits inside a group box that supplies the
# rest of the context, and a group box's own title is the label of the key its
# branch collapses onto.
#
# A plan's Start/Stop/Stride triple carries no text. The panels render all
# twenty-six of them as one composite row labelled 'Plan:', so that label is the
# same everywhere and belongs to whatever draws the composite, not here.
_OPR_LABELS: Mapping[str, Label] = {
    'optimizable': Label(
        'Orthogonal Probe Relaxation', 'Master switch for optimizing OPR mode weights.'
    ),
    'optimizer': Label('Optimizer:'),
    'step_size': Label('Step Size:', 'Optimizer step size (learning rate).'),
    'optimize_eigenmode_weights': Label(
        'Optimize Eigenmode Weights',
        'Whether to optimize eigenmode weights (i.e., the weights of the second and '
        'following OPR modes)',
    ),
    'optimize_intensity_variation': Label(
        'Optimize Intensities',
        'Whether to optimize intensity variation (i.e., the weight of the first OPR mode)',
    ),
    'update_relaxation': Label(
        'Relax Update:', 'Separate step size (relaxation) for the OPR mode weight update.'
    ),
    'EnablePrimaryModeWeightFloor': Label(
        'Primary Mode Weight Floor',
        'Clamp the weight of the primary (0th) OPR mode to a lower bound',
    ),
    'PrimaryModeWeightFloor': Label(
        'Floor:', 'Lower bound applied to the primary OPR mode weight.'
    ),
    'smoothing.enabled': Label('Smooth OPR Mode Weights', 'Smooth the OPR mode weights'),
    'smoothing.method': Label(
        'Smoothing Method:',
        'Method for smoothing OPR mode weights (MEDIAN filter or POLYNOMIAL fit).',
    ),
    'smoothing.polynomial_degree': Label(
        'Polynomial Degree:', 'Degree of the polynomial used for smoothing OPR mode weights'
    ),
}


def _opr_policy(extras: set[str]) -> FlatteningPolicy:
    return FlatteningPolicy(
        hidden=frozenset(
            {'initial_weights'}
            | _plan_noise(*_OPR_BRANCHES)
            | _optimizer_noise(*_OPR_BRANCHES)
            | extras
        ),
        prefixes={
            _PLAN: 'OptimizationPlan',
            'smoothing': 'SmoothModeWeights',
            **_plans(*_OPR_BRANCHES),
        },
        renames={
            'optimizable': 'IsOptimizable',
            'smoothing.method': '/SmoothingMethod',
            'smoothing.polynomial_degree': '/PolynomialSmoothingDegree',
        },
        labels=_OPR_LABELS,
        structural={
            'primary_mode_weight_floor': OptionalGate(
                'EnablePrimaryModeWeightFloor', 'PrimaryModeWeightFloor'
            ),
            **_stops(*_OPR_BRANCHES),
        },
    )


_RECONSTRUCTOR_LABELS: Mapping[str, Label] = {
    'num_epochs': Label('Number of Epochs:', 'Number of epochs to run'),
    'batch_size': Label('Batch Size:', 'Number of data to process in each minibatch'),
    'batching_mode': Label('Batch Mode:'),
    'compact_mode_update_clustering': Label(
        'Update Clustering:',
        'Number of epochs between cluster updates in COMPACT batching mode (0 disables).',
    ),
    'use_double_precision_for_fft': Label(
        'FFT Precision:',
        'Use double precision for critical FFT operations, overriding the compute precision.',
    ),
    'allow_nondeterministic_algorithms': Label(
        'Allow Nondeterministic Algorithms',
        'Allow nondeterministic algorithms, which can be faster but cause larger '
        'run-to-run variation.',
    ),
    'UseDoublePrecision': Label(
        'Compute Precision:', 'Floating point precision to use for computation'
    ),
    'EnableExcludeMeasuredPixelsBelow': Label(
        'Exclude Measured Pixels Below',
        'When enabled, measured pixels below the threshold will be excluded',
    ),
    'ExcludeMeasuredPixelsBelow': Label(
        'Threshold:',
        'Gradients for measured pixels with intensity at or below this threshold are zeroed.',
    ),
    'forward_model_options.low_memory_mode': Label(
        'Use Low Memory Mode',
        'When checked, forward propagation of ptychography will be done using less '
        'vectorized code. This reduces the speed, but also lowers memory usage',
    ),
    'PadForShift': Label(
        'Pad For Shift:', 'Number of pixels to pad arrays (with border values) before shifting'
    ),
    'EnableDiffractionPatternBlur': Label(
        'Blur Diffraction Patterns',
        'When enabled, diffraction patterns will be blurred with a Gaussian kernel',
    ),
    'DiffractionPatternBlurSigma': Label(
        'Sigma:',
        'Sigma (in pixels) of the Gaussian kernel used to blur simulated diffraction patterns.',
    ),
}

_DATA_LABELS: Mapping[str, Label] = {
    'fft_shift': Label(
        'FFT Shift Diffraction Patterns',
        'When checked, the diffraction patterns will be FFT-shifted',
    ),
    'save_data_on_device': Label(
        'Save Data on Device', 'Keep the diffraction data on the compute device (e.g. GPU).'
    ),
}

_ALGORITHM_LABELS: Mapping[str, Label] = {
    # Reconstructor-branch additions, which ptychodus files under the algorithm.
    'loss_function': Label('Loss Function:', 'Loss function to optimize'),
    'forward_model_class': Label('Forward Model:', 'Forward model class'),
    'exit_wave_update_relaxation': Label(
        'Exit Wave Update Relaxation:', 'Relaxation multiplier for the exit wave update'
    ),
    'chunk_length': Label('Chunk Length:'),
    'beta': Label('Beta:'),
    'method': Label('BH Method:'),
    'probe_rho': Label(
        'BH Probe Rho:',
        'Scaling factor for the probe relative to the object (~1 near-field, 0.1 far-field).',
    ),
    'probe_position_rho': Label(
        'BH Probe Position Rho:',
        'Scaling factor for probe positions relative to the object (~0.1 near-field, 1 far-field).',
    ),
    'noise_model': Label('Noise Model:', 'Noise model to use'),
    'gaussian_noise_std': Label(
        'Gaussian Noise Deviation:',
        'Standard deviation of the Gaussian noise (used only with the Gaussian noise model).',
    ),
    'single_slice_solve_obj_prb_step_size_jointly': Label(
        'Solve Object/Probe Step Size Jointly (Single Slice)',
        'When checked, object and probe step lengths are solved simultaneously in '
        'single-slice mode',
    ),
    'multislice_solve_obj_prb_step_size_jointly': Label(
        'Solve Object/Probe Step Size Jointly (Multislice)',
        'When checked, object and probe step lengths are solved simultaneously in multislice mode',
    ),
    'solve_step_sizes_only_using_first_probe_mode': Label(
        'Solve Step Sizes Only Using First Probe Mode',
        'When checked, the step sizes will be calculated using only the first probe mode',
    ),
    'momentum_acceleration_gain': Label(
        'Momentum Acceleration Gain:',
        'Gain of momentum acceleration for object and probe; 0 disables it.',
    ),
    'AutoMomentumAccelerationGradientMixingFactor': Label(
        'Use Gradient Mixing Factor',
        'Mixing factor of the current gradient into the accumulated velocity; set to 1 to '
        'reproduce PtychoShelves.',
    ),
    'MomentumAccelerationGradientMixingFactor': Label('Mixing Factor:'),
    'rescale_probe_intensity_in_first_epoch': Label(
        'Rescale Probe Intensity In First Epoch',
        'When checked, the probe intensity will be rescaled on the first epoch',
    ),
    'preconditioning_damping_factor': Label(
        'Preconditioning Damping Factor:', 'Damping factor for preconditioning the object update'
    ),
    'momentum_acceleration_memory': Label(
        'Position Momentum Memory:',
        'Number of previous epochs used to estimate friction in probe-position momentum '
        'acceleration.',
    ),
    'AutoMomentumGradientMixingFactor': Label(
        'Use Position Gradient Mixing Factor',
        'Mixing factor of the current position update into the accumulated velocity; set to '
        '1 to reproduce foldslice.',
    ),
    # Branch additions other than the reconstructor's.
    'amplitude_clamp_limit': Label(
        'Amplitude Clamp Limit:',
        'Maximum object amplitude; larger values are clamped to this limit.',
    ),
    'inertia': Label('Inertia:', 'Inertia of the object update; should be between 0 and 1.'),
    'alpha': Label('Alpha:'),
    'optimal_step_size_scaler': Label(
        'Optimal Step Size Scaler:',
        'Scaler applied to the solved optimal step size for the object update.',
    ),
    'multimodal_update': Label(
        'Multimodal Update',
        'Compute the object update direction summed over all probe modes rather than only '
        'the first mode.',
    ),
}


def _reconstructor_policy(extras: set[str]) -> FlatteningPolicy:
    """The shared half of ``reconstructor_options``; the rest is the algorithm's own."""
    return FlatteningPolicy(
        hidden=frozenset(
            {
                'compact_mode_update_clustering_stride',
                'default_device',
                'random_seed',
                'displayed_loss_function',
            }
            | extras
        ),
        prefixes={'forward_model_options': ''},
        renames={
            'num_epochs': 'NumEpochs',
            'use_double_precision_for_fft': 'UseDoublePrecisionForFFT',
            'forward_model_options.low_memory_mode': 'UseLowMemoryMode',
        },
        structural={
            'default_dtype': BoolEnum('UseDoublePrecision'),
            'exclude_measured_pixels_below': OptionalGate(
                'EnableExcludeMeasuredPixelsBelow', 'ExcludeMeasuredPixelsBelow'
            ),
            'forward_model_options.pad_for_shift': Sentinel(
                'PadForShift', value_type=int, sentinel=0
            ),
            'forward_model_options.diffraction_pattern_blur_sigma': OptionalGate(
                'EnableDiffractionPatternBlur', 'DiffractionPatternBlurSigma', fallback=1.0
            ),
        },
        labels=_RECONSTRUCTOR_LABELS,
    )


def _data_policy() -> FlatteningPolicy:
    return FlatteningPolicy(
        # All but two are filled from the product at reconstruct time.
        hidden=frozenset(
            {'data', 'free_space_propagation_distance_m', 'wavelength_m', 'valid_pixel_mask'}
        ),
        renames={'fft_shift': 'FFTShiftDiffractionPatterns'},
        labels=_DATA_LABELS,
    )


_ALGORITHM_GROUPS: Mapping[str, str] = {
    'Autodiff': 'PtyChiAutodiff',
    'BH': 'PtyChiBH',
    'DM': 'PtyChiDM',
    'LSQML': 'PtyChiLSQML',
    'PIE': 'PtyChiPIE',
    'ePIE': 'PtyChiPIE',
    'rPIE': 'PtyChiPIE',
    'RAAR': 'PtyChiRAAR',
}


def sections_for(algorithm: str, options: Any) -> Sequence[OptionsSection]:
    """Return the settings sections backing *algorithm*, given its declared options.

    The shared sections hide whatever the algorithm's subclass adds to each
    branch, because ptychodus files those additions under the algorithm's own
    group rather than the shared one.
    """
    from ptychi.api.options.base import (
        ObjectOptions,
        OPRModeWeightsOptions,
        ProbeOptions,
        ProbePositionOptions,
        ReconstructorOptions,
    )

    return [
        OptionsSection(
            'PtyChi',
            root_path='reconstructor_options',
            policy=_reconstructor_policy(
                _extras(options.reconstructor_options, ReconstructorOptions)
            ),
        ),
        OptionsSection('PtyChi', root_path='data_options', policy=_data_policy()),
        OptionsSection(
            'PtyChiObject',
            root_path='object_options',
            policy=_object_policy(_extras(options.object_options, ObjectOptions)),
        ),
        OptionsSection(
            'PtyChiProbe',
            root_path='probe_options',
            policy=_probe_policy(_extras(options.probe_options, ProbeOptions)),
        ),
        OptionsSection(
            'PtyChiProbePosition',
            root_path='probe_position_options',
            policy=_position_policy(_extras(options.probe_position_options, ProbePositionOptions)),
        ),
        OptionsSection(
            'PtyChiOPR',
            root_path='opr_mode_weight_options',
            policy=_opr_policy(_extras(options.opr_mode_weight_options, OPRModeWeightsOptions)),
        ),
        *_algorithm_sections(algorithm, options),
    ]


# Additions ptychodus does not surface: pty-chi's experimental branches, and the
# free-form parameter bag beside the Autodiff forward model.
_UNSURFACED_EXTRAS = frozenset({'experimental', 'forward_model_params'})


def _only_extras(instance: Any, base: type) -> frozenset[str]:
    """Hide everything the base class already declares, leaving the additions."""
    shared = set(fields_of(base)) & set(fields_of(type(instance)))
    return frozenset(shared | _UNSURFACED_EXTRAS)


_AUTO_MIXING = OptionalGate(
    'AutoMomentumAccelerationGradientMixingFactor',
    'MomentumAccelerationGradientMixingFactor',
    inverted=True,
)


def _algorithm_sections(algorithm: str, options: Any) -> Sequence[OptionsSection]:
    """The group holding whatever this algorithm adds to each shared branch."""
    from ptychi.api.options.base import (
        ObjectOptions,
        OPRModeWeightsOptions,
        ProbeOptions,
        ProbePositionOptions,
        ReconstructorOptions,
    )

    group = _ALGORITHM_GROUPS[algorithm]
    branches: Sequence[
        tuple[str, type, str, Mapping[str, Label], Mapping[str, StructuralAdapter]]
    ] = (
        (
            'reconstructor_options',
            ReconstructorOptions,
            '',
            {},
            _RECONSTRUCTOR_EXTRA_ADAPTERS,
        ),
        ('object_options', ObjectOptions, 'Object', {}, {}),
        ('probe_options', ProbeOptions, 'Probe', _PROBE_EXTRA_LABELS, {}),
        (
            'probe_position_options',
            ProbePositionOptions,
            'ProbePosition',
            _POSITION_EXTRA_LABELS,
            {
                'momentum_acceleration_gradient_mixing_factor': OptionalGate(
                    'AutoMomentumGradientMixingFactor',
                    'MomentumAccelerationGradientMixingFactor',
                    inverted=True,
                )
            },
        ),
        ('opr_mode_weight_options', OPRModeWeightsOptions, 'OPR', {}, {}),
    )
    sections: list[OptionsSection] = []

    for path, base, prefix, branch_labels, structural in branches:
        root = getattr(options, path)

        if not _extras(root, base) - _UNSURFACED_EXTRAS:
            continue

        sections.append(
            OptionsSection(
                group,
                root_path=path,
                policy=FlatteningPolicy(
                    base_prefix=prefix,
                    hidden=_only_extras(root, base),
                    renames=_RECONSTRUCTOR_EXTRA_NAMES if path == 'reconstructor_options' else {},
                    labels={**_ALGORITHM_LABELS, **branch_labels},
                    structural=structural,
                ),
            )
        )

    return sections


_RECONSTRUCTOR_EXTRA_NAMES: Mapping[str, str] = {
    'multislice_solve_obj_prb_step_size_jointly': 'MultisliceSolveObjectProbeStepSizeJointly',
    'single_slice_solve_obj_prb_step_size_jointly': 'SingleSliceSolveObjectProbeStepSizeJointly',
}

# 'rho' names a different quantity on each branch, so the text cannot be shared.
_PROBE_EXTRA_LABELS: Mapping[str, Label] = {
    'rho': Label(
        'BH Probe Rho:',
        'Scaling factor for the probe relative to the object (~1 near-field, 0.1 far-field).',
    ),
}

_POSITION_EXTRA_LABELS: Mapping[str, Label] = {
    'rho': Label(
        'BH Probe Position Rho:',
        'Scaling factor for probe positions relative to the object (~0.1 near-field, 1 far-field).',
    ),
}

_RECONSTRUCTOR_EXTRA_ADAPTERS: Mapping[str, StructuralAdapter] = {
    'momentum_acceleration_gradient_mixing_factor': _AUTO_MIXING,
}
