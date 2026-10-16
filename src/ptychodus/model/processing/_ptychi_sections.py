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
    )


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
        structural=dict(_stops(*_PROBE_BRANCHES)),
    )


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
    )


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
        structural={
            'primary_mode_weight_floor': OptionalGate(
                'EnablePrimaryModeWeightFloor', 'PrimaryModeWeightFloor'
            ),
            **_stops(*_OPR_BRANCHES),
        },
    )


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
    )


def _data_policy() -> FlatteningPolicy:
    return FlatteningPolicy(
        # All but two are filled from the product at reconstruct time.
        hidden=frozenset(
            {'data', 'free_space_propagation_distance_m', 'wavelength_m', 'valid_pixel_mask'}
        ),
        renames={'fft_shift': 'FFTShiftDiffractionPatterns'},
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
        tuple[str, type, str, Mapping[str, str], Mapping[str, StructuralAdapter]]
    ] = (
        (
            'reconstructor_options',
            ReconstructorOptions,
            '',
            _RECONSTRUCTOR_EXTRA_NAMES,
            _RECONSTRUCTOR_EXTRA_ADAPTERS,
        ),
        ('object_options', ObjectOptions, 'Object', {}, {}),
        ('probe_options', ProbeOptions, 'Probe', {}, {}),
        (
            'probe_position_options',
            ProbePositionOptions,
            'ProbePosition',
            {},
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

    for path, base, prefix, renames, structural in branches:
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
                    renames=renames,
                    structural=structural,
                ),
            )
        )

    return sections


_RECONSTRUCTOR_EXTRA_NAMES: Mapping[str, str] = {
    'multislice_solve_obj_prb_step_size_jointly': 'MultisliceSolveObjectProbeStepSizeJointly',
    'single_slice_solve_obj_prb_step_size_jointly': 'SingleSliceSolveObjectProbeStepSizeJointly',
}

_RECONSTRUCTOR_EXTRA_ADAPTERS: Mapping[str, StructuralAdapter] = {
    'momentum_acceleration_gradient_mixing_factor': _AUTO_MIXING,
}
