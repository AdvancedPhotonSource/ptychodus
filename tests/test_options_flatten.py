"""Unit tests for projecting a nested options declaration onto the flat INI layout."""

from __future__ import annotations

from dataclasses import field as dataclass_field
from enum import StrEnum
from pathlib import Path
import re

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
import pytest

from ptychodus.model.processing.flatten import (
    BoolEnum,
    FlatteningPolicy,
    Overlay,
    OptionalGate,
    OptionalRange,
    Sentinel,
    flatten_options,
    render_value,
    to_pascal_case,
)


class Optimizers(StrEnum):
    SGD = 'sgd'
    ADAM = 'adam'


class Plan(BaseModel):
    start: int = 0
    stop: int = -1
    stride: int = 1
    scheduler: str = 'none'


class Constraint(BaseModel):
    enabled: bool = False
    plan: Plan = Plan()
    weight: float = 1.0


class Options(BaseModel):
    optimizable: bool = True
    plan: Plan = Plan()
    optimizer: Optimizers = Optimizers.SGD
    step_size: float = Field(1.0, ge=0.0, le=10.0, title='Step Size:', description='Learning rate.')
    period_x_m: float = 0.0
    l1_norm: Constraint = Constraint()


class ExclusiveBounds(BaseModel):
    """A field can exclude an endpoint; a ptychodus parameter's range cannot."""

    above_zero: float = Field(1.0, gt=0.0)
    beta: float = Field(0.75, gt=0.5, le=1.0)
    below_one: float = Field(0.5, lt=1.0)
    unbounded: float = 1.0


@pydantic_dataclass
class InnerDataclass:
    weight: float = 1.0


@pydantic_dataclass
class OuterDataclass:
    inner: InnerDataclass = dataclass_field(default_factory=InnerDataclass)


# Most tests care about one rule at a time, so they hide the branches they are not
# exercising rather than asserting against the full key set.
_ONLY_SCALARS = frozenset({'plan', 'l1_norm'})
_ONLY_L1 = frozenset({'plan'})


def _keys(policy: FlatteningPolicy) -> list[str]:
    return [p.ini_key for p in flatten_options(Options(), policy)]


class TestNamingRules:
    def test_snake_case_becomes_pascal_case(self) -> None:
        assert to_pascal_case('step_size') == 'StepSize'

    def test_unit_suffixes_are_spelled_out(self) -> None:
        assert to_pascal_case('period_x_m') == 'PeriodXInMeters'
        assert to_pascal_case('window_size_px') == 'WindowSizeInPixels'

    def test_longest_unit_suffix_wins(self) -> None:
        """A short suffix must not shadow a compound one that also matches."""
        assert to_pascal_case('mass_attenuation_m2_per_kg').endswith('SquareMetersPerKilogram')


class TestFlattening:
    def test_nested_model_contributes_its_prefix(self) -> None:
        policy = FlatteningPolicy(hidden=_ONLY_L1, prefixes={'l1_norm': 'ConstrainL1Norm'})
        assert 'ConstrainL1NormWeight' in _keys(policy)

    def test_enable_flag_collapses_onto_the_bare_prefix(self) -> None:
        policy = FlatteningPolicy(hidden=_ONLY_L1, prefixes={'l1_norm': 'ConstrainL1Norm'})
        keys = _keys(policy)
        assert 'ConstrainL1Norm' in keys
        assert 'ConstrainL1NormEnabled' not in keys

    def test_root_level_flag_is_not_collapsed(self) -> None:
        """At the root there is no prefix to collapse onto, so the name is used."""
        assert 'Optimizable' in _keys(FlatteningPolicy(hidden=_ONLY_SCALARS))

    def test_empty_prefix_makes_a_nesting_transparent(self) -> None:
        """A plan nested in a constraint sits directly under the constraint's prefix."""
        policy = FlatteningPolicy(
            hidden=frozenset({'plan', 'l1_norm.plan.scheduler'}),
            prefixes={'l1_norm': 'ConstrainL1Norm', 'l1_norm.plan': ''},
        )
        assert {'ConstrainL1NormStart', 'ConstrainL1NormStop', 'ConstrainL1NormStride'} <= set(
            _keys(policy)
        )

    def test_rename_is_keyed_by_path_not_leaf_name(self) -> None:
        """The same leaf name means different things at different depths."""
        policy = FlatteningPolicy(
            hidden=frozenset({'plan', 'l1_norm.plan'}),
            prefixes={'l1_norm': 'ConstrainL1Norm'},
            renames={'optimizable': 'IsOptimizable'},
        )
        assert 'IsOptimizable' in _keys(policy)

    def test_an_optional_gate_expands_into_a_switch_and_a_value(self) -> None:
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            structural={'step_size': OptionalGate('EnableStep', 'StepSize')},
        )
        keys = _keys(policy)
        assert 'EnableStep' in keys and 'StepSize' in keys

    def test_adapted_keys_record_their_role_and_share_one_source(self) -> None:
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            structural={'step_size': OptionalGate('EnableStep', 'StepSize')},
        )
        produced = [p for p in flatten_options(Options(), policy) if p.structural_role]
        assert [p.structural_role for p in produced] == ['EnableStep', 'StepSize']
        assert {p.path for p in produced} == {'step_size'}

    def test_an_adapted_key_takes_its_type_from_the_adapter(self) -> None:
        """The switch is a bool even though the field it gates is a float."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            structural={'step_size': OptionalGate('EnableStep', 'StepSize')},
        )
        gate = next(p for p in flatten_options(Options(), policy) if p.ini_key == 'EnableStep')
        assert gate.annotation is bool

    def test_hiding_a_nested_model_hides_its_whole_subtree(self) -> None:
        keys = _keys(FlatteningPolicy(hidden=frozenset({'l1_norm'})))
        assert not [key for key in keys if key.startswith('L1Norm')]

    def test_verbatim_policy_persists_declared_field_names(self) -> None:
        """Two backends mirror their library's own snake_case names."""
        keys = _keys(FlatteningPolicy(hidden=_ONLY_SCALARS, verbatim=True))
        assert 'step_size' in keys and 'StepSize' not in keys

    def test_pydantic_dataclasses_nest_the_same_as_base_models(self) -> None:
        """pty-chi declares its options as pydantic dataclasses, not BaseModels.

        The pair is declared at module scope on purpose. This file postpones
        annotation evaluation, and pydantic cannot resolve an annotation naming a
        class local to a function, which leaves the nesting undetected and
        flattens the whole object as though it were a scalar.
        """
        flat = flatten_options(OuterDataclass(), FlatteningPolicy())
        # The prefix comes from the field name, not the nested class's name.
        assert [p.ini_key for p in flat] == ['InnerWeight']


class TestDeclaredMetadata:
    """Bounds, label and tool tip come from the declaration where it states them."""

    def _step_size(self) -> object:
        return next(
            p
            for p in flatten_options(Options(), FlatteningPolicy(hidden=_ONLY_SCALARS))
            if p.path == 'step_size'
        )

    def test_inclusive_bounds_are_read_from_pydantic(self) -> None:
        step = self._step_size()
        assert step.minimum == 0.0  # type: ignore[attr-defined]
        assert step.maximum == 10.0  # type: ignore[attr-defined]

    def test_title_and_description_become_label_and_tool_tip(self) -> None:
        step = self._step_size()
        assert step.label == 'Step Size:'  # type: ignore[attr-defined]
        assert step.tool_tip == 'Learning rate.'  # type: ignore[attr-defined]

    def _exclusive(self, path: str) -> object:
        return next(
            p for p in flatten_options(ExclusiveBounds(), FlatteningPolicy()) if p.path == path
        )

    def test_an_exclusive_lower_bound_becomes_the_closest_inclusive_one(self) -> None:
        """A field declared gt=0 must not admit exactly zero."""
        field = self._exclusive('above_zero')
        assert field.minimum > 0.0  # type: ignore[attr-defined]
        assert field.minimum == pytest.approx(0.0, abs=1.0e-11)  # type: ignore[attr-defined]

    def test_an_exclusive_upper_bound_becomes_the_closest_inclusive_one(self) -> None:
        field = self._exclusive('below_one')
        assert field.maximum < 1.0  # type: ignore[attr-defined]
        assert field.maximum == pytest.approx(1.0, abs=1.0e-11)  # type: ignore[attr-defined]

    def test_the_two_forms_of_a_bound_coexist_on_one_field(self) -> None:
        """pty-chi's RAAR beta is declared gt=0.5, le=1."""
        field = self._exclusive('beta')
        assert field.minimum > 0.5  # type: ignore[attr-defined]
        assert field.maximum == 1.0  # type: ignore[attr-defined]

    def test_a_field_with_no_constraint_is_left_unbounded(self) -> None:
        field = self._exclusive('unbounded')
        assert field.minimum is None and field.maximum is None  # type: ignore[attr-defined]

    def test_an_unannotated_field_carries_no_metadata(self) -> None:
        period = next(
            p
            for p in flatten_options(Options(), FlatteningPolicy(hidden=_ONLY_SCALARS))
            if p.path == 'period_x_m'
        )
        assert period.minimum is None and period.label == ''


class TestValueRendering:
    def test_enums_render_by_member_name_by_default(self) -> None:
        """ptychodus persists pty-chi enums as SGD, while the StrEnum value is sgd."""
        assert render_value(Optimizers.SGD, FlatteningPolicy()) == 'SGD'

    def test_enums_render_by_value_when_the_policy_says_so(self) -> None:
        assert render_value(Optimizers.SGD, FlatteningPolicy(enum_by_name=False)) == 'sgd'


class TestAgainstRealPtyChiOptions:
    """The A0 spike, kept: the worst-case group must stay exactly reproducible.

    The policy below is a fixture proving the facility can express pty-chi, not
    the production table -- that arrives with the backend conversion and covers
    all fifteen groups.
    """

    def test_object_options_reproduce_the_persisted_section(self) -> None:
        pytest.importorskip('ptychi')
        from ptychi.api.options.base import ObjectOptions

        prefixes = {
            'optimization_plan': 'OptimizationPlan',
            'slice_spacing_options': 'OptimizeSliceSpacing',
            'l1_norm_constraint': 'ConstrainL1Norm',
            'l2_norm_constraint': 'ConstrainL2Norm',
            'smoothness_constraint': 'ConstrainSmoothness',
            'total_variation': 'ConstrainTotalVariation',
            'hard_limits_magnitude_phase': 'ConstrainHardLimits',
            'remove_grid_artifacts': 'RemoveGridArtifacts',
            'multislice_regularization': 'RegularizeMultislice',
            'remove_object_probe_ambiguity': 'RemoveObjectProbeAmbiguity',
        }
        # Every nested options object carries its own plan, which sits directly
        # under that object's prefix rather than adding a segment of its own.
        prefixes.update({f'{path}.optimization_plan': '' for path in list(prefixes)[1:]})

        hidden = {
            'optimizer_params',
            'initial_guess',
            'slice_spacings_m',
            'pixel_size_m',
            'pixel_size_aspect_ratio',
            'determine_position_origin_coords_by',
            'position_origin_coords',
        }
        for path in list(prefixes):
            hidden |= {
                f'{path}.step_size_scheduler_class',
                f'{path}.step_size_scheduler_options',
                f'{path}.optimizer_params',
            }

        policy = FlatteningPolicy(
            hidden=frozenset(hidden),
            prefixes=prefixes,
            renames={
                'optimizable': 'IsOptimizable',
                'remove_grid_artifacts.window_size': 'WindowSizeInPixels',
                'multislice_regularization.unwrap_image_grad_method': (
                    'UnwrapPhaseImageGradientMethod'
                ),
                'multislice_regularization.unwrap_image_integration_method': (
                    'UnwrapPhaseImageIntegrationMethod'
                ),
            },
            structural={
                'hard_limits_magnitude_phase.abs_lim': OptionalRange(
                    'EnableAbs', 'AbsMin', 'AbsMax'
                ),
                'hard_limits_magnitude_phase.phase_lim': OptionalRange(
                    'EnablePhase', 'PhaseMinDeg', 'PhaseMaxDeg'
                ),
            },
        )

        produced = {p.ini_key for p in flatten_options(ObjectOptions(), policy)}

        section = (
            (Path(__file__).parent / 'data' / 'settings_defaults.ini')
            .read_text()
            .split('[PtyChiObject]')[1]
            .split('\n[')[0]
        )
        assert produced == set(re.findall(r'^(\w+) = ', section, re.M))


class TestStructuralAdapters:
    """The shapes ptychodus persists that are not the declared field itself."""

    def test_optional_gate_round_trips_a_value(self) -> None:
        adapter = OptionalGate('On', 'Value')
        assert adapter.decode(adapter.encode(2.5)) == 2.5

    def test_optional_gate_round_trips_absence(self) -> None:
        adapter = OptionalGate('On', 'Value')
        encoded = adapter.encode(None)
        assert encoded['On'] is False
        assert adapter.decode(encoded) is None

    def test_an_inverted_gate_reads_absence_as_on(self) -> None:
        """The 'Auto...' factors are absent upstream precisely when automatic."""
        adapter = OptionalGate('Auto', 'Value', inverted=True)
        assert adapter.encode(None)['Auto'] is True
        assert adapter.decode(adapter.encode(None)) is None
        assert adapter.decode(adapter.encode(0.3)) == 0.3

    def test_sentinel_round_trips_both_states(self) -> None:
        adapter = Sentinel('Stop', sentinel=-1)
        assert adapter.encode(None)['Stop'] == -1
        assert adapter.decode(adapter.encode(None)) is None
        assert adapter.decode(adapter.encode(7)) == 7

    def test_bool_enum_round_trips_both_members(self) -> None:
        adapter = BoolEnum('UseDouble', when_true='f64', when_false='f32')
        assert adapter.encode('f64')['UseDouble'] is True
        assert adapter.decode(adapter.encode('f64')) == 'f64'
        assert adapter.decode(adapter.encode('f32')) == 'f32'

    def test_optional_range_round_trips_both_states(self) -> None:
        adapter = OptionalRange('On', 'Min', 'Max')
        assert adapter.decode(adapter.encode(None)) is None
        assert adapter.decode(adapter.encode((0.1, 0.9))) == (0.1, 0.9)


class TestAbsoluteKeys:
    """A few pty-chi keys sit unprefixed inside an otherwise prefixed group."""

    def test_a_marked_name_escapes_the_group_prefix(self) -> None:
        policy = FlatteningPolicy(
            hidden=_ONLY_L1,
            prefixes={'l1_norm': 'ConstrainL1Norm'},
            renames={'l1_norm.weight': '/StandaloneWeight'},
        )
        keys = _keys(policy)
        assert 'StandaloneWeight' in keys
        assert 'ConstrainL1NormStandaloneWeight' not in keys

    def test_unmarked_names_are_still_prefixed(self) -> None:
        policy = FlatteningPolicy(hidden=_ONLY_L1, prefixes={'l1_norm': 'ConstrainL1Norm'})
        assert 'ConstrainL1NormWeight' in _keys(policy)


class TestLabels:
    """An upstream declaration cannot carry display text, so the policy supplies it."""

    def test_the_policy_supplies_text_a_declaration_lacks(self) -> None:
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            overlays={'period_x_m': Overlay('Period X:', 'Grid artifact period.')},
        )
        period = next(p for p in flatten_options(Options(), policy) if p.path == 'period_x_m')
        assert period.label == 'Period X:'
        assert period.tool_tip == 'Grid artifact period.'

    def test_the_policy_wins_over_the_declaration(self) -> None:
        """A model ptychodus owns may still need different wording in the panel."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS, overlays={'step_size': Overlay('Learning Rate:')}
        )
        step = next(p for p in flatten_options(Options(), policy) if p.path == 'step_size')
        assert step.label == 'Learning Rate:'
        # Untouched text still falls back to what the declaration states.
        assert step.tool_tip == 'Learning rate.'

    def test_each_adapted_key_can_be_labelled_separately(self) -> None:
        """One field becomes several keys, and a switch is not named like its value."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            structural={'step_size': OptionalGate('EnableStep', 'StepSize')},
            overlays={
                'EnableStep': Overlay('Limit Step Size'),
                'StepSize': Overlay('Step Size Limit:'),
            },
        )
        produced = {p.ini_key: p.label for p in flatten_options(Options(), policy)}
        assert produced['EnableStep'] == 'Limit Step Size'
        assert produced['StepSize'] == 'Step Size Limit:'

    def test_the_policy_supplies_a_bound_a_declaration_lacks(self) -> None:
        """Zero is the commonest bound here, so a truthiness test would drop it."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS, overlays={'period_x_m': Overlay(minimum=0.0)}
        )
        period = next(p for p in flatten_options(Options(), policy) if p.path == 'period_x_m')
        assert period.minimum == 0.0

    def test_the_policy_wins_over_a_declared_bound(self) -> None:
        """ptychodus narrows some upstream ranges; it must be able to say so."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS, overlays={'step_size': Overlay(minimum=1.0e-12)}
        )
        step = next(p for p in flatten_options(Options(), policy) if p.path == 'step_size')
        assert step.minimum == 1.0e-12
        # Resolved one at a time, so the declared maximum is left in place.
        assert step.maximum == 10.0

    def test_an_overlay_stating_only_text_leaves_a_zero_bound_alone(self) -> None:
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS, overlays={'step_size': Overlay('Learning Rate:')}
        )
        step = next(p for p in flatten_options(Options(), policy) if p.path == 'step_size')
        assert step.minimum == 0.0

    def test_each_adapted_key_can_be_bounded_separately(self) -> None:
        """A switch has no range; the value beside it does."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            structural={'step_size': OptionalGate('EnableStep', 'StepSize')},
            overlays={'StepSize': Overlay(minimum=0.1)},
        )
        produced = {p.ini_key: p for p in flatten_options(Options(), policy)}
        assert produced['StepSize'].minimum == 0.1
        assert produced['EnableStep'].minimum is None

    def test_the_policy_supplies_the_values_a_bare_string_accepts(self) -> None:
        """A field typed as an enum needs no entry; one typed as str does."""
        policy = FlatteningPolicy(
            hidden=_ONLY_SCALARS,
            overlays={'period_x_m': Overlay('Period X:', choices=('fine', 'coarse'))},
        )
        period = next(p for p in flatten_options(Options(), policy) if p.path == 'period_x_m')
        assert tuple(period.choices) == ('fine', 'coarse')
        assert period.label == 'Period X:'

    def test_a_field_with_no_entry_accepts_any_value(self) -> None:
        step = next(
            p
            for p in flatten_options(Options(), FlatteningPolicy(hidden=_ONLY_SCALARS))
            if p.path == 'step_size'
        )
        assert tuple(step.choices) == ()
