"""Unit tests for reading a ptychodus-owned declaration back out.

The load-bearing case is :class:`TestLegacyProbePositionKeys`. The scan-pattern
settings were hand-written long before anything was declared, and the generators
that replace them must land on the keys already in every settings file. If the
naming rule stopped reproducing them, every one of those files would quietly fall
back to defaults, which looks identical to loading correctly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ptychodus.api.constants import LengthUnit, Unit
from ptychodus.model.parameters import ComplexSequenceParameter, Parameter, ParameterGroup
from ptychodus.model.declared import (
    ParameterDescription,
    declare_in,
    describe,
    render_key,
    restore,
)


class TestRenderKey:
    def test_words_join_in_pascal_case(self) -> None:
        assert render_key('radius_scalar') == 'RadiusScalar'

    def test_a_unit_suffix_is_spelled_out(self) -> None:
        assert render_key('step_size_x_m') == 'StepSizeXInMeters'
        assert render_key('angular_shift_tr') == 'AngularShiftInTurns'
        assert render_key('probe_energy_eV') == 'ProbeEnergyInElectronVolts'

    def test_num_abbreviates_by_default(self) -> None:
        """Two backends already persist NumEpochs; expanding theirs would rename it."""
        assert render_key('num_epochs') == 'NumEpochs'
        assert render_key('num_sync_epochs') == 'NumSyncEpochs'

    def test_num_expands_on_request(self) -> None:
        assert render_key('num_epochs', expand_num=True) == 'NumberOfEpochs'

    def test_num_expands_only_as_a_whole_word(self) -> None:
        assert render_key('numerator', expand_num=True) == 'Numerator'

    def test_the_expansion_composes_with_a_unit_suffix(self) -> None:
        assert render_key('num_steps_m', expand_num=True) == 'NumberOfStepsInMeters'


class TestLegacyProbePositionKeys:
    """Every key the hand-written [ProbePositions] section holds for a pattern."""

    @pytest.mark.parametrize(
        'field_name, legacy_key',
        [
            ('num_points_x', 'NumberOfPointsX'),
            ('num_points_y', 'NumberOfPointsY'),
            ('step_size_x_m', 'StepSizeXInMeters'),
            ('step_size_y_m', 'StepSizeYInMeters'),
            ('radial_step_size_m', 'RadialStepSizeInMeters'),
            ('num_shells', 'NumberOfShells'),
            ('num_points_in_first_shell', 'NumberOfPointsInFirstShell'),
            ('radius_scalar_m', 'RadiusScalarInMeters'),
            ('amplitude_x_m', 'AmplitudeXInMeters'),
            ('amplitude_y_m', 'AmplitudeYInMeters'),
            ('angular_step_x_tr', 'AngularStepXInTurns'),
            ('angular_step_y_tr', 'AngularStepYInTurns'),
            ('angular_shift_tr', 'AngularShiftInTurns'),
        ],
    )
    def test_the_rule_reproduces_it(self, field_name: str, legacy_key: str) -> None:
        assert render_key(field_name, expand_num=True) == legacy_key

    def test_the_shared_count_gets_a_key_of_its_own(self) -> None:
        """Spiral and lissajous used to read cartesian's NumberOfPointsX."""
        assert render_key('num_points', expand_num=True) == 'NumberOfPoints'


class Shape(StrEnum):
    DISK = 'disk'
    RING = 'ring'


class Options(BaseModel):
    count: int = Field(7, ge=1, le=99, title='Count:', description='How many.')
    width_m: float = Field(1e-6, gt=0.0)
    unbounded: float = 3.0
    enabled: bool = True
    shape: Shape = Shape.RING
    defocus_m: float = Field(0.0, json_schema_extra={'display_unit': 'MICROMETER'})


def _by_name(
    options: BaseModel, *, expand_num: bool = False, prefix: str = ''
) -> dict[str, ParameterDescription]:
    return {d.name: d for d in describe(options, expand_num=expand_num, prefix=prefix)}


class TestDescribe:
    def test_fields_come_back_in_declaration_order(self) -> None:
        names = [d.name for d in describe(Options())]
        assert names == ['count', 'width_m', 'unbounded', 'enabled', 'shape', 'defocus_m']

    def test_display_text_comes_off_the_field(self) -> None:
        count = _by_name(Options())['count']
        assert count.label == 'Count:'
        assert count.tool_tip == 'How many.'

    def test_a_field_stating_no_text_reports_none_rather_than_a_name(self) -> None:
        unbounded = _by_name(Options())['unbounded']
        assert unbounded.label == ''
        assert unbounded.tool_tip == ''

    def test_inclusive_bounds_are_reported_as_declared(self) -> None:
        count = _by_name(Options())['count']
        assert (count.minimum, count.maximum) == (1, 99)

    def test_an_exclusive_bound_is_moved_inward(self) -> None:
        """A parameter clamps inclusively, so gt=0 must not admit exactly zero."""
        width = _by_name(Options())['width_m']
        assert width.minimum is not None
        assert width.minimum > 0.0
        assert width.maximum is None

    def test_the_default_is_read_off_the_instance(self) -> None:
        assert _by_name(Options(count=42))['count'].default == 42

    def test_the_unit_comes_from_the_field_name(self) -> None:
        described = _by_name(Options())
        assert described['width_m'].unit is Unit.METER
        assert described['count'].unit is None

    def test_a_declared_display_unit_is_read(self) -> None:
        assert _by_name(Options())['defocus_m'].display_unit is LengthUnit.MICROMETER

    def test_an_enum_annotation_carries_its_choices(self) -> None:
        assert list(_by_name(Options())['shape'].choices) == ['disk', 'ring']

    def test_a_plain_field_offers_no_choices(self) -> None:
        assert list(_by_name(Options())['count'].choices) == []

    def test_keys_abbreviate_num_unless_asked(self) -> None:
        class Counted(BaseModel):
            num_points: int = 1

        assert _by_name(Counted())['num_points'].key == 'NumPoints'
        assert _by_name(Counted(), expand_num=True)['num_points'].key == 'NumberOfPoints'


class TestDisplayUnitIsRefusedWhereItWouldMislead:
    def test_a_field_not_held_in_meters(self) -> None:
        class Bad(BaseModel):
            angle_tr: float = Field(0.0, json_schema_extra={'display_unit': 'MICROMETER'})

        with pytest.raises(ValueError, match='not held in meters'):
            describe(Bad())

    def test_a_name_no_length_unit_answers_to(self) -> None:
        class Bad(BaseModel):
            width_m: float = Field(0.0, json_schema_extra={'display_unit': 'FURLONG'})

        with pytest.raises(ValueError, match='unknown display unit'):
            describe(Bad())


class TestPrefix:
    def test_a_prefix_distinguishes_branches_of_the_same_shape(self) -> None:
        """Three stages of one schedule persist side by side in one group."""

        class Stage(BaseModel):
            epochs: int = 1

        assert _by_name(Stage(), prefix='Stage2')['epochs'].key == 'Stage2Epochs'

    def test_a_prefix_does_not_touch_the_declared_name(self) -> None:
        """The name is what a caller supplies values under; only the key moves."""

        class Stage(BaseModel):
            epochs: int = 1

        described = describe(Stage(), prefix='Stage3')
        assert [d.name for d in described] == ['epochs']

    def test_no_prefix_is_the_default(self) -> None:
        class Stage(BaseModel):
            epochs: int = 1

        assert _by_name(Stage())['epochs'].key == 'Epochs'


class _CoefficientOptions(BaseModel):
    scale_m: float = Field(1e-6, gt=0.0, json_schema_extra={'probe_extent': 'disk_diameter_m'})
    coefficients: list[complex] = Field(default_factory=lambda: [1 + 0j, 0.5j])


class TestComplexListsAndExtras:
    """A coefficient list persists as one parameter, and field extras pass through."""

    def _declared(
        self,
    ) -> tuple[_CoefficientOptions, list[ParameterDescription], dict[str, Parameter[Any]]]:
        options = _CoefficientOptions()
        described = list(describe(options))
        group = ParameterGroup()
        return options, described, dict(declare_in(group, described))

    def test_a_complex_list_becomes_a_complex_sequence_parameter(self) -> None:
        _, _, parameters = self._declared()
        parameter = parameters['Coefficients']

        assert isinstance(parameter, ComplexSequenceParameter)
        assert parameter.get_value() == [1 + 0j, 0.5j]

    def test_restore_carries_the_live_list_without_sharing_it(self) -> None:
        options, described, parameters = self._declared()
        parameters['Coefficients'].set_value([2j])

        restored = restore(options, described, parameters)
        parameters['Coefficients'].set_value([3j])

        assert restored.coefficients == [2j]
        assert options.coefficients == [1 + 0j, 0.5j]

    def test_field_extras_are_carried_through_untouched(self) -> None:
        _, described, _ = self._declared()
        by_name = {description.name: description for description in described}

        assert by_name['scale_m'].extra == {'probe_extent': 'disk_diameter_m'}
        assert by_name['coefficients'].extra == {}
