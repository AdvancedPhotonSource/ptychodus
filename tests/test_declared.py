"""Unit tests for reading a ptychodus-owned declaration back out.

The load-bearing case is :class:`TestLegacyProbePositionKeys`. The scan-pattern
settings were hand-written long before anything was declared, and the generators
that replace them must land on the keys already in every settings file. If the
naming rule stopped reproducing them, every one of those files would quietly fall
back to defaults, which looks identical to loading correctly.
"""

from __future__ import annotations

from enum import StrEnum

import pytest
from pydantic import BaseModel, Field

from ptychodus.api.constants import LengthUnit, Unit
from ptychodus.model.declared import ParameterDescription, describe, render_key


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


def _by_name(options: BaseModel, **kwargs: bool) -> dict[str, ParameterDescription]:
    return {d.name: d for d in describe(options, **kwargs)}


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
