"""Unit tests for LengthUnit, AngleUnit, and format_length."""

from __future__ import annotations
from decimal import Decimal

import pytest

from ptychodus.api.constants import HC_EV_ANGSTROM, TWO_PI, AngleUnit, LengthUnit, format_length


@pytest.mark.parametrize(
    'length_m, expected',
    [
        (1e-15, LengthUnit.PICOMETER),
        (1e-12, LengthUnit.PICOMETER),
        (9.99e-10, LengthUnit.PICOMETER),
        (1e-9, LengthUnit.NANOMETER),
        (9.99e-7, LengthUnit.NANOMETER),
        (1e-6, LengthUnit.MICROMETER),
        (1e-3, LengthUnit.MILLIMETER),
        (1.0, LengthUnit.METER),
        (1e30, LengthUnit.METER),
    ],
)
def test_from_meters_picks_the_largest_fitting_unit(length_m: float, expected: LengthUnit) -> None:
    assert LengthUnit.from_meters(length_m) is expected


def test_from_meters_ignores_sign() -> None:
    assert LengthUnit.from_meters(-1.5e-9) is LengthUnit.NANOMETER
    assert LengthUnit.from_meters(-2.0) is LengthUnit.METER


def test_from_meters_maps_zero_to_meters() -> None:
    """Zero carries no magnitude, so the ladder walk would otherwise return picometers."""
    assert LengthUnit.from_meters(0.0) is LengthUnit.METER


def test_from_meters_tolerates_non_finite_values() -> None:
    """Reachable from an INI: RealParameter.set_value_from_string accepts "nan"."""
    assert LengthUnit.from_meters(float('nan')) is LengthUnit.METER
    assert LengthUnit.from_meters(float('inf')) is LengthUnit.METER


def test_from_meters_skips_the_angstrom() -> None:
    """The angstrom is a decade, not a 1000-step, so auto-selection passes it over."""
    assert LengthUnit.from_meters(5e-10) is LengthUnit.PICOMETER
    assert not LengthUnit.ANGSTROM.is_si_prefixed
    assert all(unit.is_si_prefixed for unit in LengthUnit if unit is not LengthUnit.ANGSTROM)


def test_angstrom_is_still_usable_when_selected_explicitly() -> None:
    assert LengthUnit.ANGSTROM.convert(1.5e-10) == pytest.approx(1.5)
    assert LengthUnit.ANGSTROM.format(1.5e-10) == '1.5 Å'


def test_convert_and_to_meters_are_inverses() -> None:
    assert LengthUnit.MICROMETER.to_meters(800.0) == pytest.approx(8e-4)
    assert LengthUnit.MICROMETER.convert(8e-4) == pytest.approx(800.0)
    assert LengthUnit.NANOMETER.convert(LengthUnit.NANOMETER.to_meters(42.0)) == pytest.approx(42.0)


@pytest.mark.parametrize(
    'length_m, expected',
    [
        (0.0, '0 m'),
        (5e-13, '0.5 pm'),
        (1.234e-9, '1.234 nm'),
        (-1.5e-9, '-1.5 nm'),
        (1.2e-6, '1.2 µm'),
        (0.0013, '1.3 mm'),
    ],
)
def test_format_length_end_to_end(length_m: float, expected: str) -> None:
    assert format_length(length_m) == expected


def test_meters_per_unit_is_the_exact_float_literal() -> None:
    assert LengthUnit.MILLIMETER.meters_per_unit == 1e-3
    assert LengthUnit.MICROMETER.meters_per_unit == 1e-6
    assert LengthUnit.NANOMETER.meters_per_unit == 1e-9
    assert LengthUnit.ANGSTROM.meters_per_unit == 1e-10
    assert LengthUnit.PICOMETER.meters_per_unit == 1e-12


def test_decimal_meters_per_unit_is_exact() -> None:
    """The GUI multiplies by this, so binary-float contamination would be visible on screen."""
    for unit in LengthUnit:
        assert unit.decimal_meters_per_unit == Decimal(f'1e{unit.power_of_ten}')

    # Deriving the Decimal from the float instead would give 9.99999...E-7 here.
    assert Decimal(LengthUnit.MICROMETER.meters_per_unit) != Decimal('1e-6')


def test_meter_scale_factor_does_not_add_a_trailing_zero() -> None:
    """Decimal('1.0') would render a typed "1.5" back to the user as "1.50"."""
    assert LengthUnit.METER.decimal_meters_per_unit == Decimal('1')
    assert str(Decimal('1.5') * LengthUnit.METER.decimal_meters_per_unit) == '1.5'


def test_hc_ev_angstrom_is_unchanged() -> None:
    assert HC_EV_ANGSTROM == pytest.approx(12398.419843320025)


def test_angle_units_per_turn() -> None:
    assert AngleUnit.TURN.units_per_turn == 1.0
    assert AngleUnit.DEGREE.units_per_turn == 360.0
    assert AngleUnit.RADIAN.units_per_turn == TWO_PI


def test_angle_convert_and_to_turns_are_inverses() -> None:
    assert AngleUnit.DEGREE.convert(0.25) == pytest.approx(90.0)
    assert AngleUnit.DEGREE.to_turns(90.0) == pytest.approx(0.25)
    assert AngleUnit.RADIAN.convert(0.5) == pytest.approx(TWO_PI / 2)


def test_decimal_units_per_turn_is_exact_for_the_rational_units() -> None:
    assert AngleUnit.TURN.decimal_units_per_turn == Decimal(1)
    assert AngleUnit.DEGREE.decimal_units_per_turn == Decimal(360)
    assert AngleUnit.RADIAN.decimal_units_per_turn == Decimal.from_float(TWO_PI)
