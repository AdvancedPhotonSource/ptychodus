"""Unit tests for EnergyUnit and the joule conversion helpers."""

from __future__ import annotations

import pytest

from ptychodus.api.constants import ELECTRON_VOLT_J, EnergyUnit, energy_eV_to_J, energy_J_to_eV


@pytest.mark.parametrize(
    'energy_eV, expected',
    [
        (0.0, EnergyUnit.ELECTRONVOLT),
        (1.0, EnergyUnit.ELECTRONVOLT),
        (999.0, EnergyUnit.ELECTRONVOLT),
        (1_000.0, EnergyUnit.KILOELECTRONVOLT),
        (8_047.0, EnergyUnit.KILOELECTRONVOLT),
        (999_999.0, EnergyUnit.KILOELECTRONVOLT),
        (1_000_000.0, EnergyUnit.MEGAELECTRONVOLT),
        (2.5e6, EnergyUnit.MEGAELECTRONVOLT),
    ],
)
def test_from_electronvolts_picks_the_largest_fitting_unit(
    energy_eV: float,  # noqa: N803
    expected: EnergyUnit,
) -> None:
    assert EnergyUnit.from_electronvolts(energy_eV) is expected


def test_from_electronvolts_clamps_above_the_largest_unit() -> None:
    assert EnergyUnit.from_electronvolts(1e30) is EnergyUnit.MEGAELECTRONVOLT


def test_from_electronvolts_ignores_sign() -> None:
    """The ladder walk compares magnitudes, so a negative energy picks the same unit."""
    assert EnergyUnit.from_electronvolts(-8_047.0) is EnergyUnit.KILOELECTRONVOLT
    assert EnergyUnit.from_electronvolts(-0.5) is EnergyUnit.ELECTRONVOLT


def test_from_electronvolts_maps_zero_to_electronvolts() -> None:
    """Zero carries no magnitude, so the ladder walk has nothing to infer a unit from."""
    assert EnergyUnit.from_electronvolts(0.0) is EnergyUnit.ELECTRONVOLT


def test_from_electronvolts_tolerates_non_finite_values() -> None:
    """Reachable from an INI: RealParameter.set_value_from_string accepts "nan".

    LengthUnit.from_meters can leave this to the ladder walk because the meter sits at the
    top of its ladder; here the top is the megaelectronvolt, so the guard must be explicit.
    """
    assert EnergyUnit.from_electronvolts(float('nan')) is EnergyUnit.ELECTRONVOLT
    assert EnergyUnit.from_electronvolts(float('inf')) is EnergyUnit.ELECTRONVOLT
    assert EnergyUnit.from_electronvolts(float('-inf')) is EnergyUnit.ELECTRONVOLT


def test_convert_and_to_electronvolts_are_inverses() -> None:
    assert EnergyUnit.KILOELECTRONVOLT.to_electronvolts(8.047) == pytest.approx(8047.0)
    assert EnergyUnit.KILOELECTRONVOLT.convert(8047.0) == pytest.approx(8.047)
    assert EnergyUnit.MEGAELECTRONVOLT.convert(
        EnergyUnit.MEGAELECTRONVOLT.to_electronvolts(1.25)
    ) == pytest.approx(1.25)


def test_electronvolt_is_the_identity_unit() -> None:
    assert EnergyUnit.ELECTRONVOLT.to_electronvolts(8047.0) == pytest.approx(8047.0)
    assert EnergyUnit.ELECTRONVOLT.convert(8047.0) == pytest.approx(8047.0)


def test_format_in_a_fixed_unit() -> None:
    assert EnergyUnit.KILOELECTRONVOLT.format(8047.0) == '8.047 keV'
    assert EnergyUnit.ELECTRONVOLT.format(8047.0) == '8047 eV'
    assert EnergyUnit.MEGAELECTRONVOLT.format(8047.0) == '0.008047 MeV'


def test_electronvolts_per_unit_ladder_is_decimal() -> None:
    """The multiplier is derived from the exponent, so the two cannot drift apart."""
    assert EnergyUnit.ELECTRONVOLT.electronvolts_per_unit == 1.0
    assert EnergyUnit.KILOELECTRONVOLT.electronvolts_per_unit == 1e3
    assert EnergyUnit.MEGAELECTRONVOLT.electronvolts_per_unit == 1e6

    for unit in EnergyUnit:
        assert unit.electronvolts_per_unit == float(f'1e{unit.power_of_ten}')


def test_members_are_ordered_smallest_to_largest() -> None:
    """from_electronvolts walks the members in declaration order, so the order is load-bearing."""
    exponents = [unit.power_of_ten for unit in EnergyUnit]

    assert exponents == sorted(exponents)


def test_labels_are_the_conventional_spellings() -> None:
    assert [unit.label for unit in EnergyUnit] == ['eV', 'keV', 'MeV']


class TestJouleConversion:
    """The joule is not a power of ten of an electron-volt, so it is a helper, not a member."""

    def test_electronvolts_to_joules_matches_the_defining_constant(self) -> None:
        assert energy_eV_to_J(1.0) == pytest.approx(ELECTRON_VOLT_J)
        assert energy_eV_to_J(8047.0) == pytest.approx(8047.0 * ELECTRON_VOLT_J)

    def test_joules_to_electronvolts_matches_the_defining_constant(self) -> None:
        assert energy_J_to_eV(ELECTRON_VOLT_J) == pytest.approx(1.0)

    def test_round_trip_recovers_the_input_energy(self) -> None:
        for energy_eV in (100.0, 1_000.0, 8_047.0, 100_000.0):  # noqa: N806
            assert energy_J_to_eV(energy_eV_to_J(energy_eV)) == pytest.approx(energy_eV)

    def test_zero_maps_to_zero_in_both_directions(self) -> None:
        assert energy_eV_to_J(0.0) == 0.0
        assert energy_J_to_eV(0.0) == 0.0
