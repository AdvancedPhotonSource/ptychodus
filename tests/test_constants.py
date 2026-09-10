"""Unit tests for ptychodus.api.constants free functions."""

from __future__ import annotations

import numpy
import numpy.testing

from ptychodus.api.constants import (
    HC_EV_ANGSTROM,
    energy_eV_to_wavelength_m,
    wavelength_m_to_energy_eV,
)


# HC_EV_ANGSTROM = hc / eV expressed in eV*Angstrom; converting to eV*m gives the
# invariant that E * lambda_m must equal for every non-degenerate (E, lambda) pair.
_HC_EV_M = HC_EV_ANGSTROM * 1.0e-10


class TestEnergyToWavelength:
    def test_zero_energy_returns_zero(self) -> None:
        assert energy_eV_to_wavelength_m(0.0) == 0.0

    def test_positive_energy_matches_planck_relation(self) -> None:
        for energy_eV in (100.0, 1_000.0, 8_047.0, 100_000.0):  # noqa: N806
            wavelength_m = energy_eV_to_wavelength_m(energy_eV)
            numpy.testing.assert_allclose(energy_eV * wavelength_m, _HC_EV_M, rtol=1.0e-12)


class TestWavelengthToEnergy:
    def test_zero_wavelength_returns_zero(self) -> None:
        assert wavelength_m_to_energy_eV(0.0) == 0.0

    def test_round_trip_recovers_input_energy(self) -> None:
        for energy_eV in (100.0, 1_000.0, 8_047.0, 100_000.0):  # noqa: N806
            wavelength_m = energy_eV_to_wavelength_m(energy_eV)
            numpy.testing.assert_allclose(
                wavelength_m_to_energy_eV(wavelength_m), energy_eV, rtol=1.0e-12
            )
