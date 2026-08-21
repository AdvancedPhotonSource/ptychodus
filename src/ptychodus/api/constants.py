"""Mathematical and physical constants and unit enums used throughout the API."""

from __future__ import annotations
from enum import Enum
from typing import Final

import numpy

# Mathematical Constants
TWO_PI: Final[float] = 2.0 * numpy.pi
TWO_PI_J: Final[complex] = 2.0j * numpy.pi

# Physical Constants
# Source: https://physics.nist.gov/cuu/Constants/index.html
ELECTRON_VOLT_J: Final[float] = 1.602176634e-19
LIGHT_SPEED_M_PER_S: Final[float] = 299792458
PLANCK_CONSTANT_J_PER_HZ: Final[float] = 6.62607015e-34

# Unit Conversion Factors
# Naming: ONE_<UNIT>_<BASE> means "1 <unit> expressed in <base SI unit>".
# Multiply a value expressed in <unit> by the constant to get the base-SI value.
# All physical quantities in api/ are stored in base SI units (m, s, J, rad, Hz)
# with an explicit unit suffix on the identifier (_m, _s, _J, _rad, _Hz, ...);
# eV, deg, and px are the only non-SI suffixes allowed by convention.
ONE_MILLIMETER_M: Final[float] = 1e-3
ONE_MICRON_M: Final[float] = 1e-6
ONE_NANOMETER_M: Final[float] = 1e-9
ONE_ANGSTROM_M: Final[float] = 1e-10
ONE_KILOELECTRONVOLT_EV: Final[float] = 1000.0
HC_EV_ANGSTROM: Final[float] = (
    PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S / ELECTRON_VOLT_J / ONE_ANGSTROM_M
)


class ByteUnit(Enum):
    """Decimal (SI) byte units, ordered smallest to largest.

    The member value pairs the number of bytes in one unit with its display suffix.
    """

    B = (1, 'B')
    KB = (1000, 'kB')
    MB = (1000**2, 'MB')
    GB = (1000**3, 'GB')
    TB = (1000**4, 'TB')
    PB = (1000**5, 'PB')

    def __init__(self, bytes_per_unit: int, label: str) -> None:
        self.bytes_per_unit = bytes_per_unit
        self.label = label

    @classmethod
    def from_bytes(cls, nbytes: int) -> ByteUnit:
        """Largest unit that leaves a value of at least one, clamped to the ends of the ladder."""
        unit = cls.B

        for candidate in cls:
            if nbytes < candidate.bytes_per_unit:
                break

            unit = candidate

        return unit

    def convert(self, nbytes: int) -> float:
        """Express a byte count in this unit."""
        return nbytes / self.bytes_per_unit

    def format(self, nbytes: int) -> str:
        """Render a byte count in this unit, e.g. "12.34 MB"."""
        if self is ByteUnit.B:
            return f'{nbytes} {self.label}'

        return f'{self.convert(nbytes):.2f} {self.label}'


def format_bytes(nbytes: int) -> str:
    """Render a byte count in the largest unit that keeps it at or above one, e.g. "4.10 GB"."""
    return ByteUnit.from_bytes(nbytes).format(nbytes)
