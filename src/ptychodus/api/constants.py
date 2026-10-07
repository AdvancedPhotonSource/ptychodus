"""Mathematical and physical constants and unit enums used throughout the API."""

from __future__ import annotations
import math
from decimal import Decimal
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

# Unit Enums
# All physical quantities in api/ are stored in base SI units (m, s, J, rad, ...) with an
# explicit unit suffix on the identifier (_m, _s, _J, _rad, ...); eV, deg, and px are
# the only non-SI suffixes allowed by convention. A compound unit reads
# <numerator>_per_<denominator>, with multiplied units joined by _ within either part and
# the numerator omitted when it is dimensionless: m2_per_kg, J_per_m2, rad_per_m, per_s,
# per_s_m2. A bare _ never means division, and a rate is never spelled _Hz -- so that one
# quantity carries one name wherever it is derived, mapped or measured. The enums below
# convert to and from those base units at the boundaries -- file readers on the way in,
# the GUI on the way out.


class LengthUnit(Enum):
    """Length units, ordered smallest to largest.

    The member value pairs the unit's power of ten relative to the meter with its display
    label. The exponent, not a float, is the source of truth: both scale factors --
    ``meters_per_unit`` for ordinary float arithmetic and ``decimal_meters_per_unit`` for the
    exact decimal arithmetic the GUI needs -- are derived from it, so neither can drift.
    """

    PICOMETER = (-12, 'pm')
    ANGSTROM = (-10, 'Å')
    NANOMETER = (-9, 'nm')
    MICROMETER = (-6, 'µm')
    MILLIMETER = (-3, 'mm')
    METER = (0, 'm')

    def __init__(self, power_of_ten: int, label: str) -> None:
        self.power_of_ten = power_of_ten
        self.label = label
        self.meters_per_unit: float = float(f'1e{power_of_ten}')
        # The string constructor is exact and, unlike Decimal.scaleb, independent of the
        # active decimal context.
        self.decimal_meters_per_unit: Decimal = Decimal(f'1e{power_of_ten}')

    @property
    def is_si_prefixed(self) -> bool:
        """Whether this unit is the meter carrying an SI prefix, i.e. a whole 1000-step.

        The angstrom is the only member that fails this test, which is why it does not
        participate in :meth:`from_meters`.
        """
        return self.power_of_ten % 3 == 0

    @classmethod
    def from_meters(cls, length_m: float) -> LengthUnit:
        """Largest SI-prefixed unit that leaves a magnitude of at least one.

        Ignores sign, clamps to the ends of the ladder, and maps zero to meters. The angstrom
        sits one decade below the nanometer rather than a full 1000-step, so admitting it
        would make neighboring values in a single column flip between Å, nm, and pm; select it
        explicitly when it is wanted. Non-finite lengths fall through to meters.
        """
        if length_m == 0.0:
            return cls.METER

        magnitude_m = abs(length_m)
        unit = cls.PICOMETER

        for candidate in cls:
            if not candidate.is_si_prefixed:
                continue

            if magnitude_m < candidate.meters_per_unit:
                break

            unit = candidate

        return unit

    def to_meters(self, length: float) -> float:
        """Express a length given in this unit as meters."""
        return length * self.meters_per_unit

    def convert(self, length_m: float) -> float:
        """Express a length in meters in this unit."""
        return length_m / self.meters_per_unit

    def format(self, length_m: float) -> str:
        """Render a length in meters in this unit, e.g. "1.234 nm"."""
        return f'{self.convert(length_m):.4g} {self.label}'


class AngleUnit(Enum):
    """Angle units, with the turn as the canonical value.

    The member value pairs the number of these units in one full turn with its display label,
    so ``units_per_turn`` is the inverse orientation of :attr:`LengthUnit.meters_per_unit`.
    That is deliberate: the reciprocals 1/360 and 1/(2*pi) are non-terminating, so keeping the
    exact 360 and dividing loses nothing. Unlike lengths and bytes these units are a
    presentation choice rather than a magnitude ladder, so there is no auto-selecting
    alternate constructor and no formatting facade.
    """

    TURN = (1.0, 'turn')
    DEGREE = (360.0, 'deg')
    RADIAN = (TWO_PI, 'rad')

    def __init__(self, units_per_turn: float, label: str) -> None:
        self.units_per_turn = units_per_turn
        self.label = label
        # Decimal.from_float is exact for every float, including the irrational 2*pi, and is
        # not rounded by the active decimal context the way 2 * Decimal(pi) would be.
        self.decimal_units_per_turn: Decimal = Decimal.from_float(units_per_turn)

    def to_turns(self, angle: float) -> float:
        """Express an angle given in this unit as turns."""
        return angle / self.units_per_turn

    def convert(self, angle_turns: float) -> float:
        """Express an angle in turns in this unit."""
        return angle_turns * self.units_per_turn


class EnergyUnit(Enum):
    """Photon energy units, ordered smallest to largest.

    The member value pairs the unit's power of ten relative to the electron-volt with its
    display label. The exponent, not the multiplier, is the source of truth:
    ``electronvolts_per_unit`` is derived from it so the two cannot drift, matching the shape
    of :class:`LengthUnit` and :class:`ByteUnit`.

    The joule is deliberately not a member. It is not a power of ten of an electron-volt, so
    admitting it would break that invariant; convert across that boundary with
    :func:`energy_eV_to_J` and :func:`energy_J_to_eV` instead.
    """

    ELECTRONVOLT = (0, 'eV')
    KILOELECTRONVOLT = (3, 'keV')
    MEGAELECTRONVOLT = (6, 'MeV')

    def __init__(self, power_of_ten: int, label: str) -> None:
        self.power_of_ten = power_of_ten
        self.label = label
        self.electronvolts_per_unit: float = float(f'1e{power_of_ten}')

    @classmethod
    def from_electronvolts(cls, energy_eV: float) -> EnergyUnit:  # noqa: N803
        """Largest unit that leaves a magnitude of at least one.

        Ignores sign and clamps to the ends of the ladder. Zero and non-finite energies map
        to electron-volts. Unlike :meth:`LengthUnit.from_meters`, which lets non-finite
        values fall through to the meter because the meter happens to sit at the top of its
        ladder, the guard here has to be explicit: the top of this ladder is the megaelectron-
        volt, so an unguarded walk would report ``nan`` as MeV.
        """
        if energy_eV == 0.0 or not math.isfinite(energy_eV):
            return cls.ELECTRONVOLT

        magnitude_eV = abs(energy_eV)  # noqa: N806
        unit = cls.ELECTRONVOLT

        for candidate in cls:
            if magnitude_eV < candidate.electronvolts_per_unit:
                break

            unit = candidate

        return unit

    def to_electronvolts(self, energy: float) -> float:
        """Express an energy given in this unit as electron-volts."""
        return energy * self.electronvolts_per_unit

    def convert(self, energy_eV: float) -> float:  # noqa: N803
        """Express an energy in electron-volts in this unit."""
        return energy_eV / self.electronvolts_per_unit

    def format(self, energy_eV: float) -> str:  # noqa: N803
        """Render an energy in electron-volts in this unit, e.g. "8.047 keV"."""
        return f'{self.convert(energy_eV):.4g} {self.label}'


# Derived Constants
HC_EV_ANGSTROM: Final[float] = LengthUnit.ANGSTROM.convert(
    PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S / ELECTRON_VOLT_J
)


def energy_eV_to_wavelength_m(energy_eV: float) -> float:  # noqa: N802, N803
    """Photon wavelength (m) for a photon energy in electron-volts; 0.0 at zero energy."""
    hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S  # noqa: N806

    try:
        return hc_Jm / (energy_eV * ELECTRON_VOLT_J)
    except ZeroDivisionError:
        return 0.0


def wavelength_m_to_energy_eV(wavelength_m: float) -> float:  # noqa: N802
    """Photon energy (eV) for a wavelength in meters; 0.0 at zero wavelength."""
    hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S  # noqa: N806

    try:
        return hc_Jm / (wavelength_m * ELECTRON_VOLT_J)
    except ZeroDivisionError:
        return 0.0


def energy_eV_to_J(energy_eV: float) -> float:  # noqa: N802, N803
    """Photon energy in joules for a photon energy in electron-volts."""
    return energy_eV * ELECTRON_VOLT_J


def energy_J_to_eV(energy_J: float) -> float:  # noqa: N802, N803
    """Photon energy in electron-volts for a photon energy in joules."""
    return energy_J / ELECTRON_VOLT_J


def format_length(length_m: float) -> str:
    """Render a length in the largest SI-prefixed unit keeping it at or above one."""
    return LengthUnit.from_meters(length_m).format(length_m)


class ByteUnit(Enum):
    """Decimal (SI) byte units, ordered smallest to largest.

    The member value pairs the unit's power of ten relative to the byte with its display
    suffix. The exponent, not the multiplier, is the source of truth: ``bytes_per_unit`` is
    derived from it so the two cannot drift, matching the shape of :class:`LengthUnit`.
    """

    B = (0, 'B')
    KB = (3, 'kB')
    MB = (6, 'MB')
    GB = (9, 'GB')
    TB = (12, 'TB')
    PB = (15, 'PB')

    def __init__(self, power_of_ten: int, label: str) -> None:
        self.power_of_ten = power_of_ten
        self.label = label
        self.bytes_per_unit: int = 10**power_of_ten

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
