"""Mathematical and physical constants used throughout the API."""

from typing import Final

import numpy

# Mathematical Constants
BYTES_PER_MEGABYTE: Final[int] = 1000 * 1000
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
