"""Cone-beam geometry of an ESRF ID16A near-field ptychography scan.

ID16A records the geometry in the equivalent parallel-beam frame: a detector pixel
pitch, the object pixel pitch it projects to, and the propagation distance of the
parallel beam that images the same way. Ptychodus states the same geometry in the lab
frame, as an object-to-detector distance and a signed focus position. The two agree
exactly, so this module converts rather than estimating.

Writing ``M`` for the magnification, ``z_eff`` for the recorded equivalent distance
and ``z_d`` for the lab-frame object-to-detector distance::

    M     = detector_pixel / object_pixel
    z_d   = M * z_eff
    z_f   = -z_eff * M / (M - 1)

``z_f`` is negative because the focus sits upstream of the sample, which is the sign
convention the magnification is computed under. Round-tripping ``z_d / M`` recovers
the recorded ``z_eff``, and ``detector_pixel / M`` recovers the recorded object pixel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
import logging

from ptychodus.api.constants import EnergyUnit, LengthUnit

from ._scan import parse_motor

logger = logging.getLogger(__name__)

_OPTIC_USED: Final[str] = 'optic_used'
_PIXEL_SIZE: Final[str] = 'pixel_size'
_PROPAGATION_DISTANCE: Final[str] = 'propagation_distance'
_ENERGY: Final[str] = 'energy'


@dataclass(frozen=True)
class ESRFID16AGeometry:
    """Lab-frame geometry derived from an ID16A detector header."""

    detector_pixel_size_m: float
    object_pixel_size_m: float
    magnification: float
    detector_distance_m: float
    focus_object_distance_m: float
    photon_energy_eV: float  # noqa: N815


def _header_float(header: Mapping[str, str], key: str) -> float:
    try:
        return float(header[key])
    except KeyError:
        raise ValueError(f'The detector header has no "{key}" entry.') from None
    except ValueError:
        raise ValueError(f'The "{key}" header entry is not numeric: "{header[key]}".') from None


def compute_geometry(
    header: Mapping[str, str],
    *,
    sample_x0_mm: float | None = None,
    cross_check_rtol: float = 1.0e-3,
) -> ESRFID16AGeometry:
    """Convert an ID16A detector header into lab-frame geometry.

    `sample_x0_mm` is the master's reference sample position. When it is supplied, the
    focus-to-object distance it implies together with the header's own ``sx`` motor is
    an independent measurement of the same quantity, and a disagreement beyond
    `cross_check_rtol` is logged. It is logged rather than raised because the header
    key naming the detector pitch is not self-describing: a cross-check that fires
    says the geometry deserves a look, not that the file is unreadable.

    Raises:
        ValueError: if a required header entry is missing or non-numeric, or if the
            implied magnification is degenerate.
    """
    micrometer = LengthUnit.MICROMETER.meters_per_unit
    millimeter = LengthUnit.MILLIMETER.meters_per_unit

    detector_pixel_size_m = _header_float(header, _OPTIC_USED) * micrometer
    object_pixel_size_m = _header_float(header, _PIXEL_SIZE) * micrometer
    equivalent_distance_m = _header_float(header, _PROPAGATION_DISTANCE) * millimeter
    photon_energy_eV = (  # noqa: N806
        _header_float(header, _ENERGY) * EnergyUnit.KILOELECTRONVOLT.electronvolts_per_unit
    )

    if object_pixel_size_m <= 0.0:
        raise ValueError(f'The "{_PIXEL_SIZE}" header entry must be positive.')

    magnification = detector_pixel_size_m / object_pixel_size_m

    # A magnification of one is a parallel beam, where no focus exists to place and
    # the conversion below divides by zero.
    if magnification <= 1.0:
        raise ValueError(
            f'Detector pixel {detector_pixel_size_m:.3e} m over object pixel '
            f'{object_pixel_size_m:.3e} m gives a magnification of {magnification:.6g}, '
            'which is not a cone beam.'
        )

    detector_distance_m = magnification * equivalent_distance_m
    focus_object_distance_m = -equivalent_distance_m * magnification / (magnification - 1.0)

    _log_cross_check(
        header,
        focus_object_distance_m=focus_object_distance_m,
        sample_x0_mm=sample_x0_mm,
        rtol=cross_check_rtol,
    )

    return ESRFID16AGeometry(
        detector_pixel_size_m=detector_pixel_size_m,
        object_pixel_size_m=object_pixel_size_m,
        magnification=magnification,
        detector_distance_m=detector_distance_m,
        focus_object_distance_m=focus_object_distance_m,
        photon_energy_eV=photon_energy_eV,
    )


def _log_cross_check(
    header: Mapping[str, str],
    *,
    focus_object_distance_m: float,
    sample_x0_mm: float | None,
    rtol: float,
) -> None:
    """Compare the derived focus distance against the sample stage's own reading."""
    if sample_x0_mm is None:
        return

    # sx is recorded in millimeters, unlike the micrometer scan motors beside it.
    sample_x_mm = parse_motor(header, 'sx')

    if sample_x_mm is None:
        return

    measured_m = (sample_x_mm - sample_x0_mm) * LengthUnit.MILLIMETER.meters_per_unit
    derived_m = -focus_object_distance_m

    if measured_m <= 0.0:
        return

    if abs(measured_m - derived_m) > rtol * abs(measured_m):
        logger.warning(
            'Focus-object distance from the detector header (%.6g m) disagrees with the '
            'sample stage reading sx - sx0 (%.6g m) by more than %.3g relative. Check the '
            'detector pixel size the header records.',
            derived_m,
            measured_m,
            rtol,
        )
    else:
        logger.debug(
            'Focus-object distance %.6g m agrees with sx - sx0 (%.6g m).', derived_m, measured_m
        )
