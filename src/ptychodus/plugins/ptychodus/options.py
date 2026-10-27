"""What Kickstart accepts, declared for ptychodus to generate settings from.

Kickstart is ptychodus's own, so the declaration carries its bounds, choices and display
text inline through ``pydantic.Field``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'KickstartOptions',
    'default_options',
]


class KickstartOptions(BaseModel):
    """Everything ptychodus exposes for Kickstart, in the shape its panel uses."""

    model_config = ConfigDict(validate_assignment=True)

    defocus_distance_m: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        title='Defocus Distance:',
        description='How far past its focus the object sits, positive downstream. A '
        'product naming a focus position overrides it; zero leaves it to the parallax.',
        json_schema_extra={'display_unit': 'MICROMETER'},
    )
    parallax_bin_size_px: int = Field(
        4,
        ge=1,
        le=64,
        title='Parallax Bin Size [px]:',
        description='Detector pixels per side of a bright-field bin. Every pattern is kept '
        'binned, so larger bins cost less memory but leave fewer bins in the bright field.',
    )
    parallax_min_significance: float = Field(
        3.0,
        ge=0.0,
        le=100.0,
        title='Parallax Minimum Significance:',
        description='Standard errors a parallax defocus must reach to be used.',
    )
    parallax_min_defocus_dof: float = Field(
        1.0,
        ge=0.5,
        le=10.0,
        title='Parallax Minimum Defocus [DOF]:',
        description='Depths of focus (lambda / NA**2) a parallax defocus must reach to be '
        'used; nearer focus the bright-field images are not shifted copies of one another. '
        'Equivalently, the minimum Fresnel number of the probe shadow.',
    )
    parallax_rtol: float = Field(
        0.5,
        ge=0.0,
        le=10.0,
        title='Parallax Relative Tolerance:',
        description='Fraction of a stated defocus a resolved parallax may differ by before '
        'a warning.',
    )
    incident_photon_count_quantile: float = Field(
        0.99,
        ge=0.5,
        le=1.0,
        title='Incident Photon Count Quantile:',
        description='Quantile of the per-pattern totals taken as the unattenuated '
        'incident photon count, which the probe is scaled to and the object referenced to.',
    )
    probe_bright_field_mad_threshold: float = Field(
        4.5,
        ge=0.0,
        le=100.0,
        title='Probe Bright Field Threshold [MAD]:',
        description='Keeps only the bright field of the mean pattern the probe is estimated '
        'from: pixels within this many median absolute deviations of the background are '
        'zeroed before back-propagating.',
    )
    update_count: int = Field(
        10,
        ge=1,
        le=1000,
        title='Update Count:',
        description='How many times the estimate is shown while the patterns are '
        'accumulated, the last after all of them; intermediate updates skip the parallax. '
        'One shows only the final estimate.',
    )
    phase_regularization_fov_fraction: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        title='Phase Regularization [FOV]:',
        description='Damps object phase variations longer than this fraction of the field '
        'of view, the drift integrating gradient noise piles up; zero disables it.',
    )


def default_options() -> KickstartOptions:
    return KickstartOptions()
