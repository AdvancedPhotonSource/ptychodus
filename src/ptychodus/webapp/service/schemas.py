"""Pydantic read models of the indexed resources and of the reports about them."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ptychodus.api.analyze import (
    ObjectComparisonScalars,
    ObjectMetricsScalars,
    ProbeComparisonScalars,
    ProbeMetricsScalars,
    ProbePositionsComparisonScalars,
    ProbePositionsMetricsScalars,
)
from ptychodus.api.diffraction import Polarization
from ptychodus.api.processing import RunState
from ptychodus.api.scattering.geometry import ScatteringRegime
from ptychodus.webapp.db.base import IngestState


_TILT_ANGLE_DESCRIPTION = 'The sample tilt APS 4-ID POLAR records for an XMCD projection, which sets the magnetization component the contrast is sensitive to. Stored but read by no computation, and unrelated to the scattering geometry.'


class _RowBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    folder_path: str
    manifest_mtime: datetime | None
    created_from_manifest_at: datetime | None
    ingest_state: IngestState
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class RunRead(BaseModel):
    """How a processed resource came to be, from its manifest's run record."""

    state: RunState
    reason: str | None
    progress: int
    progress_goal: int = Field(description='The progress a finished run reaches; 0 if unknown.')
    command: str
    updated_at: datetime


class CampaignRead(_RowBase):
    label: str
    comments: str
    sample_name: str
    sample_description: str
    tags: list[str]


class DerivedFromEdge(BaseModel):
    kind: Literal['diffraction', 'product', 'fluorescence']
    uuid: UUID
    # The target's display name -- a diffraction dataset's label, a product's name --
    # so a client can show what the edge points at without a second request. Empty
    # when the target is not indexed.
    label: str = ''


class DiffractionRead(_RowBase):
    label: str
    comments: str
    campaign_uuid: UUID | None
    derived_from: list[DerivedFromEdge] = []
    run: RunRead | None = None

    detector_distance_m: float | None
    photon_energy_eV: float | None  # noqa: N815
    probe_photon_count: int | None
    exposure_time_s: float | None
    tomography_angle_deg: float | None
    tilt_angle_deg: float | None = Field(default=None, description=_TILT_ANGLE_DESCRIPTION)
    polarization: Polarization | None = None
    crop_center_x_px: int | None
    crop_center_y_px: int | None

    # The assembled patterns: diffraction.h5 records the stack after cropping and
    # binning, so the extent and pixel pitch below are the processed ones. The raw
    # detector's are not persisted.
    pattern_dtype: str | None
    pattern_height_px: int | None
    pattern_width_px: int | None
    num_patterns_total: int | None
    detector_pixel_width_m: float | None
    detector_pixel_height_m: float | None
    num_bad_pixels: int | None = None
    nbytes: int | None = None


class ProductRead(_RowBase):
    derived_from: list[DerivedFromEdge] = []
    run: RunRead | None = None

    name: str | None
    comments: str | None
    detector_distance_m: float | None
    focus_object_distance_m: float | None
    far_field: bool | None
    photon_energy_eV: float | None  # noqa: N815
    probe_photon_count: float | None
    exposure_time_s: float | None
    i0_photons_per_count: float | None = None
    mass_attenuation_m2_per_kg: float | None
    tomography_angle_deg: float | None
    tilt_angle_deg: float | None = Field(default=None, description=_TILT_ANGLE_DESCRIPTION)
    polarization: Polarization | None = None
    two_theta_deg: float | None = None
    detector_gamma_deg: float | None = None
    incidence_angle_deg: float | None = None
    scattering_regime: ScatteringRegime | None = Field(
        default=None,
        description='Transmission, Bragg or Laue, derived from the scattering angles. Off '
        'axis, the positions, probe and object are in the exit frame of that geometry.',
    )

    object_layers: int | None
    object_height_px: int | None
    object_width_px: int | None
    object_pixel_width_m: float | None
    object_pixel_height_m: float | None
    probe_modes: int | None
    probe_height_px: int | None
    probe_width_px: int | None
    probe_pixel_width_m: float | None = None
    probe_pixel_height_m: float | None = None
    probe_coherence: float | None = None
    num_probe_positions: int | None
    num_loss_epochs: int | None
    probe_path_length_m: float | None = None
    probe_dtype: str | None = None
    object_dtype: str | None = None
    probe_nbytes: int | None = None
    object_nbytes: int | None = None
    probe_positions_nbytes: int | None = None
    probe_mode_relative_power: list[float] = []
    object_layer_spacing_m: list[float] = []

    # Sum of the layer spacings and the photon energy's wavelength, derived on read
    # like the block below.
    object_total_thickness_m: float | None = None
    photon_wavelength_m: float | None = None

    # Derived on read from the columns above; never stored, so the formula has one
    # home in ptychodus.api.propagate. The last five additionally need the detector,
    # which product.h5 does not record -- they are null unless the manifest's
    # derived_from names a diffraction resource the webapp has indexed.
    probe_wavenumber_per_m: float | None = None
    probe_angular_wavenumber_rad_per_m: float | None = None
    probe_photon_flux_per_s: float | None = None
    probe_power_W: float | None = None  # noqa: N815
    object_plane_propagation_distance_m: float | None = None
    magnification: float | None = None
    object_plane_pixel_width_m: float | None = None
    object_plane_pixel_height_m: float | None = None
    fresnel_number: float | None = None
    detector_numerical_aperture: float | None = None
    depth_of_field_m: float | None = None


class FluorescenceRead(_RowBase):
    label: str
    comments: str
    derived_from: list[DerivedFromEdge] = []
    run: RunRead | None = None

    element_names: list[str]
    element_counts: list[float] = []
    element_nbytes: list[int] = []
    map_height_px: int | None
    map_width_px: int | None
    nbytes: int | None = None


ItemT = TypeVar('ItemT', bound=BaseModel)


class Page(BaseModel, Generic[ItemT]):
    items: list[ItemT]
    total: int
    limit: int
    offset: int


class ResourceRef(BaseModel):
    kind: Literal['campaign', 'diffraction', 'product', 'fluorescence']
    uuid: UUID


# The four resource kinds, as the literal type the API exposes. Shared so that a
# helper resolving a kind at run time can hand back something assignable here
# rather than a bare str.
ResourceKindLiteral = Literal['campaign', 'diffraction', 'product', 'fluorescence']


class LineageNode(BaseModel):
    kind: ResourceKindLiteral
    uuid: UUID
    label: str = ''


class LineageRead(BaseModel):
    node: LineageNode
    ancestors: list[LineageNode]
    descendants: list[LineageNode]
    campaign: CampaignRead | None


class ResourceStats(BaseModel):
    campaign_count: int
    diffraction_count: int
    product_count: int
    fluorescence_count: int
    invalid_count: int


class HealthRead(BaseModel):
    status: Literal['ok', 'degraded']
    db: Literal['ok', 'down']
    watcher: Literal['alive', 'dead', 'disabled']


class ReindexResponse(BaseModel):
    status: Literal['accepted']
    job_id: str


_UNDEFINED_AS_NULL = (
    'Statistics a report cannot define, such as I0 on a product without it, are null.'
)


class _MetricsRead(BaseModel):
    uuid: UUID
    run: RunRead | None = Field(description='The run record of the product analyzed.')


class ProbeMetricsRead(_MetricsRead):
    """A product's probe: its size, entropy, mode spectrum and scan overlap."""

    scalars: ProbeMetricsScalars = Field(description=_UNDEFINED_AS_NULL)


class ObjectMetricsRead(_MetricsRead):
    """A product's object over the region its scan illuminated, ambiguities removed."""

    scalars: ObjectMetricsScalars = Field(description=_UNDEFINED_AS_NULL)


class ProbePositionsMetricsRead(_MetricsRead):
    """A product's probe positions: their extent, spacing, scan indexes and I0."""

    scalars: ProbePositionsMetricsScalars = Field(description=_UNDEFINED_AS_NULL)


class _ComparisonRead(_MetricsRead):
    reference_uuid: UUID
    reference_run: RunRead | None = Field(description='The run record of the reference.')


class ProbeComparisonRead(_ComparisonRead):
    """A product's probe compared with a reference product's."""

    scalars: ProbeComparisonScalars = Field(description=_UNDEFINED_AS_NULL)


class ObjectComparisonRead(_ComparisonRead):
    """A product's projected object compared with a reference product's."""

    scalars: ObjectComparisonScalars = Field(description=_UNDEFINED_AS_NULL)


class ProbePositionsComparisonRead(_ComparisonRead):
    """How a product's probe positions are displaced from a reference product's."""

    scalars: ProbePositionsComparisonScalars = Field(description=_UNDEFINED_AS_NULL)


SpectralSNRConventionName = Literal['half_split', 'repeat']
"""A :class:`~ptychodus.api.metrics.SpectralSNRConvention` member, by its lowercase name."""

# What each analysis parameter means, stated once for every transport.
REFERENCE_UUID_DESCRIPTION = 'The reference product the product is compared with.'
EROSION_DIVISOR_DESCRIPTION = (
    'The illuminated region is eroded by the smaller probe extent divided by this before scoring.'
)
RESAMPLE_TEST_DESCRIPTION = (
    "Resample the product onto the reference's grids where they differ; otherwise a mismatch "
    'is refused.'
)
MAD_THRESHOLD_DESCRIPTION = (
    'Significance threshold of the phase support, in median absolute deviations above the '
    'background.'
)
CONVENTION_DESCRIPTION = (
    'half_split for reconstructions from complementary halves of one dataset, repeat for '
    'independent repeat scans.'
)
SNR_THRESHOLD_DESCRIPTION = 'The spectral SNR at which the resolution is read.'


def _widen_numbers(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_widen_numbers(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    widened = {key: _widen_numbers(value) for key, value in schema.items()}

    if widened.get('type') == 'number':
        del widened['type']
        widened['anyOf'] = [{'type': 'number'}, {'type': 'null'}]

    return widened


def nullable_numbers_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema of `model` with every number also admitting null.

    JSON has no NaN or infinity, so a float field holding one serializes as null, which a
    client validating against the plain schema would refuse.
    """
    return _widen_numbers(model.model_json_schema())
