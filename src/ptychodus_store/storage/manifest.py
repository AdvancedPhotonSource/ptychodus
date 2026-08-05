from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ptychodus.api.diffraction import Polarization

MANIFEST_FILENAME = 'manifest.json'


class ResourceKind(StrEnum):
    CAMPAIGN = 'campaign'
    DIFFRACTION = 'diffraction'
    PRODUCT = 'product'
    FLUORESCENCE = 'fluorescence'


# Kinds that may appear as a derived_from target. Campaign is excluded by design
# (campaign is context, not derivation).
DERIVATION_TARGET_KINDS: frozenset[str] = frozenset(
    {ResourceKind.DIFFRACTION, ResourceKind.PRODUCT, ResourceKind.FLUORESCENCE}
)

# HDF5-owned keys per kind: a manifest MUST NOT carry these (single-source rule).
# Values listed here are read from the companion HDF5 file at reconciliation time.
HDF5_OWNED_KEYS: dict[str, frozenset[str]] = {
    ResourceKind.CAMPAIGN: frozenset(),
    ResourceKind.DIFFRACTION: frozenset(
        {
            'pattern_dtype',
            'pattern_shape',
            'num_patterns_total',
            'detector_pixel_width_m',
            'detector_pixel_height_m',
        }
    ),
    ResourceKind.PRODUCT: frozenset(
        {
            'name',
            'comments',
            'detector_distance_m',
            'probe_energy_eV',
            'probe_photon_count',
            'exposure_time_s',
            'mass_attenuation_m2_kg',
            'tomography_angle_deg',
            'tilt_angle_deg',
            'polarization',
            'object_shape',
            'object_pixel_width_m',
            'object_pixel_height_m',
            'probe_shape',
            'num_scan_points',
            'num_loss_epochs',
        }
    ),
    ResourceKind.FLUORESCENCE: frozenset({'element_names', 'map_shape'}),
}


class DerivedFromRef(BaseModel):
    """A typed pointer to another resource that this node was derived from."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['diffraction', 'product', 'fluorescence']
    uuid: UUID


class _ManifestBase(BaseModel):
    """Fields common to every kind."""

    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1] = 1
    uuid: UUID
    created_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _reject_hdf5_owned_keys_in_extra(self) -> _ManifestBase:
        owned = HDF5_OWNED_KEYS.get(self.kind, frozenset())  # type: ignore[attr-defined]
        clashes = sorted(owned & set(self.extra.keys()))
        if clashes:
            raise ValueError(
                f'manifest extra carries HDF5-owned key(s) {clashes}; '
                'these values must live in the HDF5 file only (single source of truth).'
            )
        return self


class CampaignManifest(_ManifestBase):
    kind: Literal['campaign'] = 'campaign'
    label: str = ''
    comments: str = ''
    sample_name: str = ''
    sample_description: str = ''
    tags: list[str] = Field(default_factory=list)


class DiffractionManifest(_ManifestBase):
    kind: Literal['diffraction'] = 'diffraction'
    label: str = ''
    comments: str = ''
    campaign_uuid: UUID | None = None
    derived_from: list[DerivedFromRef] = Field(default_factory=list)
    detector_distance_m: float | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    probe_photon_count: int | None = None
    exposure_time_s: float | None = None
    tomography_angle_deg: float | None = None
    tilt_angle_deg: float | None = None
    polarization: Polarization | None = None
    crop_center_x_px: int | None = None
    crop_center_y_px: int | None = None
    files: dict[str, str] = Field(default_factory=lambda: {'diffraction': 'diffraction.h5'})

    @field_validator('derived_from')
    @classmethod
    def _no_self_ref(cls, v: list[DerivedFromRef]) -> list[DerivedFromRef]:
        return v


class ProductManifest(_ManifestBase):
    kind: Literal['product'] = 'product'
    derived_from: list[DerivedFromRef] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=lambda: {'product': 'product.h5'})


class FluorescenceManifest(_ManifestBase):
    kind: Literal['fluorescence'] = 'fluorescence'
    label: str = ''
    comments: str = ''
    derived_from: list[DerivedFromRef] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=lambda: {'fluorescence': 'fluorescence.h5'})


Manifest = Annotated[
    Union[CampaignManifest, DiffractionManifest, ProductManifest, FluorescenceManifest],
    Field(discriminator='kind'),
]


_KIND_TO_MODEL: dict[str, type[_ManifestBase]] = {
    ResourceKind.CAMPAIGN: CampaignManifest,
    ResourceKind.DIFFRACTION: DiffractionManifest,
    ResourceKind.PRODUCT: ProductManifest,
    ResourceKind.FLUORESCENCE: FluorescenceManifest,
}


class ManifestLoadError(Exception):
    """Raised when a manifest fails to parse, validate, or pass cross-checks."""


def load_manifest(path: Path, *, expected_kind: str, expected_uuid: UUID) -> _ManifestBase:
    """Read and validate a manifest.json from `path` against the expected kind and folder UUID.

    The manifest's `kind` must match `expected_kind` (derived from the folder layout),
    its `uuid` must equal `expected_uuid` (the folder name), and `derived_from` entries
    must not self-reference.
    """
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestLoadError(f'failed to read or parse {path}: {exc}') from exc

    declared_kind = raw.get('kind')
    if declared_kind != expected_kind:
        raise ManifestLoadError(
            f'{path}: kind={declared_kind!r} does not match folder kind {expected_kind!r}'
        )

    model_cls = _KIND_TO_MODEL[expected_kind]
    try:
        manifest = model_cls.model_validate(raw)
    except ValidationError as exc:
        raise ManifestLoadError(f'{path}: validation error: {exc}') from exc

    if manifest.uuid != expected_uuid:
        raise ManifestLoadError(
            f'{path}: uuid={manifest.uuid} does not match folder name {expected_uuid}'
        )

    derived = getattr(manifest, 'derived_from', None)
    if derived:
        for ref in derived:
            if ref.uuid == manifest.uuid:
                raise ManifestLoadError(
                    f'{path}: derived_from entry self-references uuid {ref.uuid}'
                )

    return manifest
