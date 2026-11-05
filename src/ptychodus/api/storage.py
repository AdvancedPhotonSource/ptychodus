"""The storage root: one folder per resource, named by kind and uuid, with a manifest.

A storage root holds ``<root>/<kind>/<uuid>/`` folders, each with a ``manifest.json``
and the HDF5 file it names. The manifest carries what the file does not -- labels,
campaign, lineage -- and the file stays the single source of everything it holds.
Resources are named by their uuid alone; a uuid4 does not collide across kinds, so
:meth:`StorageLayout.locate` finds any resource from its uuid, and a caller never needs
a path.

The chainable savers are the only way a resource folder is created: each is handed the
data, assembles the resource in ``<root>/.staging/`` and renames it into place in one
step, so a resource appears complete or not at all, and returns its uuid for the next
saver's ``derived_from`` or ``campaign``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Union, cast
from uuid import UUID, uuid4
import json
import shutil

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .assemble import AssembledDiffractionData
from .diffraction import Polarization
from .io import save_diffraction_data, save_product
from .product import Product

__all__ = [
    'CampaignManifest',
    'DERIVATION_TARGET_KINDS',
    'DerivedFromRef',
    'DiffractionManifest',
    'FluorescenceManifest',
    'HDF5_OWNED_KEYS',
    'LayoutError',
    'MANIFEST_FILENAME',
    'Manifest',
    'ManifestLoadError',
    'ProductManifest',
    'ResourceKind',
    'ResourceLocation',
    'STORAGE_ROOT_ENVIRONMENT_VARIABLE',
    'StorageLayout',
    'load_manifest',
    'resolve_storage_inputs',
    'save_campaign_resource',
    'save_diffraction_resource',
    'save_manifest',
    'save_product_resource',
]

MANIFEST_FILENAME: Final[str] = 'manifest.json'

STORAGE_ROOT_ENVIRONMENT_VARIABLE: Final[str] = 'PTYCHODUS_WEBAPP_STORAGE_ROOT'
"""The environment variable naming the storage root, as the web service reads it."""


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
            'num_bad_pixels',
            'nbytes',
        }
    ),
    ResourceKind.PRODUCT: frozenset(
        {
            'name',
            'comments',
            'detector_distance_m',
            'focus_object_distance_m',
            'far_field',
            'photon_energy_eV',
            'probe_photon_count',
            'exposure_time_s',
            'mass_attenuation_m2_per_kg',
            'tomography_angle_deg',
            'tilt_angle_deg',
            'polarization',
            'two_theta_deg',
            'detector_gamma_deg',
            'incidence_angle_deg',
            'object_shape',
            'object_pixel_width_m',
            'object_pixel_height_m',
            'probe_shape',
            'num_scan_points',
            'num_loss_epochs',
            'scan_length_m',
            'probe_dtype',
            'object_dtype',
            'probe_nbytes',
            'object_nbytes',
            'scan_nbytes',
            'probe_mode_relative_power',
            'object_layer_spacing_m',
        }
    ),
    ResourceKind.FLUORESCENCE: frozenset(
        {'element_names', 'map_shape', 'element_counts', 'nbytes'}
    ),
}

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


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
    photon_energy_eV: float | None = None  # noqa: N815
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


def load_manifest(path: Path, *, expected_kind: str, expected_uuid: UUID) -> Manifest:
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

    return cast(Manifest, manifest)


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Write `manifest` to `path` for :func:`load_manifest` to read back."""
    path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_KIND_VALUES: frozenset[str] = frozenset(k.value for k in ResourceKind)


class LayoutError(ValueError):
    """Raised when a path or uuid cannot be mapped to the storage layout."""


@dataclass(frozen=True)
class ResourceLocation:
    """A resolved <storage_root>/<kind>/<uuid>/ folder."""

    kind: str
    uuid: UUID
    folder: Path

    @property
    def manifest_path(self) -> Path:
        return self.folder / MANIFEST_FILENAME


class StorageLayout:
    """Resolves and validates paths within the storage root."""

    STAGING_DIRECTORY: Final[str] = '.staging'
    """Hidden folder, beside the kinds, in which the savers assemble a resource."""

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def kind_dir(self, kind: str) -> Path:
        if kind not in _KIND_VALUES:
            raise LayoutError(f'unknown kind {kind!r}')
        return self._root / kind

    def resource_folder(self, kind: str, uuid: UUID) -> Path:
        return self.kind_dir(kind) / str(uuid)

    def manifest_path(self, kind: str, uuid: UUID) -> Path:
        return self.resource_folder(kind, uuid) / MANIFEST_FILENAME

    def ensure_kind_dirs(self) -> None:
        """Create the four top-level kind directories if they do not exist."""
        for kind in ResourceKind:
            (self._root / kind).mkdir(parents=True, exist_ok=True)

    def locate(self, uuid: UUID) -> ResourceLocation:
        """The resource named `uuid`, whatever its kind.

        Raises:
            LayoutError: if no resource has that uuid.
        """
        for kind in ResourceKind:
            folder = self._root / kind / str(uuid)

            if (folder / MANIFEST_FILENAME).is_file():
                return ResourceLocation(kind=kind, uuid=uuid, folder=folder)

        raise LayoutError(f'No resource {uuid} in the storage root {self._root}.')

    def parse_manifest_path(self, manifest_path: Path) -> ResourceLocation:
        """Map a `manifest.json` path to its (kind, uuid, folder).

        Expected shape: `<storage_root>/<kind>/<uuid>/manifest.json`.
        """
        path = manifest_path.resolve()
        if path.name != MANIFEST_FILENAME:
            raise LayoutError(f'not a manifest file: {manifest_path}')

        folder = path.parent
        kind_dir = folder.parent

        try:
            kind_dir.relative_to(self._root)
        except ValueError as exc:
            raise LayoutError(f'{manifest_path} is outside storage root {self._root}') from exc

        if kind_dir.parent != self._root:
            raise LayoutError(
                f'{manifest_path} is not at the expected depth '
                f'<storage_root>/<kind>/<uuid>/manifest.json'
            )

        kind = kind_dir.name
        if kind not in _KIND_VALUES:
            raise LayoutError(f'{manifest_path} sits under unknown kind {kind!r}')

        try:
            uuid = UUID(folder.name)
        except ValueError as exc:
            raise LayoutError(
                f'{manifest_path} parent folder name {folder.name!r} is not a UUID'
            ) from exc

        return ResourceLocation(kind=kind, uuid=uuid, folder=folder)

    def iter_manifest_paths(self) -> list[Path]:
        """Return all `manifest.json` paths in the storage root, campaigns first."""
        paths: list[Path] = []
        for kind in (
            ResourceKind.CAMPAIGN,
            ResourceKind.DIFFRACTION,
            ResourceKind.PRODUCT,
            ResourceKind.FLUORESCENCE,
        ):
            kind_dir = self._root / kind
            if not kind_dir.is_dir():
                continue
            for child in sorted(kind_dir.iterdir()):
                if not child.is_dir():
                    continue
                m = child / MANIFEST_FILENAME
                if m.is_file():
                    paths.append(m)
        return paths


def _load_located(layout: StorageLayout, uuid: UUID, kind: str) -> tuple[Path, Manifest]:
    location = layout.locate(uuid)

    if location.kind != kind:
        raise LayoutError(f'Resource {uuid} is a {location.kind}, not a {kind}.')

    manifest = load_manifest(location.manifest_path, expected_kind=kind, expected_uuid=uuid)
    return location.folder, manifest


def resolve_storage_inputs(layout: StorageLayout, product_uuid: UUID) -> tuple[Path, Path]:
    """Return the `(diffraction_file, product_file)` a stored product reconstructs from.

    The diffraction file is the one the product manifest's `derived_from` names.

    Raises:
        LayoutError: if `product_uuid` names no product, or the product names no
            single diffraction resource.
        ManifestLoadError: if a manifest fails to load.
    """
    product_folder, product = _load_located(layout, product_uuid, ResourceKind.PRODUCT)

    if not isinstance(product, ProductManifest):
        raise ManifestLoadError(f'Resource {product_uuid} has no product manifest.')

    sources = [ref for ref in product.derived_from if ref.kind == ResourceKind.DIFFRACTION]

    if len(sources) != 1:
        raise LayoutError(
            f'Product {product_uuid} names {len(sources)} diffraction resources; expected one.'
        )

    diffraction_folder, diffraction = _load_located(
        layout, sources[0].uuid, ResourceKind.DIFFRACTION
    )

    if not isinstance(diffraction, DiffractionManifest):
        raise ManifestLoadError(f'Resource {sources[0].uuid} has no diffraction manifest.')

    return (
        diffraction_folder / diffraction.files['diffraction'],
        product_folder / product.files['product'],
    )


# ---------------------------------------------------------------------------
# Chainable savers
# ---------------------------------------------------------------------------


def _derived_from(layout: StorageLayout, uuids: Sequence[UUID]) -> list[DerivedFromRef]:
    references: list[DerivedFromRef] = []

    for uuid in uuids:
        location = layout.locate(uuid)

        if location.kind not in DERIVATION_TARGET_KINDS:
            raise LayoutError(f'A resource cannot be derived from {location.kind} {uuid}.')

        references.append(DerivedFromRef.model_validate({'kind': location.kind, 'uuid': uuid}))

    return references


def _save_resource(
    layout: StorageLayout,
    kind: ResourceKind,
    manifest: Manifest,
    save_files: Callable[[Path], None],
) -> UUID:
    """Assemble a resource in staging, then rename it into place; return its uuid."""
    target = layout.resource_folder(kind, manifest.uuid)

    if target.exists():
        raise LayoutError(f'{target} already exists.')

    staging = layout.root / StorageLayout.STAGING_DIRECTORY / str(manifest.uuid)
    staging.mkdir(parents=True)

    try:
        save_files(staging)
        save_manifest(staging / MANIFEST_FILENAME, manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return manifest.uuid


def save_campaign_resource(
    layout: StorageLayout,
    *,
    label: str,
    comments: str = '',
    sample_name: str = '',
    sample_description: str = '',
) -> UUID:
    """Save a campaign that diffraction resources can name; return its uuid."""
    manifest = CampaignManifest(
        uuid=uuid4(),
        created_at=datetime.now(timezone.utc),
        label=label,
        comments=comments,
        sample_name=sample_name,
        sample_description=sample_description,
    )
    return _save_resource(layout, ResourceKind.CAMPAIGN, manifest, lambda folder: None)


def save_diffraction_resource(
    layout: StorageLayout,
    data: AssembledDiffractionData,
    *,
    label: str = '',
    comments: str = '',
    campaign: UUID | None = None,
    derived_from: Sequence[UUID] = (),
    detector_distance_m: float | None = None,
    photon_energy_eV: float | None = None,  # noqa: N803
    exposure_time_s: float | None = None,
    tomography_angle_deg: float | None = None,
    tilt_angle_deg: float | None = None,
    polarization: Polarization | None = None,
    crop_center_x_px: int | None = None,
    crop_center_y_px: int | None = None,
) -> UUID:
    """Save assembled diffraction data as a resource; return its uuid.

    `campaign` and `derived_from` name resources already saved, by uuid.

    Raises:
        LayoutError: if `campaign` is not a campaign, or a `derived_from` resource is
            missing or of a kind nothing derives from; nothing is written then.
    """
    if campaign is not None and layout.locate(campaign).kind != ResourceKind.CAMPAIGN:
        raise LayoutError(f'Resource {campaign} is not a campaign.')

    manifest = DiffractionManifest(
        uuid=uuid4(),
        created_at=datetime.now(timezone.utc),
        label=label,
        comments=comments,
        campaign_uuid=campaign,
        derived_from=_derived_from(layout, derived_from),
        detector_distance_m=detector_distance_m,
        photon_energy_eV=photon_energy_eV,
        exposure_time_s=exposure_time_s,
        tomography_angle_deg=tomography_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        polarization=polarization,
        crop_center_x_px=crop_center_x_px,
        crop_center_y_px=crop_center_y_px,
    )
    file_name = manifest.files['diffraction']
    return _save_resource(
        layout,
        ResourceKind.DIFFRACTION,
        manifest,
        lambda folder: save_diffraction_data(folder / file_name, data),
    )


def save_product_resource(
    layout: StorageLayout, product: Product, *, derived_from: Sequence[UUID] = ()
) -> UUID:
    """Save a product as a resource; return its uuid.

    Raises:
        LayoutError: if a `derived_from` resource is missing or of a kind nothing
            derives from; nothing is written then.
    """
    manifest = ProductManifest(
        uuid=uuid4(),
        created_at=datetime.now(timezone.utc),
        derived_from=_derived_from(layout, derived_from),
    )
    file_name = manifest.files['product']
    return _save_resource(
        layout,
        ResourceKind.PRODUCT,
        manifest,
        lambda folder: save_product(folder / file_name, product),
    )
