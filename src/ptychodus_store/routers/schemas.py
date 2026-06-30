"""Pydantic response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ptychodus_store.db.base import IngestState


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


class CampaignRead(_RowBase):
    label: str
    comments: str
    sample_name: str
    sample_description: str
    tags: list[str]


class DerivedFromEdge(BaseModel):
    kind: Literal['diffraction', 'product', 'fluorescence']
    uuid: UUID


class DiffractionRead(_RowBase):
    label: str
    comments: str
    campaign_uuid: UUID | None
    derived_from: list[DerivedFromEdge] = []

    detector_distance_m: float | None
    probe_energy_eV: float | None  # noqa: N815
    probe_photon_count: int | None
    exposure_time_s: float | None
    tomography_angle_deg: float | None
    crop_center_x_px: int | None
    crop_center_y_px: int | None

    pattern_dtype: str | None
    pattern_height_px: int | None
    pattern_width_px: int | None
    num_patterns_total: int | None
    detector_pixel_width_m: float | None
    detector_pixel_height_m: float | None


class ProductRead(_RowBase):
    derived_from: list[DerivedFromEdge] = []

    name: str | None
    comments: str | None
    detector_distance_m: float | None
    probe_energy_eV: float | None  # noqa: N815
    probe_photon_count: int | None
    exposure_time_s: float | None
    mass_attenuation_m2_kg: float | None
    tomography_angle_deg: float | None

    object_layers: int | None
    object_height_px: int | None
    object_width_px: int | None
    object_pixel_width_m: float | None
    object_pixel_height_m: float | None
    probe_modes: int | None
    probe_height_px: int | None
    probe_width_px: int | None
    num_scan_points: int | None
    num_loss_epochs: int | None


class FluorescenceRead(_RowBase):
    label: str
    comments: str
    derived_from: list[DerivedFromEdge] = []

    element_names: list[str]
    map_height_px: int | None
    map_width_px: int | None


ItemT = TypeVar('ItemT', bound=BaseModel)


class Page(BaseModel, Generic[ItemT]):
    items: list[ItemT]
    total: int
    limit: int
    offset: int


class ResourceRef(BaseModel):
    kind: Literal['campaign', 'diffraction', 'product', 'fluorescence']
    uuid: UUID


class LineageNode(BaseModel):
    kind: Literal['campaign', 'diffraction', 'product', 'fluorescence']
    uuid: UUID
    label: str = ''


class LineageRead(BaseModel):
    node: LineageNode
    ancestors: list[LineageNode]
    descendants: list[LineageNode]
    campaign: CampaignRead | None


class StoreStats(BaseModel):
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
