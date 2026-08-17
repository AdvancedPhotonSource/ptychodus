from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from ptychodus_store.db.base import Base, IngestState


def _ts() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


def _updated_ts() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )


class Campaign(Base):
    __tablename__ = 'campaign'

    uuid: Mapped[UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True)
    label: Mapped[str] = mapped_column(String, default='', nullable=False)
    comments: Mapped[str] = mapped_column(String, default='', nullable=False)
    sample_name: Mapped[str] = mapped_column(String, default='', nullable=False)
    sample_description: Mapped[str] = mapped_column(String, default='', nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    folder_path: Mapped[str] = mapped_column(String, nullable=False)
    manifest_mtime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_from_manifest_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingest_state: Mapped[IngestState] = mapped_column(
        String, default=IngestState.DISCOVERED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _updated_ts()

    __table_args__ = (
        Index('ix_campaign_ingest_state', 'ingest_state'),
        Index('ix_campaign_sample_name', 'sample_name'),
    )


class Diffraction(Base):
    __tablename__ = 'diffraction'

    uuid: Mapped[UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True)
    label: Mapped[str] = mapped_column(String, default='', nullable=False)
    comments: Mapped[str] = mapped_column(String, default='', nullable=False)
    campaign_uuid: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey('campaign.uuid', ondelete='SET NULL'),
        nullable=True,
    )

    # Manifest-supplied
    detector_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_energy_eV: Mapped[float | None] = mapped_column(Float, nullable=True)  # noqa: N815
    probe_photon_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exposure_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tomography_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    tilt_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    polarization: Mapped[str | None] = mapped_column(String, nullable=True)
    crop_center_x_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crop_center_y_px: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # HDF5-derived
    pattern_dtype: Mapped[str | None] = mapped_column(String, nullable=True)
    pattern_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pattern_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_patterns_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detector_pixel_width_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_pixel_height_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Bookkeeping
    folder_path: Mapped[str] = mapped_column(String, nullable=False)
    manifest_mtime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_from_manifest_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingest_state: Mapped[IngestState] = mapped_column(
        String, default=IngestState.DISCOVERED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _updated_ts()

    __table_args__ = (
        Index('ix_diffraction_campaign_uuid', 'campaign_uuid'),
        Index('ix_diffraction_probe_energy_eV', 'probe_energy_eV'),
        Index('ix_diffraction_tomography_angle_deg', 'tomography_angle_deg'),
        Index('ix_diffraction_tilt_angle_deg', 'tilt_angle_deg'),
        Index('ix_diffraction_ingest_state', 'ingest_state'),
    )


class Product(Base):
    __tablename__ = 'product'

    uuid: Mapped[UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True)

    # HDF5-derived (product.h5 is the source of truth for everything below)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    comments: Mapped[str | None] = mapped_column(String, nullable=True)
    detector_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_energy_eV: Mapped[float | None] = mapped_column(Float, nullable=True)  # noqa: N815
    probe_photon_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exposure_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    mass_attenuation_m2_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    tomography_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    tilt_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    polarization: Mapped[str | None] = mapped_column(String, nullable=True)
    object_layers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    object_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    object_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    object_pixel_width_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    object_pixel_height_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_modes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probe_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probe_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_scan_points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_loss_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Bookkeeping
    folder_path: Mapped[str] = mapped_column(String, nullable=False)
    manifest_mtime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_from_manifest_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingest_state: Mapped[IngestState] = mapped_column(
        String, default=IngestState.DISCOVERED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _updated_ts()

    __table_args__ = (
        Index('ix_product_probe_energy_eV', 'probe_energy_eV'),
        Index('ix_product_ingest_state', 'ingest_state'),
    )


class Fluorescence(Base):
    __tablename__ = 'fluorescence'

    uuid: Mapped[UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True)
    label: Mapped[str] = mapped_column(String, default='', nullable=False)
    comments: Mapped[str] = mapped_column(String, default='', nullable=False)

    # HDF5-derived
    element_names: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    map_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    map_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Bookkeeping
    folder_path: Mapped[str] = mapped_column(String, nullable=False)
    manifest_mtime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_from_manifest_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingest_state: Mapped[IngestState] = mapped_column(
        String, default=IngestState.DISCOVERED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _updated_ts()

    __table_args__ = (Index('ix_fluorescence_ingest_state', 'ingest_state'),)


class DerivationEdge(Base):
    """Flattens manifest `derived_from` lists for fast lineage queries.

    No DB-level FK enforcement: target spans 4 tables. Integrity is checked at
    ingest time and reflected in the source row's `ingest_state`.
    """

    __tablename__ = 'derivation_edge'

    source_uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), primary_key=True
    )
    target_uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), primary_key=True
    )
    source_kind: Mapped[str] = mapped_column(String, nullable=False)
    target_kind: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index('ix_derivation_edge_source', 'source_kind', 'source_uuid'),
        Index('ix_derivation_edge_target', 'target_kind', 'target_uuid'),
    )
