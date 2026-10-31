"""The pvapy-free core of ``ptychodus-stream``: one scan at a time, from frames to the store.

A scan is collected from a stream of detector frames and probe positions, paired by
scan index; a frame joins the scan only once positions cover its index. Each scan is
prepared exactly as batch assembly prepares a dataset, its probe and object estimated
by the same generator plugins the GUI runs, and the finished scan written into the
webapp storage root as a diffraction resource and a product derived from it.

Everything here is driven through :class:`ScanPipeline`, which owns two threads: one
that ingests what the stream delivers and one that estimates, publishes and writes.
The PV contract -- the layouts the positions arrive in -- is defined here as plain
dicts, so the processor and the replay script share one definition of it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID
import logging
import queue
import tempfile
import threading
import time

import numpy
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptychodus.api.accumulate import DiffractionMomentAccumulator, DiffractionMoments
from ptychodus.api.accumulate import estimate_incident_beam
from ptychodus.api.assemble import ArrayPreprocessor, AssembledDiffractionData, DetectorCorrections
from ptychodus.api.diffraction import (
    BadPixels,
    BeamCenter,
    CropRegion,
    DetectorBackground,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    FlatField,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.interpolate import interpolate_at_indexes
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider, compute_object_geometry
from ptychodus.api.parallax import ParallaxEstimate, estimate_defocus_from_parallax
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import BinningStep, DiffractionPrepPipeline
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import ProductGeometry, compute_product_geometry
from ptychodus.api.simulate.object import ObjectGenerator
from ptychodus.api.simulate.probe import ProbeGenerator, rescale_probe_intensity
from ptychodus.api.typing import BooleanArrayType, IntegerArrayType, RealArrayType
from ptychodus.api.storage import (
    StorageLayout,
    save_campaign_resource,
    save_diffraction_resource,
    save_product_resource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PositionSchema(StrEnum):
    """How probe positions arrive; see the PV contract in the streaming documentation."""

    BATCHED = 'batched'
    """One structure with a field per axis, each ``{num_events, event_ids, positions}``."""
    SGSOCKET = 'sgsocket'
    """One NTNDArray per event, its ``uniqueId`` the index and its attributes the values."""


class PvProtocol(StrEnum):
    CA = 'ca'
    PVA = 'pva'


class AxisSource(BaseModel):
    """Where one position axis is found, and its unit."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    name: str
    """The structure field (batched) or NDAttribute (sgsocket) carrying the axis."""
    scale_m: float = 1.0
    """Metres per unit of the stream's values; a negative scale flips the axis."""


class PixelMapSource(BaseModel):
    """A detector map read from a file by a reader plugin."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    file_path: Path
    file_type: str


class StreamConfiguration(BaseModel):
    """Everything ``ptychodus-stream`` is configured with; every key is snake_case.

    Read from YAML or JSON with :meth:`from_file`, and updated by a control-channel
    ``configure`` through :meth:`merged`.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    # Channels
    input_channel: str
    output_pv_prefix: str
    trigger_pv: str
    position_channel: str
    scan_number_pv: str | None = None
    tomography_angle_pv: str | None = None
    pv_protocol: PvProtocol = PvProtocol.CA

    # Experiment
    photon_energy_eV: float = Field(gt=0.0)  # noqa: N815
    detector_distance_m: float = Field(gt=0.0)
    detector_pixel_size_m: float = Field(gt=0.0)
    focus_object_distance_m: float = 0.0
    exposure_time_s: float = Field(0.0, ge=0.0)
    i0_photons_per_count: float = Field(0.0, ge=0.0)
    tomography_angle_deg: float = 0.0

    # Pixel maps
    bad_pixels_file: PixelMapSource | None = None
    background_file: PixelMapSource | None = None
    flat_field_file: PixelMapSource | None = None

    # Preparation
    crop_center_x_px: int | None = None
    crop_center_y_px: int | None = None
    crop_width_px: int | None = Field(None, gt=0)
    crop_height_px: int | None = Field(None, gt=0)
    pipeline: DiffractionPrepPipeline = Field(default_factory=DiffractionPrepPipeline)
    total_counts_lower_bound: int | None = None
    total_counts_upper_bound: int | None = None

    # Positions
    position_schema: PositionSchema = PositionSchema.BATCHED
    x_axis: AxisSource = AxisSource(name='x')
    y_axis: AxisSource = AxisSource(name='y')
    i0_name: str | None = None
    index_offset: int = 0

    # Models
    probe_generator: str = 'average_pattern'
    probe_options: dict[str, Any] = Field(default_factory=dict)
    object_generator: str = 'stxm_dpc'
    object_options: dict[str, Any] = Field(default_factory=dict)

    # Parallax
    parallax_bin_size_px: int = Field(4, ge=0)
    parallax_min_significance: float = Field(3.0, ge=0.0)
    parallax_min_defocus_dof: float = Field(1.0, ge=0.0)

    # Timing
    estimate_period_s: float = Field(5.0, gt=0.0)
    settle_s: float = Field(2.0, ge=0.0)
    unpaired_retention_s: float = Field(60.0, gt=0.0)
    num_frames_per_block: int = Field(64, ge=1)
    consumer_queue_size: int = Field(100_000, ge=1)
    queue_warning_frames: int = Field(10_000, ge=1)

    # Output
    output_directory: Path | None = None
    campaign_label: str = ''

    @model_validator(mode='after')
    def _check_crop(self) -> StreamConfiguration:
        crop = [
            self.crop_center_x_px,
            self.crop_center_y_px,
            self.crop_width_px,
            self.crop_height_px,
        ]

        if any(value is None for value in crop) and not all(value is None for value in crop):
            raise ValueError('Give all four crop keys, or none.')

        return self

    @classmethod
    def from_file(cls, file_path: Path) -> StreamConfiguration:
        """Read a configuration from YAML or JSON (YAML parses JSON too)."""
        return cls.model_validate(yaml.safe_load(file_path.read_text()))

    def merged(self, overrides: Mapping[str, Any]) -> StreamConfiguration:
        """This configuration with `overrides` replacing the keys they name, validated.

        A nested key -- an axis, a pixel map, the pipeline -- is replaced whole.
        """
        return type(self).model_validate({**self.model_dump(), **overrides})

    def get_read_region(self) -> CropRegion | None:
        """The detector region frames are cropped to, or None for the whole frame."""
        if (
            self.crop_center_x_px is None
            or self.crop_center_y_px is None
            or self.crop_width_px is None
            or self.crop_height_px is None
        ):
            return None

        return CropRegion.from_center_extent(
            BeamCenter(x_px=self.crop_center_x_px, y_px=self.crop_center_y_px),
            ImageExtent(width_px=self.crop_width_px, height_px=self.crop_height_px),
        )

    def create_metadata(self, name: str, *, tomography_angle_deg: float) -> ProductMetadata:
        """The product metadata of a scan named `name` at `tomography_angle_deg`."""
        return ProductMetadata(
            name=name,
            comments='ptychodus-stream',
            detector_distance_m=self.detector_distance_m,
            photon_energy_eV=self.photon_energy_eV,
            probe_photon_count=0.0,
            exposure_time_s=self.exposure_time_s,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=tomography_angle_deg,
            focus_object_distance_m=self.focus_object_distance_m,
            i0_photons_per_count=self.i0_photons_per_count,
        )


def _validated_options(defaults: BaseModel, overrides: Mapping[str, Any], name: str) -> BaseModel:
    unknown = sorted(set(overrides) - set(type(defaults).model_fields))

    if unknown:
        raise ValueError(f'{name} accepts no option {", ".join(unknown)}.')

    return type(defaults).model_validate({**defaults.model_dump(), **overrides})


def prepare_storage_root(directory: Path) -> StorageLayout:
    """Create the kind directories of the storage root `directory`, and check it is writable.

    Raises:
        OSError: if the directories cannot be created or written to.
    """
    layout = StorageLayout(directory)
    layout.ensure_kind_dirs()

    with tempfile.NamedTemporaryFile(dir=layout.root, prefix='.writable-'):
        pass

    return layout


@dataclass(frozen=True)
class ScanSetup:
    """A configuration with everything it names loaded and checked: maps, models, store."""

    configuration: StreamConfiguration
    bad_pixels: BadPixels | None
    background: DetectorBackground | None
    flat_field: FlatField | None
    probe_generator: ProbeGenerator
    probe_options: BaseModel
    object_generator: ObjectGenerator
    object_options: BaseModel
    layout: StorageLayout | None

    @classmethod
    def from_configuration(
        cls, configuration: StreamConfiguration, registry: PluginRegistry
    ) -> ScanSetup:
        """Load the maps, resolve the models and prepare the store `configuration` names.

        Raises:
            LookupError: if a reader or generator is not registered.
            ValueError: if a model option is unknown or invalid.
            OSError: if a map cannot be read or the store is not writable.
        """
        bad_pixels = None
        background = None
        flat_field = None

        if configuration.bad_pixels_file is not None:
            source = configuration.bad_pixels_file
            reader = registry.bad_pixels_file_readers.get_strategy_by_name(source.file_type)
            bad_pixels = reader.read(source.file_path)

        if configuration.background_file is not None:
            source = configuration.background_file
            map_reader = registry.background_file_readers.get_strategy_by_name(source.file_type)
            background = map_reader.read(source.file_path)

        if configuration.flat_field_file is not None:
            source = configuration.flat_field_file
            map_reader = registry.flat_field_file_readers.get_strategy_by_name(source.file_type)
            flat_field = map_reader.read(source.file_path)

        probe_generator = registry.probe_generators.get_strategy_by_name(
            configuration.probe_generator
        )
        object_generator = registry.object_generators.get_strategy_by_name(
            configuration.object_generator
        )
        output_directory = configuration.output_directory
        return cls(
            configuration=configuration,
            bad_pixels=bad_pixels,
            background=background,
            flat_field=flat_field,
            probe_generator=probe_generator,
            probe_options=_validated_options(
                probe_generator.get_default_options(),
                configuration.probe_options,
                configuration.probe_generator,
            ),
            object_generator=object_generator,
            object_options=_validated_options(
                object_generator.get_default_options(),
                configuration.object_options,
                configuration.object_generator,
            ),
            layout=None if output_directory is None else prepare_storage_root(output_directory),
        )


# ---------------------------------------------------------------------------
# The PV contract
# ---------------------------------------------------------------------------


class Axis(StrEnum):
    X = 'x'
    Y = 'y'
    I0 = 'i0'


@dataclass(frozen=True)
class PositionUpdate:
    """Values of one axis keyed by scan index: metres for x and y, counts for I0."""

    axis: Axis
    indexes: IntegerArrayType
    values: RealArrayType


def pack_batched_axes(
    axes: Mapping[str, tuple[Sequence[int] | IntegerArrayType, Sequence[float] | RealArrayType]],
) -> dict[str, dict[str, Any]]:
    """The batched structure's value: per field, ``{num_events, event_ids, positions}``."""
    return {
        name: {
            'num_events': len(event_ids),
            'event_ids': numpy.asarray(event_ids, dtype=numpy.uint64),
            'positions': numpy.asarray(positions, dtype=numpy.float64),
        }
        for name, (event_ids, positions) in axes.items()
    }


def _configured_axes(configuration: StreamConfiguration) -> dict[Axis, tuple[str, float]]:
    axes = {
        Axis.X: (configuration.x_axis.name, configuration.x_axis.scale_m),
        Axis.Y: (configuration.y_axis.name, configuration.y_axis.scale_m),
    }

    if configuration.i0_name is not None:
        axes[Axis.I0] = (configuration.i0_name, 1.0)

    return axes


def parse_batched_axes(
    value: Mapping[str, Any], configuration: StreamConfiguration
) -> list[PositionUpdate]:
    """Read the configured axes from a batched structure's value; absent fields are skipped."""
    updates: list[PositionUpdate] = []

    for axis, (name, scale) in _configured_axes(configuration).items():
        fields = value.get(name)

        if not fields:
            continue

        event_ids = numpy.asarray(fields['event_ids'], dtype=numpy.int64)
        positions = numpy.asarray(fields['positions'], dtype=numpy.float64)

        if event_ids.size != positions.size:
            raise ValueError(
                f'Axis "{name}" carries {event_ids.size} event ids for {positions.size} positions.'
            )

        if event_ids.size > 0:
            updates.append(
                PositionUpdate(axis, event_ids + configuration.index_offset, positions * scale)
            )

    return updates


def pack_sgsocket_attributes(
    configuration: StreamConfiguration, *, x_m: float, y_m: float, i0: float | None = None
) -> dict[str, float]:
    """The NDAttribute values, by configured name and in stream units, of one event."""
    attributes = {
        configuration.x_axis.name: x_m / configuration.x_axis.scale_m,
        configuration.y_axis.name: y_m / configuration.y_axis.scale_m,
    }

    if configuration.i0_name is not None and i0 is not None:
        attributes[configuration.i0_name] = i0

    return attributes


def _attribute_value(value: Any) -> float:
    # pvapy gives an NTAttribute's variant value as ({'value': x}, {'value': type}).
    if isinstance(value, tuple):
        value = value[0]

    if isinstance(value, Mapping):
        value = value['value']

    return float(value)


def parse_sgsocket_event(
    value: Mapping[str, Any], configuration: StreamConfiguration
) -> list[PositionUpdate]:
    """Read one event's configured axes from an NTNDArray value's attributes."""
    attributes = {attribute['name']: attribute['value'] for attribute in value.get('attribute', ())}
    index = numpy.array([int(value['uniqueId']) + configuration.index_offset])
    updates: list[PositionUpdate] = []

    for axis, (name, scale) in _configured_axes(configuration).items():
        if name in attributes:
            reading = _attribute_value(attributes[name]) * scale
            updates.append(PositionUpdate(axis, index, numpy.array([reading])))

    return updates


def parse_positions(
    value: Mapping[str, Any], configuration: StreamConfiguration
) -> list[PositionUpdate]:
    """Read the positions in `value` by the configured schema."""
    if configuration.position_schema is PositionSchema.SGSOCKET:
        return parse_sgsocket_event(value, configuration)

    return parse_batched_axes(value, configuration)


def display_images(array: numpy.ndarray) -> tuple[RealArrayType, RealArrayType]:
    """The amplitude and phase, as float32 images, of the first mode or layer of `array`."""
    first = array[0] if array.ndim == 3 else array
    amplitude: RealArrayType = numpy.abs(first).astype(numpy.float32)
    phase: RealArrayType = numpy.angle(first).astype(numpy.float32)
    return amplitude, phase


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@dataclass
class _AxisSamples:
    indexes: list[IntegerArrayType] = field(default_factory=list)
    values: list[RealArrayType] = field(default_factory=list)
    arrival_s: list[RealArrayType] = field(default_factory=list)
    low: int | None = None
    high: int | None = None

    def extend(self, indexes: IntegerArrayType, values: RealArrayType, arrival_s: float) -> None:
        self.indexes.append(indexes)
        self.values.append(values)
        self.arrival_s.append(numpy.full(indexes.shape, arrival_s))

        if indexes.size > 0:
            low = int(indexes.min())
            high = int(indexes.max())
            self.low = low if self.low is None else min(self.low, low)
            self.high = high if self.high is None else max(self.high, high)

    def arrays(self) -> tuple[IntegerArrayType, RealArrayType, RealArrayType]:
        if not self.indexes:
            return (
                numpy.zeros(0, dtype=numpy.int64),
                numpy.zeros(0),
                numpy.zeros(0),
            )

        return (
            numpy.concatenate(self.indexes),
            numpy.concatenate(self.values),
            numpy.concatenate(self.arrival_s),
        )


class ProbePositionStream:
    """Probe positions arriving axis by axis, each axis interpolated on its own.

    Appended by one thread and read by others, under a lock. Readers copy references
    under it and join the arrays outside, since appended arrays are never mutated.
    """

    def __init__(self, axes: Collection[Axis]) -> None:
        self._lock = threading.Lock()
        self._axes = {axis: _AxisSamples() for axis in axes}

    def append(self, update: PositionUpdate, arrival_s: float) -> None:
        samples = self._axes.get(update.axis)

        if samples is None:
            return

        with self._lock:
            samples.extend(
                numpy.asarray(update.indexes, dtype=numpy.int64),
                numpy.asarray(update.values, dtype=numpy.float64),
                arrival_s,
            )

    def _snapshot(self) -> dict[Axis, _AxisSamples]:
        with self._lock:
            return {
                axis: _AxisSamples(list(s.indexes), list(s.values), list(s.arrival_s))
                for axis, s in self._axes.items()
            }

    def coverage(self) -> tuple[int, int] | None:
        """The index range both x and y cover, or None while they cover none."""
        with self._lock:
            x = self._axes[Axis.X]
            y = self._axes[Axis.Y]

            if x.low is None or x.high is None or y.low is None or y.high is None:
                return None

            low = max(x.low, y.low)
            high = min(x.high, y.high)

        return (low, high) if low <= high else None

    def at(self, indexes: IntegerArrayType) -> tuple[BooleanArrayType, ProbePositionSequence]:
        """The positions at `indexes`, and the mask of those x and y both cover.

        Each axis is interpolated by :func:`interpolate_at_indexes`. I0 is attached
        when it covers every covered index, and otherwise left out with a warning, since
        a position sequence carries I0 on every position or on none.
        """
        targets = numpy.asarray(indexes, dtype=numpy.int64)
        resampled: dict[Axis, tuple[BooleanArrayType, RealArrayType]] = {}

        for axis, samples in self._snapshot().items():
            source_indexes, values, _ = samples.arrays()
            in_range, at_targets = interpolate_at_indexes(source_indexes, values, targets)
            full = numpy.full(targets.shape, numpy.nan)
            full[in_range] = at_targets
            resampled[axis] = (in_range, full)

        covered = resampled[Axis.X][0] & resampled[Axis.Y][0]
        x_m = resampled[Axis.X][1]
        y_m = resampled[Axis.Y][1]
        i0: RealArrayType | None = None

        if Axis.I0 in resampled:
            i0_in_range, i0_values = resampled[Axis.I0]

            if (i0_in_range | ~covered).all():
                i0 = i0_values
            else:
                logger.warning('I0 does not cover every paired frame; leaving it out.')

        positions = [
            ProbePosition(
                index=int(targets[k]),
                x_m=float(x_m[k]),
                y_m=float(y_m[k]),
                probe_photon_count=None if i0 is None else float(i0[k]),
            )
            for k in numpy.flatnonzero(covered)
        ]
        return covered, ProbePositionSequence(positions)

    def prune(self, older_than_s: float, frame_index_range: tuple[int, int] | None) -> None:
        """Drop samples older than `older_than_s` that no frame could pair with.

        A sample is kept while it lies within `frame_index_range` or is the nearest
        anchor on either side of it. With no range, every old sample goes.
        """
        with self._lock:
            for samples in self._axes.values():
                if not samples.indexes:
                    continue

                indexes = numpy.concatenate(samples.indexes)
                values = numpy.concatenate(samples.values)
                arrival_s = numpy.concatenate(samples.arrival_s)
                keep = arrival_s >= older_than_s

                if frame_index_range is not None:
                    low, high = frame_index_range
                    keep |= (indexes >= low) & (indexes <= high)
                    below = indexes[indexes < low]
                    above = indexes[indexes > high]

                    if below.size > 0:
                        keep |= indexes == below.max()

                    if above.size > 0:
                        keep |= indexes == above.min()

                samples.indexes = [indexes[keep]]
                samples.values = [values[keep]]
                samples.arrival_s = [arrival_s[keep]]
                samples.low = int(indexes[keep].min()) if keep.any() else None
                samples.high = int(indexes[keep].max()) if keep.any() else None


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One detector frame, its scan index and when it reached the processor."""

    index: int
    image: numpy.ndarray
    arrival_s: float


@dataclass
class _Block:
    data: AssembledDiffractionData
    arrival_s: RealArrayType


class ScanSnapshot(ProbeGeometryProvider, ObjectGeometryProvider):
    """A scan's moments and paired positions, as the generator plugins read them."""

    def __init__(
        self,
        metadata: ProductMetadata,
        detector_extent: ImageExtent,
        detector_pixel_geometry: PixelGeometry,
        positions: ProbePositionSequence,
        moments: DiffractionMoments,
        *,
        mean_amplitude_transmission: float = 1.0,
    ) -> None:
        self.metadata = metadata
        self._detector_extent = detector_extent
        self._detector_pixel_geometry = detector_pixel_geometry
        self._positions = positions
        self._moments = moments
        self._mean_amplitude_transmission = mean_amplitude_transmission

    def with_metadata(
        self, metadata: ProductMetadata, *, mean_amplitude_transmission: float
    ) -> ScanSnapshot:
        return ScanSnapshot(
            metadata,
            self._detector_extent,
            self._detector_pixel_geometry,
            self._positions,
            self._moments,
            mean_amplitude_transmission=mean_amplitude_transmission,
        )

    def _derived(self) -> ProductGeometry:
        return compute_product_geometry(
            photon_energy_eV=self.metadata.photon_energy_eV,
            probe_photon_count=self.metadata.probe_photon_count,
            exposure_time_s=self.metadata.exposure_time_s,
            detector_distance_m=self.metadata.detector_distance_m,
            focus_object_distance_m=self.metadata.focus_object_distance_m,
            far_field=self.metadata.far_field,
            detector_extent=self._detector_extent,
            detector_pixel_geometry=self._detector_pixel_geometry,
        )

    @property
    def detector_distance_m(self) -> float:
        return self.metadata.detector_distance_m

    @property
    def far_field(self) -> bool:
        return self.metadata.far_field

    @property
    def focus_object_distance_m(self) -> float:
        return self.metadata.focus_object_distance_m

    @property
    def probe_photon_count(self) -> float:
        return self.metadata.probe_photon_count

    @property
    def photon_wavelength_m(self) -> float:
        return self.metadata.photon_wavelength_m

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return self._derived().probe_power_W

    @property
    def num_scan_points(self) -> int:
        return len(self._positions)

    @property
    def mean_amplitude_transmission(self) -> float:
        return self._mean_amplitude_transmission

    @property
    def object_plane_propagation_distance_m(self) -> float:
        return self._derived().object_plane_propagation_distance_m

    def get_detector_pixel_geometry(self) -> PixelGeometry:
        return self._detector_pixel_geometry

    def get_probe_geometry(self) -> ProbeGeometry:
        pixel_geometry = self._derived().object_plane_pixel_geometry
        return ProbeGeometry(
            width_px=self._detector_extent.width_px,
            height_px=self._detector_extent.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    def get_probe_positions(self) -> ProbePositionSequence:
        return self._positions

    def get_object_geometry(self) -> ObjectGeometry:
        return compute_object_geometry(self._positions, self.get_probe_geometry())

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return None

    def get_diffraction_moments(self) -> DiffractionMoments:
        return self._moments


class ScanSession:
    """One scan: its frames, pending until positions cover them, and its positions.

    Only one thread changes a session; others may take a :meth:`snapshot` of it. The
    preprocessing, the accumulator and the detector geometry are built from the first
    frame, since the configuration does not state the detector's extent.
    """

    def __init__(self, setup: ScanSetup, *, campaign: _Campaign | None = None) -> None:
        configuration = setup.configuration
        self.setup = setup
        self.campaign = campaign
        self.positions = ProbePositionStream(_configured_axes(configuration).keys())
        self.first_frame_time: datetime | None = None
        self._preprocessor: ArrayPreprocessor | None = None
        self._accumulator: DiffractionMomentAccumulator | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._frame_dtype: numpy.dtype | None = None
        self._pending: list[_Block] = []
        self._paired: list[AssembledDiffractionData] = []
        self._nonfinite_pixels: list[BadPixels] = []
        self._seen_indexes: set[int] = set()
        self._last_index: int | None = None
        self._index_range: tuple[int, int] | None = None
        self._last_prune_s = float('-inf')
        # Written by the owning thread only.
        self.num_frames_received = 0
        self.num_frames_paired = 0
        self.num_frames_dropped = 0
        self.num_frames_rejected = 0
        self.num_frames_filtered = 0
        self.num_frames_duplicated = 0
        self.num_frames_regressed = 0

    @property
    def num_frames_pending(self) -> int:
        return sum(block.data.get_num_patterns() for block in self._pending)

    @property
    def has_frames(self) -> bool:
        return self.num_frames_paired > 0 or bool(self._pending)

    def _build(self, frame: Frame) -> bool:
        configuration = self.setup.configuration
        height, width = frame.image.shape
        metadata = DiffractionMetadata(
            num_patterns_per_array=[],
            pattern_dtype=frame.image.dtype,
            detector_extent=ImageExtent(width_px=width, height_px=height),
            detector_pixel_geometry=PixelGeometry(
                width_m=configuration.detector_pixel_size_m,
                height_m=configuration.detector_pixel_size_m,
            ),
        )

        try:
            dataset = SimpleDiffractionDataset(
                metadata,
                DiffractionDatasetLayoutNode.create_root(),
                [],
                self.setup.bad_pixels,
                background=self.setup.background,
                flat_field=self.setup.flat_field,
            )
            preprocessor = ArrayPreprocessor.from_dataset(
                dataset,
                configuration.pipeline,
                read_region=configuration.get_read_region(),
                total_counts_lower_bound=configuration.total_counts_lower_bound,
                total_counts_upper_bound=configuration.total_counts_upper_bound,
            )
        except ValueError:
            logger.exception(f'Cannot prepare {width}x{height} frames with this configuration!')
            return False

        processed_height, processed_width = preprocessor.processed_bad_pixels.shape
        bin_size_px = configuration.parallax_bin_size_px
        binning: BinningStep | None = None

        if bin_size_px > 0:
            if bin_size_px <= min(processed_height, processed_width):
                binning = BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px)
            else:
                logger.warning(
                    f'Parallax bins of {bin_size_px} px do not fit the '
                    f'{processed_width}x{processed_height} prepared frames; no parallax.'
                )

        self._preprocessor = preprocessor
        self._accumulator = DiffractionMomentAccumulator(
            processed_height,
            processed_width,
            bad_pixels=preprocessor.processed_bad_pixels,
            corrections=DetectorCorrections(
                background=preprocessor.carried_background,
                flat_field=preprocessor.carried_flat_field,
                subtract_background=True,
                divide_flat_field=True,
            ),
            binning=binning,
        )
        self._frame_shape = frame.image.shape
        self._frame_dtype = frame.image.dtype
        self.first_frame_time = datetime.now(timezone.utc)
        return True

    def add_frames(self, frames: Sequence[Frame]) -> None:
        """Preprocess `frames` into the pending buffer; reject those of another shape."""
        accepted: list[Frame] = []

        for frame in frames:
            self.num_frames_received += 1

            if self._frame_shape is None and not self._build(frame):
                self.num_frames_rejected += 1
                continue

            if frame.image.shape != self._frame_shape or frame.image.dtype != self._frame_dtype:
                self.num_frames_rejected += 1
                logger.warning(
                    f'Frame {frame.index} is {frame.image.dtype}{frame.image.shape}, not '
                    f'{self._frame_dtype}{self._frame_shape}; rejected.'
                )
                continue

            if frame.index in self._seen_indexes:
                self.num_frames_duplicated += 1

                if self.num_frames_duplicated == 1:
                    logger.warning(f'Frame index {frame.index} repeats within the scan.')

            if self._last_index is not None and frame.index < self._last_index:
                self.num_frames_regressed += 1

                if self.num_frames_regressed == 1:
                    logger.warning(
                        f'Frame index {frame.index} follows {self._last_index}; '
                        'indexes went backwards.'
                    )

            self._seen_indexes.add(frame.index)
            self._last_index = frame.index
            accepted.append(frame)

        if not accepted or self._preprocessor is None:
            return

        indexes = numpy.array([frame.index for frame in accepted], dtype=numpy.intp)
        array = SimpleDiffractionArray(
            'stream', indexes, numpy.stack([frame.image for frame in accepted])
        )
        block = self._preprocessor(array, on_nonfinite_pixels=self._nonfinite_pixels.append)
        arrival_s = {frame.index: frame.arrival_s for frame in accepted}
        self.num_frames_filtered += len(accepted) - block.get_num_patterns()
        self._pending.append(
            _Block(block, numpy.array([arrival_s[int(i)] for i in block.get_indexes()]))
        )
        low = int(indexes.min())
        high = int(indexes.max())

        if self._index_range is not None:
            low = min(low, self._index_range[0])
            high = max(high, self._index_range[1])

        self._index_range = (low, high)

    def add_positions(self, update: PositionUpdate, arrival_s: float) -> None:
        self.positions.append(update, arrival_s)

    def promote(self, now_s: float) -> int:
        """Pair what positions now cover; drop what waited too long. Return the newly paired."""
        coverage = self.positions.coverage()
        retention_s = self.setup.configuration.unpaired_retention_s
        newly_paired = 0
        still_pending: list[_Block] = []

        for block in self._pending:
            indexes = numpy.asarray(block.data.get_indexes())
            covered = numpy.zeros(indexes.shape, dtype=bool)

            if coverage is not None:
                covered = (indexes >= coverage[0]) & (indexes <= coverage[1])

            expired = ~covered & (block.arrival_s < now_s - retention_s)

            if covered.any():
                self._pair(block.data, covered)
                newly_paired += int(covered.sum())

            if expired.any():
                self.num_frames_dropped += int(expired.sum())

            waiting = ~covered & ~expired

            if waiting.any():
                still_pending.append(_Block(_select(block.data, waiting), block.arrival_s[waiting]))

        self._pending = still_pending
        self.num_frames_paired += newly_paired

        # Pruning joins every sample, so it runs at most once a second.
        if now_s - self._last_prune_s >= 1.0:
            self.positions.prune(now_s - retention_s, self._index_range)
            self._last_prune_s = now_s

        return newly_paired

    def _pair(self, data: AssembledDiffractionData, mask: BooleanArrayType) -> None:
        paired = data if mask.all() else _select(data, mask)

        if self._accumulator is not None:
            self._accumulator.add(paired.get_indexes(), paired.get_patterns())

        self._paired.append(paired)

    def drop_pending(self) -> int:
        """Drop every frame still waiting for positions; return how many."""
        dropped = self.num_frames_pending
        self.num_frames_dropped += dropped
        self._pending = []
        return dropped

    def snapshot(self, metadata: ProductMetadata) -> ScanSnapshot | None:
        """The paired frames' moments and positions, or None before anything paired."""
        accumulator = self._accumulator
        preprocessor = self._preprocessor

        if accumulator is None or preprocessor is None:
            return None

        moments = accumulator.get_moments()

        if moments is None:
            return None

        _, positions = self.positions.at(moments.indexes)
        height, width = moments.mean_frame.shape
        return ScanSnapshot(
            metadata,
            ImageExtent(width_px=width, height_px=height),
            preprocessor.processed_pixel_geometry,
            positions,
            moments,
        )

    def assemble(self) -> AssembledDiffractionData | None:
        """Gather the paired frames into one buffer, releasing each block as it is copied.

        Call only once nothing else adds to the session.
        """
        preprocessor = self._preprocessor

        if preprocessor is None or not self._paired:
            return None

        num_patterns = sum(block.get_num_patterns() for block in self._paired)
        height, width = preprocessor.processed_bad_pixels.shape
        bad_pixels = preprocessor.processed_bad_pixels.copy()

        for block in self._paired:
            bad_pixels |= block.get_bad_pixels()

        data = AssembledDiffractionData(
            -numpy.ones(num_patterns, dtype=numpy.intp),
            numpy.zeros((num_patterns, height, width), dtype=self._paired[0].get_patterns_dtype()),
            preprocessor.processed_pixel_geometry,
            bad_pixels,
            background=preprocessor.carried_background,
            flat_field=preprocessor.carried_flat_field,
        )
        offset = 0

        while self._paired:
            block = self._paired.pop(0)
            data.assemble(block, offset)
            offset += block.get_num_patterns()

        return data


def _select(data: AssembledDiffractionData, mask: BooleanArrayType) -> AssembledDiffractionData:
    return AssembledDiffractionData(
        numpy.asarray(data.get_indexes())[mask],
        numpy.asarray(data.get_patterns())[mask],
        data.get_pixel_geometry(),
        data.get_bad_pixels(),
        background=data.get_background(),
        flat_field=data.get_flat_field(),
    )


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------


def estimate_product(
    snapshot: ScanSnapshot, setup: ScanSetup, rng: numpy.random.Generator
) -> Product:
    """A product from a scan's moments, by the GUI's sequence.

    1. The incident beam -- photon count and mean transmission -- from the totals,
       corrected for beam decay by the positions' I0.
    2. When no focus is stated and the moments were binned, the parallax defocus; a
       resolved estimate sets the focus.
    3. The probe from its generator, scaled to the incident photon count.
    4. The object from its generator, with that probe.
    """
    configuration = setup.configuration
    moments = snapshot.get_diffraction_moments()
    positions = snapshot.get_probe_positions()
    metadata = snapshot.metadata
    beam = estimate_incident_beam(
        moments,
        positions.get_probe_photon_counts_by_index(),
        i0_photons_per_count=metadata.i0_photons_per_count,
    )
    metadata = replace(metadata, probe_photon_count=beam.incident_photon_count)
    snapshot = snapshot.with_metadata(
        metadata, mean_amplitude_transmission=beam.mean_amplitude_transmission
    )

    if metadata.focus_object_distance_m == 0.0 and moments.binning is not None:
        parallax = _estimate_parallax(snapshot, beam.moments)

        if parallax is not None and parallax.is_resolved(
            minimum_significance=configuration.parallax_min_significance,
            minimum_defocus_dof=configuration.parallax_min_defocus_dof,
        ):
            metadata = replace(metadata, focus_object_distance_m=-parallax.defocus_distance_m)
            snapshot = snapshot.with_metadata(
                metadata, mean_amplitude_transmission=beam.mean_amplitude_transmission
            )

    probe = rescale_probe_intensity(
        setup.probe_generator.generate(setup.probe_options, snapshot),
        metadata.probe_photon_count,
    )
    object_ = setup.object_generator.generate(setup.object_options, snapshot, rng=rng, probe=probe)
    return Product(
        metadata=metadata,
        probe_positions=positions,
        probes=ProbeEnsemble.from_probe(probe),
        object_=object_,
        losses=[],
    )


def _estimate_parallax(
    snapshot: ScanSnapshot, moments: DiffractionMoments
) -> ParallaxEstimate | None:
    try:
        estimate = estimate_defocus_from_parallax(
            moments,
            snapshot.get_probe_positions(),
            snapshot.get_object_geometry(),
            snapshot.get_probe_geometry(),
            photon_wavelength_m=snapshot.photon_wavelength_m,
        )
    except ValueError as exc:
        logger.warning(f'No parallax estimate: {exc}')
        return None

    if estimate is not None:
        logger.info(
            f'Parallax defocus {estimate.defocus_distance_m * 1e3:.4g} '
            f'+/- {estimate.standard_error_m * 1e3:.2g} mm, '
            f'Fresnel number {estimate.fresnel_number:.3g}.'
        )

    return estimate


# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanReport:
    """What became of a finished scan, as published on ``<prefix>:scan``."""

    label: str
    diffraction_uuid: UUID | None
    product_uuid: UUID | None
    num_frames_paired: int
    num_frames_dropped: int
    written: bool


@dataclass
class _Campaign:
    """A campaign the scans name: saved, and so given its uuid, by the worker."""

    label: str
    uuid: UUID | None = None


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class ScanPublisher(Protocol):
    """Where a pipeline's results go; called from its worker thread only."""

    def publish_estimate(self, product: Product) -> None: ...

    def publish_scan(self, report: ScanReport) -> None: ...


@dataclass(frozen=True)
class _FrameItem:
    frame: Frame


@dataclass(frozen=True)
class _PositionsItem:
    updates: Sequence[PositionUpdate]
    arrival_s: float


@dataclass(frozen=True)
class _TriggerItem:
    arrival_s: float


@dataclass(frozen=True)
class _ConfigureItem:
    setup: ScanSetup


@dataclass(frozen=True)
class _StopItem:
    pass


_Item = _FrameItem | _PositionsItem | _TriggerItem | _ConfigureItem | _StopItem


@dataclass(frozen=True)
class _CampaignJob:
    layout: StorageLayout
    campaign: _Campaign


@dataclass(frozen=True)
class _FinishJob:
    session: ScanSession
    label: str
    tomography_angle_deg: float


@dataclass(frozen=True)
class _StopJob:
    pass


_Job = _CampaignJob | _FinishJob | _StopJob


@dataclass
class _Settle:
    cut_s: float
    label: str
    tomography_angle_deg: float
    deferred_setup: ScanSetup | None = None


class ScanPipeline:
    """Scans from a stream, without losing data while it is busy.

    Producers -- the detector, the positioner, the trigger, ``configure`` and
    ``stop`` -- only enqueue, each item stamped with its arrival. Two threads of the
    pipeline's own do the rest:

    - **ingest** is the only thread that changes a scan: it prepares frames, pairs them
      with positions, and decides where scans end;
    - **worker** is the only thread that estimates, publishes or writes files, so
      publishes are in order and store writes are serialized.

    A trigger is acknowledged at once, then the scan settles: whatever arrives within
    ``settle_s`` of the trigger still belongs to it. At the cut, frames still waiting
    for positions are dropped, the scan goes to the worker to be finished, and the
    next scan begins with the same configuration.
    """

    def __init__(
        self,
        setup: ScanSetup,
        publisher: ScanPublisher,
        *,
        acknowledge_trigger: Callable[[], None] = lambda: None,
        read_scan_number: Callable[[], int | None] = lambda: None,
        read_tomography_angle_deg: Callable[[], float | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
        rng: numpy.random.Generator | None = None,
        idle_s: float = 0.05,
    ) -> None:
        self._publisher = publisher
        self._acknowledge_trigger = acknowledge_trigger
        self._read_scan_number = read_scan_number
        self._read_tomography_angle_deg = read_tomography_angle_deg
        self._clock = clock
        self._rng = numpy.random.default_rng() if rng is None else rng
        self._idle_s = idle_s
        self._items: queue.SimpleQueue[_Item] = queue.SimpleQueue()
        self._jobs: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._campaign: _Campaign | None = None
        self._session = self._begin(setup)
        self._settle: _Settle | None = None
        self._ingest_thread = threading.Thread(target=self._ingest, name='stream-ingest')
        self._worker_thread = threading.Thread(target=self._work, name='stream-worker')
        # Each counter has exactly one writing thread.
        self._num_frames_enqueued = 0  # producer
        self._num_frames_dequeued = 0  # ingest
        self._max_frames_queued = 0  # ingest
        self._num_frames_discarded = 0  # ingest
        self._num_trigger_failures = 0  # ingest
        self._num_ingest_errors = 0  # ingest
        self._finished_counts: dict[str, int] = {}  # ingest
        self._num_estimates = 0  # worker
        self._last_estimate_s: float | None = None  # worker
        self._last_estimate_duration_s = 0.0  # worker
        self._num_scans_stored = 0  # worker
        self._num_worker_errors = 0  # worker

    # Producers ---------------------------------------------------------------

    def start(self) -> None:
        self._ingest_thread.start()
        self._worker_thread.start()

    def submit_frame(self, index: int, image: numpy.ndarray) -> None:
        """Enqueue a copy of one frame; returns at once."""
        self._num_frames_enqueued += 1
        self._items.put(_FrameItem(Frame(index, numpy.array(image, copy=True), self._clock())))

    def submit_positions(self, updates: Sequence[PositionUpdate]) -> None:
        if updates:
            self._items.put(_PositionsItem(updates, self._clock()))

    def trigger(self) -> None:
        self._items.put(_TriggerItem(self._clock()))

    def configure(self, setup: ScanSetup) -> None:
        self._items.put(_ConfigureItem(setup))

    def stop(self, timeout_s: float | None = None) -> None:
        """Finish the scan in progress, store it, and stop both threads."""
        self._items.put(_StopItem())
        self._ingest_thread.join(timeout_s)
        self._worker_thread.join(timeout_s)

    def get_counters(self) -> dict[str, int | float | bool]:
        session = self._session
        finished = self._finished_counts
        return {
            'num_frames_queued': self._num_frames_enqueued - self._num_frames_dequeued,
            'max_frames_queued': self._max_frames_queued,
            'num_frames_received': finished.get('received', 0) + session.num_frames_received,
            'num_frames_paired': finished.get('paired', 0) + session.num_frames_paired,
            'num_frames_pending': session.num_frames_pending,
            'num_frames_dropped': finished.get('dropped', 0) + session.num_frames_dropped,
            'num_frames_rejected': finished.get('rejected', 0) + session.num_frames_rejected,
            'num_frames_filtered': finished.get('filtered', 0) + session.num_frames_filtered,
            'num_frames_duplicated': finished.get('duplicated', 0) + session.num_frames_duplicated,
            'num_frames_regressed': finished.get('regressed', 0) + session.num_frames_regressed,
            'num_frames_discarded': self._num_frames_discarded,
            'num_trigger_failures': self._num_trigger_failures,
            'num_ingest_errors': self._num_ingest_errors,
            'num_estimates': self._num_estimates,
            'last_estimate_duration_s': self._last_estimate_duration_s,
            'last_estimate_age_s': (
                -1.0 if self._last_estimate_s is None else self._clock() - self._last_estimate_s
            ),
            'num_scans_stored': self._num_scans_stored,
            'num_worker_errors': self._num_worker_errors,
            'is_settling': self._settle is not None,
            'is_ingest_alive': self._ingest_thread.is_alive(),
            'is_worker_alive': self._worker_thread.is_alive(),
        }

    # Ingest ------------------------------------------------------------------

    def _begin(self, setup: ScanSetup) -> ScanSession:
        label = setup.configuration.campaign_label
        current = '' if self._campaign is None else self._campaign.label

        if label != current:
            self._campaign = None

            if label and setup.layout is not None:
                self._campaign = _Campaign(label)
                # Queued ahead of every scan that names it, so saved before them.
                self._jobs.put(_CampaignJob(setup.layout, self._campaign))

        return ScanSession(setup, campaign=self._campaign)

    def _ingest(self) -> None:
        frames: list[Frame] = []

        while True:
            try:
                item: _Item | None = self._items.get(timeout=self._next_timeout_s())
            except queue.Empty:
                item = None

            try:
                if self._settle is not None and self._is_past_cut(item):
                    self._flush(frames)
                    self._cut()

                match item:
                    case None:
                        self._flush(frames)
                        self._session.promote(self._clock())
                    case _FrameItem(frame):
                        self._count_dequeued()
                        frames.append(frame)

                        if len(frames) >= self._session.setup.configuration.num_frames_per_block:
                            self._flush(frames)
                    case _PositionsItem(updates, arrival_s):
                        for update in updates:
                            self._session.add_positions(update, arrival_s)

                        self._flush(frames)
                        self._session.promote(self._clock())
                    case _TriggerItem(arrival_s):
                        self._flush(frames)
                        self._on_trigger(arrival_s)
                    case _ConfigureItem(setup):
                        self._flush(frames)
                        self._on_configure(setup)
                    case _StopItem():
                        self._flush(frames)
                        self._on_stop()
                        return
            except Exception:
                self._num_ingest_errors += 1
                logger.exception('Ingest failed on one item; continuing.')
                frames = []

    def _next_timeout_s(self) -> float:
        if self._settle is None:
            return self._idle_s

        return max(0.0, min(self._idle_s, self._settle.cut_s - self._clock()))

    def _is_past_cut(self, item: _Item | None) -> bool:
        assert self._settle is not None

        match item:
            case None:
                return self._clock() >= self._settle.cut_s
            case _FrameItem(frame):
                return frame.arrival_s >= self._settle.cut_s
            case _PositionsItem(_, arrival_s) | _TriggerItem(arrival_s):
                return arrival_s >= self._settle.cut_s
            case _:
                return False

    def _count_dequeued(self) -> None:
        queued = self._num_frames_enqueued - self._num_frames_dequeued
        self._num_frames_dequeued += 1

        if queued > self._max_frames_queued:
            self._max_frames_queued = queued

            if queued >= self._session.setup.configuration.queue_warning_frames:
                logger.warning(f'{queued} frames are waiting to be ingested.')

    def _flush(self, frames: list[Frame]) -> None:
        if frames:
            self._session.add_frames(frames)
            frames.clear()
            self._session.promote(self._clock())

    def _on_trigger(self, arrival_s: float) -> None:
        try:
            self._acknowledge_trigger()
        except Exception:
            self._num_trigger_failures += 1
            logger.exception('Failed to acknowledge the trigger.')

        self._begin_settle(arrival_s)

    def _begin_settle(self, arrival_s: float) -> None:
        if self._settle is not None:
            logger.info('Trigger received while a scan is settling; ignored.')
            return

        if not self._session.has_frames:
            logger.warning('Trigger received with no frames in the scan; ignored.')
            return

        configuration = self._session.setup.configuration
        scan_number = self._read_scan_number()
        angle_deg = self._read_tomography_angle_deg()
        start = self._session.first_frame_time or datetime.now(timezone.utc)

        if scan_number is None:
            label = f'scan {start.isoformat(timespec="seconds")}'

            if configuration.scan_number_pv is not None:
                logger.warning('No scan number to label the scan with; using its start time.')
        else:
            label = f'scan {scan_number}'

        if angle_deg is None:
            angle_deg = configuration.tomography_angle_deg

            if configuration.tomography_angle_pv is not None:
                logger.warning('No tomography angle read; using the configured one.')

        self._settle = _Settle(arrival_s + configuration.settle_s, label, angle_deg)

    def _cut(self) -> None:
        assert self._settle is not None
        settle = self._settle
        session = self._session
        session.promote(self._clock())
        dropped = session.drop_pending()

        if dropped:
            logger.warning(f'{dropped} frames had no positions when "{settle.label}" ended.')

        self._retire(session)
        self._jobs.put(_FinishJob(session, settle.label, settle.tomography_angle_deg))
        self._settle = None
        self._session = self._begin(settle.deferred_setup or session.setup)

    def _retire(self, session: ScanSession) -> None:
        counts = self._finished_counts
        self._finished_counts = {
            'received': counts.get('received', 0) + session.num_frames_received,
            'paired': counts.get('paired', 0) + session.num_frames_paired,
            'dropped': counts.get('dropped', 0) + session.num_frames_dropped,
            'rejected': counts.get('rejected', 0) + session.num_frames_rejected,
            'filtered': counts.get('filtered', 0) + session.num_frames_filtered,
            'duplicated': counts.get('duplicated', 0) + session.num_frames_duplicated,
            'regressed': counts.get('regressed', 0) + session.num_frames_regressed,
        }

    def _on_configure(self, setup: ScanSetup) -> None:
        if self._settle is not None:
            self._settle.deferred_setup = setup
            logger.info('Configuration received while a scan is settling; applied at its end.')
            return

        session = self._session
        discarded = session.num_frames_received

        if discarded:
            logger.warning(f'Configuration received mid-scan; discarding {discarded} frames.')
            self._num_frames_discarded += discarded

        self._retire(session)
        self._session = self._begin(setup)

    def _on_stop(self) -> None:
        if self._settle is not None:
            self._cut()
        elif self._session.has_frames:
            # Finished as if triggered, though nothing was written to the trigger PV.
            self._begin_settle(self._clock())

            if self._settle is not None:
                self._cut()

        self._jobs.put(_StopJob())

    # Worker ------------------------------------------------------------------

    def _work(self) -> None:
        next_estimate_s = self._clock() + self._session.setup.configuration.estimate_period_s
        estimated: tuple[ScanSession, int] | None = None

        while True:
            try:
                job: _Job | None = self._jobs.get(timeout=max(0.0, next_estimate_s - self._clock()))
            except queue.Empty:
                job = None

            try:
                match job:
                    case None:
                        session = self._session
                        progress = (session, session.num_frames_paired)

                        if session.num_frames_paired > 0 and progress != estimated:
                            self._estimate_periodically(session)
                            estimated = progress

                        period_s = session.setup.configuration.estimate_period_s
                        next_estimate_s = self._clock() + period_s
                    case _CampaignJob(layout, campaign):
                        campaign.uuid = save_campaign_resource(layout, label=campaign.label)
                    case _FinishJob(session, label, angle_deg):
                        self._finish(session, label, angle_deg)
                    case _StopJob():
                        return
            except Exception:
                self._num_worker_errors += 1
                logger.exception('The worker failed on one job; continuing.')

    def _estimate(self, session: ScanSession, label: str, angle_deg: float) -> Product | None:
        configuration = session.setup.configuration
        metadata = configuration.create_metadata(label, tomography_angle_deg=angle_deg)
        snapshot = session.snapshot(metadata)

        if snapshot is None:
            return None

        began_s = self._clock()
        product = estimate_product(snapshot, session.setup, self._rng)
        self._last_estimate_duration_s = self._clock() - began_s
        self._last_estimate_s = self._clock()
        self._num_estimates += 1
        return product

    def _estimate_periodically(self, session: ScanSession) -> None:
        configuration = session.setup.configuration

        try:
            product = self._estimate(
                session, 'scan in progress', configuration.tomography_angle_deg
            )
        except Exception:
            self._num_worker_errors += 1
            logger.exception('The periodic estimate failed.')
            return

        # A scan that ended while it was being estimated has its own final estimate.
        if product is not None and session is self._session:
            self._publish(lambda: self._publisher.publish_estimate(product))

    def _publish(self, publish: Callable[[], None]) -> None:
        # Publishing is for display; its failure must not cost a scan its storage.
        try:
            publish()
        except Exception:
            self._num_worker_errors += 1
            logger.exception('Publishing failed.')

    def _finish(self, session: ScanSession, label: str, angle_deg: float) -> None:
        product: Product | None = None

        try:
            product = self._estimate(session, label, angle_deg)
        except Exception:
            self._num_worker_errors += 1
            logger.exception(f'The final estimate of "{label}" failed; storing its frames only.')

        if product is not None:
            final = product
            self._publish(lambda: self._publisher.publish_estimate(final))

        layout = session.setup.layout
        diffraction_uuid: UUID | None = None
        product_uuid: UUID | None = None
        written = False
        data = session.assemble()

        if layout is not None and data is not None:
            configuration = session.setup.configuration
            campaign = session.campaign

            try:
                diffraction_uuid = save_diffraction_resource(
                    layout,
                    data,
                    label=label,
                    campaign=None if campaign is None else campaign.uuid,
                    detector_distance_m=configuration.detector_distance_m,
                    photon_energy_eV=configuration.photon_energy_eV,
                    exposure_time_s=configuration.exposure_time_s,
                    tomography_angle_deg=angle_deg,
                    crop_center_x_px=configuration.crop_center_x_px,
                    crop_center_y_px=configuration.crop_center_y_px,
                )

                if product is not None:
                    product_uuid = save_product_resource(
                        layout, product, derived_from=[diffraction_uuid]
                    )

                written = True
                self._num_scans_stored += 1
                logger.info(f'Stored "{label}" as diffraction {diffraction_uuid}.')
            except Exception:
                self._num_worker_errors += 1
                logger.exception(f'Failed to store "{label}"!')

        report = ScanReport(
            label=label,
            diffraction_uuid=diffraction_uuid,
            product_uuid=product_uuid,
            num_frames_paired=session.num_frames_paired,
            num_frames_dropped=session.num_frames_dropped,
            written=written,
        )
        self._publish(lambda: self._publisher.publish_scan(report))
