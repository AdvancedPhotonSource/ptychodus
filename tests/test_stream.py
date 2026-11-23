"""The pvapy-free core of ptychodus-stream: configuration, contract, scans, storage, pipeline.

The scans are the far-field disk-probe simulations of ``parallax_scan_fixtures``; a
streamed scan must give exactly what the same patterns give in one batch pass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import cache
from pathlib import Path
import itertools
import json
import logging
import threading
import time

import numpy
import numpy.testing
import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.assemble import AssembledDiffractionData, assemble_dataset
from ptychodus.api.diffraction import (
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import load_diffraction_data, load_product
from ptychodus.api.defocus import AutofocusEstimate, ParallaxEstimate
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
    TrainOutput,
)
from ptychodus.cli._stream import (
    Axis,
    Frame,
    PositionSchema,
    PositionUpdate,
    ProbePositionStream,
    ScanPipeline,
    ScanReport,
    ScanSession,
    ScanSetup,
    ScanSnapshot,
    StreamConfiguration,
    StreamReconstruction,
    display_images,
    estimate_product,
    pack_batched_axes,
    pack_batched_positions,
    pack_sgsocket_attributes,
    parse_batched_axes,
    parse_sgsocket_event,
)
from ptychodus.api.constants import PI
from ptychodus.api.storage import (
    DerivedFromRef,
    DiffractionManifest,
    ProductManifest,
    ResourceKind,
    StorageLayout,
    load_manifest,
    resolve_storage_inputs,
)

from parallax_scan_fixtures import METADATA, make_positions, simulate_scan


@cache
def _registry() -> PluginRegistry:
    return PluginRegistry.load_plugins()


def _scan() -> AssembledDiffractionData:
    return simulate_scan(0.0)


def _configuration(**overrides: object) -> StreamConfiguration:
    pitch_m = _scan().get_pixel_geometry().width_m
    values: dict[str, object] = {
        'input_channel': 'det:image',
        'output_pv_prefix': 'ptycho',
        'trigger_pv': 'scan:done',
        'position_channel': 'scan:positions',
        'photon_energy_eV': METADATA.photon_energy_eV,
        'detector_distance_m': METADATA.detector_distance_m,
        'detector_pixel_size_m': pitch_m,
        'parallax_bin_size_px': 2,
        'estimate_period_s': 3600.0,
    }
    values.update(overrides)
    return StreamConfiguration.model_validate(values)


def _setup(**overrides: object) -> ScanSetup:
    return ScanSetup.from_configuration(_configuration(**overrides), _registry())


def _frames() -> tuple[numpy.ndarray, numpy.ndarray]:
    data = _scan()
    return numpy.array(data.get_indexes()), numpy.array(data.get_patterns())


def _position_updates() -> list[PositionUpdate]:
    positions = ProbePositionSequence(make_positions())
    indexes = numpy.array(positions.get_indexes())
    return [
        PositionUpdate(Axis.X, indexes, numpy.array(positions.get_coordinates_x_m())),
        PositionUpdate(Axis.Y, indexes, numpy.array(positions.get_coordinates_y_m())),
    ]


def _sorted(moments: DiffractionMoments) -> tuple[numpy.ndarray, ...]:
    order = numpy.argsort(moments.indexes)
    return (
        moments.indexes[order],
        moments.detector_sums[order],
        moments.center_of_mass_x_px[order],
        moments.center_of_mass_y_px[order],
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_configuration_round_trips_through_yaml_and_json(tmp_path: Path) -> None:
    configuration = _configuration(
        crop_center_x_px=32,
        crop_center_y_px=32,
        crop_width_px=48,
        crop_height_px=48,
        pipeline={'steps': [{'type': 'binning', 'bin_size_x': 2, 'bin_size_y': 2}]},
        x_axis={'name': 'SampleX', 'scale_m': -1e-6},
        output_directory=str(tmp_path),
    )
    yaml_file = tmp_path / 'scan.yaml'
    json_file = tmp_path / 'scan.json'
    yaml_file.write_text(yaml.safe_dump(configuration.model_dump(mode='json')))
    json_file.write_text(json.dumps(configuration.model_dump(mode='json')))

    assert StreamConfiguration.from_file(yaml_file) == configuration
    assert StreamConfiguration.from_file(json_file) == configuration
    assert configuration.pipeline.steps == (BinningStep(bin_size_x=2, bin_size_y=2),)


def test_merged_applies_a_partial_update_and_rejects_an_invalid_one() -> None:
    configuration = _configuration()

    assert configuration.merged({'settle_s': 5.0}).settle_s == 5.0
    assert configuration.merged({'settle_s': 5.0}).input_channel == 'det:image'

    with pytest.raises(ValidationError):
        configuration.merged({'settle_s': -1.0})

    with pytest.raises(ValidationError):
        configuration.merged({'no_such_key': 1})

    with pytest.raises(ValidationError, match='crop'):
        configuration.merged({'crop_center_x_px': 3})


def test_merged_merges_reconstruction_options_until_the_algorithm_changes() -> None:
    configuration = _configuration(
        reconstruction={
            'library': 'Fake',
            'algorithm': 'Echo',
            'options': {'num_epochs': 4, 'probe': {'enabled': True, 'step': 0.5}},
        }
    )

    retuned = configuration.merged({'reconstruction': {'options': {'probe': {'step': 0.1}}}})
    assert retuned.reconstruction is not None
    assert retuned.reconstruction.library == 'Fake'
    assert retuned.reconstruction.algorithm == 'Echo'
    assert retuned.reconstruction.options == {
        'num_epochs': 4,
        'probe': {'enabled': True, 'step': 0.1},
    }

    switched = configuration.merged({'reconstruction': {'algorithm': 'Other'}})
    assert switched.reconstruction is not None
    assert switched.reconstruction.library == 'Fake'
    assert switched.reconstruction.options == {}

    assert configuration.merged({'reconstruction': None}).reconstruction is None


def test_only_reconstruction_keys_leave_the_scan_alone() -> None:
    configuration = _configuration()
    reconstructing = configuration.merged(
        {'reconstruction': {'library': 'Fake'}, 'trained_model_file': '/models/model.zip'}
    )
    retimed = configuration.merged({'settle_s': 5.0})

    assert not configuration.differs_for_scans(reconstructing)
    assert configuration.differs_for_reconstruction(reconstructing)
    assert configuration.differs_for_scans(retimed)
    assert not configuration.differs_for_reconstruction(retimed)


class _EchoOptions(BaseModel):
    num_epochs: int = Field(2, ge=1)


class _Echo(Reconstructor):
    """Yields its initial guess once per epoch; holds before an epoch on cue."""

    def __init__(self) -> None:
        self.hold_before: int | None = None
        self.holding = threading.Event()
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return 'Echo'

    def get_default_options(self) -> BaseModel:
        return _EchoOptions()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        assert isinstance(options, _EchoOptions)

        for epoch in range(1, options.num_epochs + 1):
            if epoch == self.hold_before and not self.holding.is_set():
                self.holding.set()
                self.release.wait(timeout=60.0)

            yield ReconstructOutput(
                input_.product, progress=epoch, progress_goal=options.num_epochs
            )


class _Trained(TrainableReconstructor):
    def __init__(self) -> None:
        self.model_file: Path | None = None

    @property
    def name(self) -> str:
        return 'Trained'

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        yield ReconstructOutput(input_.product)

    def is_model_loaded(self) -> bool:
        return self.model_file is not None

    def get_model_file_description(self) -> str:
        return 'Fake Model'

    def get_model_file_extensions(self) -> tuple[str, ...]:
        return ('.zip',)

    def load_model(self, file_path: Path) -> None:
        self.model_file = file_path

    def save_model(self, file_path: Path) -> None:
        raise NotImplementedError

    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        yield from ()


class _FakeLibrary(ReconstructorLibrary):
    def __init__(self, reconstructors: Iterable[Reconstructor]) -> None:
        super().__init__('fake', 'fake-distribution')
        self._offered = list(reconstructors)

    def _build(self) -> Iterable[Reconstructor]:
        return self._offered


def _fake_registry(*reconstructors: Reconstructor) -> PluginRegistry:
    registry = PluginRegistry()
    registry.reconstructor_libraries.register_plugin(
        _FakeLibrary(reconstructors), simple_name='fake', display_name='Fake'
    )
    registry.reconstructor_libraries.freeze()
    return registry


def test_a_reconstruction_resolves_with_its_options_applied() -> None:
    registry = _fake_registry(_Echo())
    configuration = _configuration(
        reconstruction={'library': 'Fake', 'algorithm': 'Echo', 'options': {'num_epochs': 5}}
    )

    reconstruction = StreamReconstruction.from_configuration(configuration, registry)

    assert reconstruction is not None
    assert reconstruction.name == 'Fake Echo'
    assert reconstruction.options == _EchoOptions(num_epochs=5)
    assert StreamReconstruction.from_configuration(_configuration(), registry) is None


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'reconstruction': {'library': 'Missing'}}, 'Missing'),
        ({'reconstruction': {'library': 'Fake', 'options': {'no_such': 1}}}, 'no_such'),
        ({'reconstruction': {'library': 'Fake', 'options': {'num_epochs': 0}}}, 'num_epochs'),
        ({'trained_model_file': '/models/model.zip'}, 'no reconstruction'),
        (
            {'reconstruction': {'library': 'Fake'}, 'trained_model_file': '/models/model.zip'},
            'takes no trained model',
        ),
    ],
)
def test_a_reconstruction_that_cannot_run_is_refused(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        StreamReconstruction.from_configuration(
            _configuration(**overrides), _fake_registry(_Echo())
        )


def test_a_trained_model_is_loaded_into_a_copy() -> None:
    trained = _Trained()
    registry = _fake_registry(trained)

    with pytest.raises(ValueError, match='set trained_model_file'):
        StreamReconstruction.from_configuration(
            _configuration(reconstruction={'library': 'Fake'}), registry
        )

    reconstruction = StreamReconstruction.from_configuration(
        _configuration(reconstruction={'library': 'Fake'}, trained_model_file='/m/model.zip'),
        registry,
    )

    assert reconstruction is not None
    assert isinstance(reconstruction.reconstructor, _Trained)
    assert reconstruction.reconstructor.model_file == Path('/m/model.zip')
    assert trained.model_file is None


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_batched_axes_round_trip_with_scale_and_offset() -> None:
    configuration = _configuration(
        x_axis={'name': 'X', 'scale_m': -1e-6},
        y_axis={'name': 'Y', 'scale_m': 1e-6},
        i0_name='I0',
        index_offset=-1,
    )
    value = pack_batched_axes(
        {'X': ([1, 2, 3], [5.0, 6.0, 7.0]), 'Y': ([1, 2], [8.0, 9.0]), 'I0': ([3], [1e5])}
    )
    updates = {update.axis: update for update in parse_batched_axes(value, configuration)}

    numpy.testing.assert_array_equal(updates[Axis.X].indexes, [0, 1, 2])
    numpy.testing.assert_allclose(updates[Axis.X].values, [-5e-6, -6e-6, -7e-6])
    numpy.testing.assert_allclose(updates[Axis.Y].values, [8e-6, 9e-6])
    numpy.testing.assert_array_equal(updates[Axis.I0].indexes, [2])
    assert value['X']['num_events'] == 3


def test_a_packed_batch_of_positions_parses_back_to_metres() -> None:
    configuration = _configuration(
        x_axis={'name': 'X', 'scale_m': 1e-6}, y_axis={'name': 'Y', 'scale_m': -1e-6}
    )
    value = pack_batched_positions(configuration, [4, 5], [1e-6, 2e-6], [3e-6, 4e-6])
    updates = {update.axis: update for update in parse_batched_axes(value, configuration)}

    numpy.testing.assert_allclose(value['X']['positions'], [1.0, 2.0])
    numpy.testing.assert_allclose(updates[Axis.X].values, [1e-6, 2e-6])
    numpy.testing.assert_allclose(updates[Axis.Y].values, [3e-6, 4e-6])
    numpy.testing.assert_array_equal(updates[Axis.Y].indexes, [4, 5])


def test_an_axis_missing_from_a_batch_is_skipped() -> None:
    updates = parse_batched_axes(pack_batched_axes({'x': ([1], [0.5])}), _configuration())

    assert [update.axis for update in updates] == [Axis.X]


def test_sgsocket_events_round_trip_through_pvapys_attribute_form() -> None:
    configuration = _configuration(
        position_schema=PositionSchema.SGSOCKET,
        x_axis={'name': 'SampleX', 'scale_m': 1e-6},
        y_axis={'name': 'SampleY', 'scale_m': -1e-6},
        i0_name='I0',
        index_offset=10,
    )
    attributes = pack_sgsocket_attributes(configuration, x_m=2e-6, y_m=3e-6, i0=40.0)
    # pvapy hands an NTAttribute's variant value as ({'value': x}, {'value': type}).
    value = {
        'uniqueId': 5,
        'attribute': [
            {'name': name, 'value': ({'value': reading}, {'value': 'DOUBLE'})}
            for name, reading in attributes.items()
        ],
    }
    updates = {update.axis: update for update in parse_sgsocket_event(value, configuration)}

    assert attributes == pytest.approx({'SampleX': 2.0, 'SampleY': -3.0, 'I0': 40.0})
    numpy.testing.assert_array_equal(updates[Axis.X].indexes, [15])
    numpy.testing.assert_allclose(updates[Axis.X].values, [2e-6])
    numpy.testing.assert_allclose(updates[Axis.Y].values, [3e-6])
    numpy.testing.assert_allclose(updates[Axis.I0].values, [40.0])


def test_display_images_are_the_first_modes_amplitude_and_phase() -> None:
    array = numpy.stack([numpy.full((2, 3), 2j), numpy.full((2, 3), 5.0 + 0j)])
    amplitude, phase = display_images(array)

    assert amplitude.dtype == numpy.float32 and phase.dtype == numpy.float32
    numpy.testing.assert_allclose(amplitude, 2.0)
    numpy.testing.assert_allclose(phase, PI / 2)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_axes_at_different_rates_interpolate_independently() -> None:
    stream = ProbePositionStream([Axis.X, Axis.Y])
    stream.append(PositionUpdate(Axis.X, numpy.array([0, 2, 4]), numpy.array([0.0, 2.0, 4.0])), 0)
    stream.append(PositionUpdate(Axis.Y, numpy.array([1, 3]), numpy.array([10.0, 30.0])), 0)

    covered, positions = stream.at(numpy.array([0, 1, 2, 3, 4]))

    assert stream.coverage() == (1, 3)
    numpy.testing.assert_array_equal(covered, [False, True, True, True, False])
    numpy.testing.assert_allclose(positions.get_coordinates_x_m(), [1.0, 2.0, 3.0])
    numpy.testing.assert_allclose(positions.get_coordinates_y_m(), [10.0, 20.0, 30.0])
    assert positions.get_i0() is None


def test_i0_is_left_out_when_it_does_not_cover_every_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = ProbePositionStream([Axis.X, Axis.Y, Axis.I0])
    indexes = numpy.arange(4)
    stream.append(PositionUpdate(Axis.X, indexes, numpy.zeros(4)), 0)
    stream.append(PositionUpdate(Axis.Y, indexes, numpy.zeros(4)), 0)
    stream.append(PositionUpdate(Axis.I0, indexes[:2], numpy.ones(2)), 0)

    with caplog.at_level(logging.WARNING):
        _, partial = stream.at(indexes)

    _, covered = stream.at(indexes[:2])

    assert partial.get_i0() is None
    assert any('I0' in record.getMessage() for record in caplog.records)
    counts = covered.get_i0()
    assert counts is not None
    numpy.testing.assert_allclose(counts, [1.0, 1.0])


def test_pruning_keeps_what_frames_still_need() -> None:
    stream = ProbePositionStream([Axis.X, Axis.Y])

    for axis in (Axis.X, Axis.Y):
        stream.append(PositionUpdate(axis, numpy.arange(10), numpy.arange(10.0)), 0.0)

    stream.prune(older_than_s=1.0, frame_index_range=(4, 6))

    covered, positions = stream.at(numpy.arange(10))
    numpy.testing.assert_array_equal(numpy.flatnonzero(covered), [3, 4, 5, 6, 7])
    numpy.testing.assert_allclose(positions.get_coordinates_x_m(), [3.0, 4.0, 5.0, 6.0, 7.0])

    stream.prune(older_than_s=1.0, frame_index_range=None)
    assert stream.coverage() is None


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------


def _streamed_session(setup: ScanSetup, *, seed: int = 4) -> ScanSession:
    """A session fed every frame in shuffled chunks, positions first."""
    indexes, patterns = _frames()
    session = ScanSession(setup)

    for update in _position_updates():
        session.add_positions(update, 0.0)

    order = numpy.random.default_rng(seed).permutation(indexes.size)

    for chunk in numpy.array_split(order, 7):
        session.add_frames([_frame(int(indexes[k]), patterns[k]) for k in chunk])
        session.promote(0.0)

    return session


def _frame(index: int, image: numpy.ndarray, arrival_s: float = 0.0) -> Frame:
    return Frame(index, image, arrival_s)


def test_a_streamed_scan_equals_one_batch_pass() -> None:
    setup = _setup()
    configuration = setup.configuration
    session = _streamed_session(setup)
    indexes, patterns = _frames()
    metadata = configuration.create_metadata('scan', tomography_angle_deg=0.0)
    streamed = session.snapshot(metadata)
    batch_moments = compute_diffraction_moments(
        indexes,
        patterns,
        bad_pixels=numpy.array(_scan().get_bad_pixels()),
        binning=BinningStep(bin_size_x=2, bin_size_y=2),
    )

    assert streamed is not None and batch_moments is not None

    for got, want in zip(_sorted(streamed.get_diffraction_moments()), _sorted(batch_moments)):
        numpy.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    numpy.testing.assert_allclose(
        streamed.get_diffraction_moments().mean_pattern, batch_moments.mean_pattern, rtol=1e-12
    )

    batch = ScanSnapshot(
        metadata,
        ImageExtent(width_px=patterns.shape[2], height_px=patterns.shape[1]),
        _scan().get_pixel_geometry(),
        ProbePositionSequence(make_positions()),
        batch_moments,
    )
    from_stream = estimate_product(streamed, setup)
    from_batch = estimate_product(batch, setup)

    numpy.testing.assert_allclose(
        from_stream.probes.get_array(), from_batch.probes.get_array(), rtol=1e-9, atol=1e-9
    )
    numpy.testing.assert_allclose(
        from_stream.object_.get_array(), from_batch.object_.get_array(), rtol=1e-9, atol=1e-9
    )
    assert from_stream.probes.get_geometry() == from_batch.probes.get_geometry()


def test_frames_pair_once_covered_and_expire_when_never_covered() -> None:
    indexes, patterns = _frames()
    session = ScanSession(_setup(unpaired_retention_s=10.0))
    session.add_frames([_frame(int(indexes[k]), patterns[k], 0.0) for k in range(20)])

    assert session.promote(1.0) == 0
    assert session.num_frames_pending == 20

    x = numpy.arange(10)
    session.add_positions(PositionUpdate(Axis.X, x, numpy.zeros(10)), 1.0)
    session.add_positions(PositionUpdate(Axis.Y, x, numpy.zeros(10)), 1.0)

    assert session.promote(2.0) == 10
    assert session.num_frames_pending == 10

    session.promote(30.0)

    assert session.num_frames_paired == 10
    assert session.num_frames_dropped == 10
    assert session.num_frames_pending == 0


def test_assembling_a_scan_equals_assembling_its_paired_frames() -> None:
    setup = _setup(crop_center_x_px=32, crop_center_y_px=32, crop_width_px=48, crop_height_px=48)
    session = _streamed_session(setup)
    indexes, patterns = _frames()
    data = _scan()
    dataset = SimpleDiffractionDataset(
        DiffractionMetadata(
            num_patterns_per_array=[indexes.size],
            pattern_dtype=patterns.dtype,
            detector_extent=ImageExtent(width_px=patterns.shape[2], height_px=patterns.shape[1]),
            detector_pixel_geometry=data.get_pixel_geometry(),
        ),
        DiffractionDatasetLayoutNode.create_root(),
        [SimpleDiffractionArray('all', indexes, patterns)],
        numpy.array(data.get_bad_pixels()),
    )
    expected = assemble_dataset(dataset, read_region=setup.configuration.get_read_region())
    assembled = session.assemble()

    assert assembled is not None
    order = numpy.argsort(assembled.get_indexes())
    numpy.testing.assert_array_equal(
        numpy.asarray(assembled.get_indexes())[order], expected.get_indexes()
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(assembled.get_patterns())[order], expected.get_patterns()
    )
    assert assembled.get_pixel_geometry() == expected.get_pixel_geometry()


def test_a_frame_of_another_shape_or_dtype_is_rejected() -> None:
    indexes, patterns = _frames()
    session = ScanSession(_setup())
    session.add_frames(
        [
            _frame(0, patterns[0]),
            _frame(1, patterns[1][:-1]),
            _frame(2, patterns[2].astype(numpy.float32)),
            _frame(3, patterns[3]),
        ]
    )

    assert session.num_frames_rejected == 2
    assert session.num_frames_pending == 2


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Recorder:
    """Records what the pipeline publishes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def publish_estimate(self, product: Product) -> None:
        with self.lock:
            self.events.append(('estimate', product.metadata.name))

    def publish_scan(self, report: ScanReport) -> None:
        with self.lock:
            self.events.append(('scan', report))

    def publish_reconstruction(self, label: str, output: ReconstructOutput) -> None:
        with self.lock:
            self.events.append(('reconstruction', (label, output.progress)))

    def reports(self) -> list[ScanReport]:
        return [event for kind, event in self.events if isinstance(event, ScanReport)]

    def estimate_names(self) -> list[str]:
        with self.lock:
            return [str(event) for kind, event in self.events if kind == 'estimate']

    def reconstructions(self) -> list[object]:
        with self.lock:
            return [event for kind, event in self.events if kind == 'reconstruction']


class _HoldingRecorder(_Recorder):
    """Holds the worker in its first publish until released, so the worker is busy on cue."""

    def __init__(self) -> None:
        super().__init__()
        self.holding = threading.Event()
        self.release = threading.Event()

    def publish_estimate(self, product: Product) -> None:
        if not self.holding.is_set():
            self.holding.set()
            self.release.wait(timeout=60.0)

        super().publish_estimate(product)


def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 60.0) -> bool:
    """Poll *predicate* until it holds or *timeout_s* passes; return whether it held."""
    deadline_s = time.monotonic() + timeout_s

    while not predicate():
        if time.monotonic() > deadline_s:
            return False

        time.sleep(0.005)

    return True


def _submit_scan(
    pipeline: ScanPipeline, frame_range: range, *, offset: int = 0, positions: bool = True
) -> None:
    indexes, patterns = _frames()

    if positions:
        for update in _position_updates():
            keep = numpy.isin(update.indexes, numpy.array(frame_range))
            pipeline.submit_positions(
                [PositionUpdate(update.axis, update.indexes[keep] + offset, update.values[keep])]
            )

    for k in frame_range:
        pipeline.submit_frame(int(indexes[k]) + offset, patterns[k])


def _pipeline(
    recorder: _Recorder, clock: _Clock | None = None, **overrides: object
) -> tuple[ScanPipeline, list[str]]:
    acknowledged: list[str] = []
    overrides.setdefault('settle_s', 1.0)
    pipeline = ScanPipeline(
        _setup(**overrides),
        recorder,
        acknowledge_trigger=lambda: acknowledged.append('ack'),
        clock=time.monotonic if clock is None else clock,
        idle_s=0.01,
    )
    return pipeline, acknowledged


def test_the_settle_is_decided_by_arrival_even_when_ingest_is_backlogged() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline, acknowledged = _pipeline(recorder, clock)

    # Everything is queued before ingest starts, so ingest is as backlogged as it gets.
    _submit_scan(pipeline, range(0, 100))
    clock.now = 1.0
    pipeline.trigger()  # the settle ends at 2.0
    clock.now = 1.5
    _submit_scan(pipeline, range(100, 150))
    clock.now = 3.0
    _submit_scan(pipeline, range(150, 361))
    pipeline.start()
    pipeline.stop(timeout_s=120.0)

    first, second = recorder.reports()
    assert acknowledged == ['ack']
    assert first.num_frames_paired == 150
    assert second.num_frames_paired == 211
    assert pipeline.get_counters()['num_frames_queued'] == 0


def test_a_trigger_during_a_write_is_not_lost(tmp_path: Path) -> None:
    """The second scan and its trigger arrive while the worker is held finishing the first."""
    recorder = _HoldingRecorder()
    pipeline, acknowledged = _pipeline(recorder, output_directory=str(tmp_path), settle_s=0.1)
    pipeline.start()
    _submit_scan(pipeline, range(0, 361))
    pipeline.trigger()
    assert recorder.holding.wait(timeout=60.0), 'the first scan never reached the worker'
    _submit_scan(pipeline, range(0, 361), offset=1000)
    pipeline.trigger()
    recorder.release.set()
    pipeline.stop(timeout_s=120.0)

    reports = recorder.reports()
    assert acknowledged == ['ack', 'ack']
    assert [report.num_frames_paired for report in reports] == [361, 361]
    assert all(report.written and report.product_uuid is not None for report in reports)
    assert len(list((tmp_path / 'product').iterdir())) == 2


def test_an_empty_trigger_and_a_trigger_while_settling_change_nothing() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline, acknowledged = _pipeline(recorder, clock)
    pipeline.trigger()
    _submit_scan(pipeline, range(0, 50))
    clock.now = 1.0
    pipeline.trigger()
    clock.now = 1.5
    pipeline.trigger()
    pipeline.start()
    pipeline.stop(timeout_s=120.0)

    assert acknowledged == ['ack', 'ack', 'ack']
    assert [report.num_frames_paired for report in recorder.reports()] == [50]


def test_frames_still_unpaired_at_the_cut_are_dropped_and_counted() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder, clock)
    _submit_scan(pipeline, range(0, 50))
    _submit_scan(pipeline, range(50, 60), positions=False)
    pipeline.trigger()
    clock.now = 5.0
    pipeline.start()
    # The settle has expired, so ingest cuts on its own; a stop would cut too.
    assert _wait_until(lambda: pipeline.get_counters()['num_frames_dropped'] > 0)
    pipeline.stop(timeout_s=120.0)

    (report,) = recorder.reports()
    assert report.num_frames_paired == 50
    assert report.num_frames_dropped == 10


def test_a_mid_scan_configure_discards_and_one_while_settling_waits() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder, clock)
    _submit_scan(pipeline, range(0, 30))
    pipeline.configure(_setup(settle_s=1.0, campaign_label='second'))
    _submit_scan(pipeline, range(30, 80))
    pipeline.trigger()
    pipeline.configure(_setup(settle_s=1.0, campaign_label='third'))
    _submit_scan(pipeline, range(80, 90))  # still within the settle
    clock.now = 5.0
    _submit_scan(pipeline, range(90, 100))
    pipeline.start()
    pipeline.stop(timeout_s=120.0)

    counters = pipeline.get_counters()
    assert counters['num_frames_discarded'] == 30
    assert [report.num_frames_paired for report in recorder.reports()] == [60, 10]


def test_free_running_frames_between_scans_never_accumulate() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder, clock, unpaired_retention_s=1.0)
    _submit_scan(pipeline, range(0, 40), positions=False)  # no positions: free-running
    clock.now = 10.0
    _submit_scan(pipeline, range(40, 100), offset=0)
    pipeline.start()
    pipeline.stop(timeout_s=120.0)

    (report,) = recorder.reports()
    assert report.num_frames_paired == 60
    assert pipeline.get_counters()['num_frames_dropped'] == 40


def test_publishes_are_ordered_and_failures_are_counted(tmp_path: Path) -> None:
    class _Failing(_Recorder):
        def publish_estimate(self, product: Product) -> None:
            super().publish_estimate(product)
            raise RuntimeError('display server down')

    recorder = _Failing()
    pipeline, _ = _pipeline(
        recorder, output_directory=str(tmp_path), estimate_period_s=0.05, settle_s=0.1
    )
    pipeline.start()
    _submit_scan(pipeline, range(0, 361))
    # Order means something only once an in-progress estimate precedes the final one.
    assert _wait_until(lambda: 'scan in progress' in recorder.estimate_names())
    pipeline.trigger()
    pipeline.stop(timeout_s=120.0)

    names = recorder.estimate_names()
    scan_at = next(k for k, (kind, _) in enumerate(recorder.events) if kind == 'scan')
    assert names[-1].startswith('scan ') and names[-1] != 'scan in progress'
    assert all(kind != 'estimate' for kind, _ in recorder.events[scan_at:])
    assert pipeline.get_counters()['num_worker_errors'] >= 1
    assert recorder.reports()[0].written


def test_stop_finishes_and_stores_the_scan_in_progress(tmp_path: Path) -> None:
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder, output_directory=str(tmp_path), campaign_label='demo')
    pipeline.start()
    _submit_scan(pipeline, range(0, 100))
    pipeline.stop(timeout_s=120.0)

    (report,) = recorder.reports()
    assert report.written and report.num_frames_paired == 100
    assert report.diffraction_uuid is not None and report.product_uuid is not None
    layout = StorageLayout(tmp_path)
    diffraction = load_manifest(
        layout.manifest_path(ResourceKind.DIFFRACTION, report.diffraction_uuid),
        expected_kind=ResourceKind.DIFFRACTION,
        expected_uuid=report.diffraction_uuid,
    )
    product = load_manifest(
        layout.manifest_path(ResourceKind.PRODUCT, report.product_uuid),
        expected_kind=ResourceKind.PRODUCT,
        expected_uuid=report.product_uuid,
    )
    (campaign_folder,) = (tmp_path / ResourceKind.CAMPAIGN).iterdir()
    assert isinstance(diffraction, DiffractionManifest)
    assert isinstance(product, ProductManifest)
    assert str(diffraction.campaign_uuid) == campaign_folder.name
    assert product.derived_from == [
        DerivedFromRef(kind='diffraction', uuid=report.diffraction_uuid)
    ]

    for manifest in (diffraction, product):
        assert manifest.run is not None
        assert manifest.run.command == 'ptychodus-stream'
        assert manifest.run.state == 'succeeded'
        assert manifest.run.options is None

    numpy.testing.assert_array_equal(
        load_diffraction_data(resolve_storage_inputs(layout, report.product_uuid).diffraction_file)
        .get_indexes()
        .size,
        100,
    )


def test_a_failed_estimate_stores_the_diffraction_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> Product:
        raise RuntimeError('no estimate')

    monkeypatch.setattr('ptychodus.cli._stream.estimate_product', fail)
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder, output_directory=str(tmp_path))
    pipeline.start()
    _submit_scan(pipeline, range(0, 50))
    pipeline.stop(timeout_s=120.0)

    (report,) = recorder.reports()
    assert report.written and report.diffraction_uuid is not None
    assert report.product_uuid is None
    assert not any((tmp_path / ResourceKind.PRODUCT).iterdir())
    assert pipeline.get_counters()['num_worker_errors'] >= 1


def test_the_trigger_takes_the_scan_number_and_angle_and_survives_a_failed_ack(
    tmp_path: Path,
) -> None:
    def failing_ack() -> None:
        raise TimeoutError('trigger PV unreachable')

    recorder = _Recorder()
    pipeline = ScanPipeline(
        _setup(output_directory=str(tmp_path), settle_s=0.05),
        recorder,
        acknowledge_trigger=failing_ack,
        read_scan_number=lambda: 42,
        read_tomography_angle_deg=lambda: 37.5,
        idle_s=0.01,
    )
    pipeline.start()
    _submit_scan(pipeline, range(0, 361))
    pipeline.trigger()
    pipeline.stop(timeout_s=120.0)

    (report,) = recorder.reports()
    assert report.label == 'scan 42' and report.written
    assert pipeline.get_counters()['num_trigger_failures'] == 1
    assert report.product_uuid is not None and report.diffraction_uuid is not None
    layout = StorageLayout(tmp_path)
    product_file = resolve_storage_inputs(layout, report.product_uuid).product_file
    product = load_product(product_file)
    assert product.metadata.name == 'scan 42'
    assert product.metadata.tomography_angle_deg == 37.5
    diffraction = load_manifest(
        layout.manifest_path(ResourceKind.DIFFRACTION, report.diffraction_uuid),
        expected_kind=ResourceKind.DIFFRACTION,
        expected_uuid=report.diffraction_uuid,
    )
    assert isinstance(diffraction, DiffractionManifest)
    assert diffraction.tomography_angle_deg == 37.5


def test_positions_before_the_first_frame_are_kept_and_index_anomalies_counted() -> None:
    recorder = _Recorder()
    pipeline, _ = _pipeline(recorder)
    indexes, patterns = _frames()
    pipeline.submit_positions(_position_updates())
    pipeline.start()

    for k in (5, 6, 6, 3, 7):
        pipeline.submit_frame(int(indexes[k]), patterns[k])

    pipeline.stop(timeout_s=120.0)

    counters = pipeline.get_counters()
    assert counters['num_frames_paired'] == 5
    assert counters['num_frames_duplicated'] == 1
    assert counters['num_frames_regressed'] == 1


def test_the_parallax_is_not_measured_in_near_field(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _setup()
    session = _streamed_session(setup)
    metadata = setup.configuration.create_metadata('scan', tomography_angle_deg=0.0)
    snapshot = session.snapshot(replace(metadata, far_field=False))
    assert snapshot is not None

    def fail(*args: object) -> None:
        raise AssertionError('The parallax answers in far field only.')

    monkeypatch.setattr('ptychodus.cli._stream._estimate_parallax', fail)

    product = estimate_product(snapshot, setup)

    assert product.metadata.focus_object_distance_m == 0.0


def test_an_explicit_zero_focus_skips_the_parallax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero stated is a statement; only an unset focus is measured."""
    setup = _setup(focus_object_distance_m=0.0)
    session = _streamed_session(setup)
    snapshot = session.snapshot(
        setup.configuration.create_metadata('scan', tomography_angle_deg=0.0)
    )
    assert snapshot is not None

    def fail(*args: object) -> None:
        raise AssertionError('A stated focus is not measured.')

    monkeypatch.setattr('ptychodus.cli._stream._estimate_parallax', fail)

    product = estimate_product(snapshot, setup)

    assert product.metadata.focus_object_distance_m == 0.0


def test_an_unset_focus_is_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _setup()
    assert setup.configuration.focus_object_distance_m is None
    session = _streamed_session(setup)
    snapshot = session.snapshot(
        setup.configuration.create_metadata('scan', tomography_angle_deg=0.0)
    )
    assert snapshot is not None
    calls: list[object] = []

    def record(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr('ptychodus.cli._stream._estimate_parallax', record)

    estimate_product(snapshot, setup)

    assert len(calls) == 1


_RESOLVED_PARALLAX = ParallaxEstimate(
    defocus_distance_m=2.0e-3, standard_error_m=1.0e-5, depth_of_focus_m=1.0e-4, num_bins=9
)


def _estimate_with_autofocus(
    monkeypatch: pytest.MonkeyPatch, *, final: bool, **overrides: object
) -> Product:
    """estimate_product with a resolved parallax and autofocus answering 2.6 mm."""
    setup = _setup(**overrides)
    session = _streamed_session(setup)
    snapshot = session.snapshot(
        setup.configuration.create_metadata('scan', tomography_angle_deg=0.0)
    )
    assert snapshot is not None
    refined = AutofocusEstimate(
        defocus_distance_m=2.6e-3,
        candidates_m=(),
        residuals=(),
        num_patterns=1,
        kept_initial=False,
        on_edge=False,
    )
    monkeypatch.setattr(
        'ptychodus.cli._stream._estimate_parallax', lambda *args: _RESOLVED_PARALLAX
    )
    monkeypatch.setattr(
        'ptychodus.cli._stream.estimate_defocus_by_autofocus', lambda *args, **kwargs: refined
    )
    data = session.assemble() if final else None
    return estimate_product(snapshot, setup, assembled_data=data)


def test_the_final_estimate_refines_the_defocus_by_autofocus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _estimate_with_autofocus(monkeypatch, final=True)

    assert product.metadata.focus_object_distance_m == pytest.approx(-2.6e-3)


def test_a_periodic_estimate_uses_the_parallax_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Periodic estimates see moments only, which autofocus cannot use."""
    product = _estimate_with_autofocus(monkeypatch, final=False)

    assert product.metadata.focus_object_distance_m == pytest.approx(-2.0e-3)


def test_the_parallax_setting_skips_autofocus(monkeypatch: pytest.MonkeyPatch) -> None:
    product = _estimate_with_autofocus(monkeypatch, final=True, defocus_estimate='parallax')

    assert product.metadata.focus_object_distance_m == pytest.approx(-2.0e-3)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _echo_reconstruction(echo: _Echo) -> StreamReconstruction:
    return StreamReconstruction('Fake Echo', echo, _EchoOptions())


def _numbered_pipeline(
    recorder: _Recorder, clock: _Clock, reconstruction: StreamReconstruction | None
) -> ScanPipeline:
    return ScanPipeline(
        _setup(settle_s=1.0),
        recorder,
        reconstruction=reconstruction,
        read_scan_number=itertools.count(1).__next__,
        clock=clock,
        idle_s=0.01,
    )


def test_each_finished_scan_is_reconstructed_and_the_newest_waiting_one_wins() -> None:
    echo = _Echo()
    echo.hold_before = 1
    clock = _Clock()
    recorder = _Recorder()
    pipeline = _numbered_pipeline(recorder, clock, _echo_reconstruction(echo))

    for k in range(3):
        clock.now = 3.0 * k
        _submit_scan(pipeline, range(0, 100), offset=1000 * k)
        clock.now += 1.0
        pipeline.trigger()

    clock.now = 10.0
    pipeline.start()
    # The first reconstruction is held while all three scans finish behind it.
    assert _wait_until(lambda: len(recorder.reports()) == 3)
    echo.release.set()
    assert _wait_until(lambda: ('scan 3', 2) in recorder.reconstructions())
    pipeline.stop(timeout_s=120.0)

    assert recorder.reconstructions() == [
        ('scan 1', 1),
        ('scan 1', 2),
        ('scan 3', 1),
        ('scan 3', 2),
    ]
    counters = pipeline.get_counters()
    assert counters['num_reconstructions'] == 2
    assert counters['num_reconstructions_dropped'] == 1
    assert counters['num_reconstruction_errors'] == 0


def test_setting_a_reconstruction_mid_scan_keeps_the_scan() -> None:
    clock = _Clock()
    recorder = _Recorder()
    pipeline = _numbered_pipeline(recorder, clock, None)
    _submit_scan(pipeline, range(0, 100))
    pipeline.set_reconstruction(_echo_reconstruction(_Echo()))
    clock.now = 1.0
    pipeline.trigger()
    clock.now = 3.0
    pipeline.start()
    assert _wait_until(lambda: len(recorder.reconstructions()) == 2)
    pipeline.stop(timeout_s=120.0)

    assert [report.num_frames_paired for report in recorder.reports()] == [100]
    assert pipeline.get_counters()['num_frames_discarded'] == 0
    assert recorder.reconstructions() == [('scan 1', 1), ('scan 1', 2)]


def test_stop_abandons_a_reconstruction_at_its_next_output() -> None:
    echo = _Echo()
    echo.hold_before = 2
    clock = _Clock()
    recorder = _Recorder()
    pipeline = _numbered_pipeline(recorder, clock, _echo_reconstruction(echo))
    _submit_scan(pipeline, range(0, 100))
    clock.now = 1.0
    pipeline.trigger()
    clock.now = 3.0
    pipeline.start()
    assert echo.holding.wait(timeout=60.0), 'the reconstruction never started'
    stopping = threading.Thread(target=pipeline.stop, kwargs={'timeout_s': 120.0})
    stopping.start()
    assert _wait_until(lambda: pipeline._is_stopping)
    echo.release.set()
    stopping.join(timeout=120.0)

    assert recorder.reconstructions() == [('scan 1', 1), ('scan 1', 2)]
    assert pipeline.get_counters()['num_reconstructions'] == 0
    assert not pipeline.get_counters()['is_reconstruct_alive']
