"""Unit tests for the generic :class:`SubprocessReconstructor` adapter.

These tests spawn real subprocesses (so they exercise the pickling, log
forwarding, sentinel/error paths, and cleanup) but use fake entry points that
touch no GPU framework. See ``tests/subprocess_child_fixtures.py``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput
from ptychodus.model.processing._subprocess_protocol import ChildError
from ptychodus.model.processing.subprocess_reconstructor import (
    SubprocessReconstructor,
)


# The spawned child resolves the fixtures module by dotted path against the
# sys.path it inherits from this process, so the tests directory must be on it
# before any child is spawned. pytest's default import mode already does this;
# the insert keeps it true under other invocations.
_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

FIXTURES_MODULE = 'subprocess_child_fixtures'


def _minimal_product() -> Product:
    metadata = ProductMetadata(
        name='fake',
        comments='',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
    )
    pixel_geometry = PixelGeometry(1.0e-9, 1.0e-9)
    center = ObjectCenter(0.0, 0.0)
    return Product(
        metadata=metadata,
        probe_positions=ProbePositionSequence([]),
        probes=ProbeSequence(
            array=numpy.zeros((1, 1, 4, 4), dtype=numpy.complex64),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.zeros((1, 4, 4), dtype=numpy.complex64),
            layer_spacing_m=[],
            pixel_geometry=pixel_geometry,
            center=center,
        ),
        losses=[],
    )


def _minimal_reconstruct_input() -> ReconstructInput:
    product = _minimal_product()
    return ReconstructInput(
        diffraction_patterns=numpy.zeros((1, 4, 4), dtype=numpy.float32),
        bad_pixels=numpy.zeros((4, 4), dtype=numpy.bool_),
        product=product,
    )


def test_reconstruct_streams_all_outputs_in_order() -> None:
    product = _minimal_product()

    def build_payload(parameters: ReconstructInput, _loaded: Path | None) -> Any:
        return {'n': 3, 'product': product}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:yield_n_outputs',
        progress_goal_fn=lambda: 3,
        build_reconstruct_payload=build_payload,
    )

    outputs = list(adapter.reconstruct(_minimal_reconstruct_input()))

    assert len(outputs) == 3
    assert [o.progress for o in outputs] == [1, 2, 3]


def test_child_exception_propagates_as_child_error() -> None:
    def build_payload(parameters: ReconstructInput, _loaded: Path | None) -> Any:
        return {'message': 'boom'}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:raise_immediately',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=build_payload,
    )

    with pytest.raises(ChildError) as excinfo:
        list(adapter.reconstruct(_minimal_reconstruct_input()))

    assert excinfo.value.child_exception_type == 'ValueError'
    # The original exception is picklable, so the parent got it back.
    assert isinstance(excinfo.value.child_exception, ValueError)
    assert 'boom' in str(excinfo.value.child_exception)


def test_hanging_child_is_terminated_on_iterator_close() -> None:
    def build_payload(parameters: ReconstructInput, _loaded: Path | None) -> Any:
        return {}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:hang_forever',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=build_payload,
        terminate_grace_sec=1.0,
    )

    # Get one message (there is none coming) - close the iterator to trigger cleanup.
    iterator = adapter.reconstruct(_minimal_reconstruct_input())
    # Close immediately; the context manager's finally must terminate the child.
    # Reconstructor.reconstruct is declared Iterator, but the generator-close path
    # is exactly what this test exercises.
    cast(Generator[ReconstructOutput, None, None], iterator).close()


def test_child_log_is_forwarded_to_parent_logger(caplog: pytest.LogCaptureFixture) -> None:
    product = _minimal_product()

    def build_payload(parameters: ReconstructInput, _loaded: Path | None) -> Any:
        return {'product': product, 'log_message': 'hello from the child'}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:emit_log_then_output',
        progress_goal_fn=lambda: 1,
        build_reconstruct_payload=build_payload,
    )

    with caplog.at_level(logging.WARNING, logger=FIXTURES_MODULE):
        outputs = list(adapter.reconstruct(_minimal_reconstruct_input()))

    assert len(outputs) == 1
    assert any('hello from the child' in rec.message for rec in caplog.records)


def test_settings_sync_message_invokes_callback() -> None:
    product = _minimal_product()
    seen: list[dict[str, dict[str, str]]] = []

    def build_payload(parameters: ReconstructInput, _loaded: Path | None) -> Any:
        return {'product': product, 'settings': {'GroupA': {'p': '42'}}}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:emit_settings_sync_then_output',
        progress_goal_fn=lambda: 1,
        build_reconstruct_payload=build_payload,
        apply_settings_sync=seen.append,
    )

    outputs = list(adapter.reconstruct(_minimal_reconstruct_input()))

    assert len(outputs) == 1
    assert seen == [{'GroupA': {'p': '42'}}]


def test_train_records_model_saved_path(tmp_path: Path) -> None:
    saved_path = tmp_path / 'ckpt.bin'
    saved_path.write_bytes(b'x')

    def build_train_payload(input_path: Path, output_path: Path) -> Any:
        return {'saved_path': str(saved_path)}

    def build_reconstruct_payload(_p: ReconstructInput, _loaded: Path | None) -> Any:
        return {}

    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:hang_forever',
        progress_goal_fn=lambda: 1,
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=f'{FIXTURES_MODULE}:train_and_save',
        build_train_payload=build_train_payload,
        model_file_extension='.bin',
    )

    assert not adapter.is_model_loaded()

    outputs = list(adapter.train(tmp_path, tmp_path))

    assert len(outputs) == 1
    assert outputs[0].progress == 1
    assert adapter.is_model_loaded()

    # save_model must copy the recorded path.
    dest = tmp_path / 'copied.bin'
    adapter.save_model(dest)
    assert dest.read_bytes() == b'x'


def test_non_trainable_train_raises() -> None:
    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:hang_forever',
        progress_goal_fn=lambda: 1,
        build_reconstruct_payload=lambda p, m: {},
    )

    with pytest.raises(NotImplementedError):
        list(adapter.train(Path('/tmp'), Path('/tmp')))


def test_save_without_load_or_train_raises() -> None:
    adapter = SubprocessReconstructor(
        name='FAKE',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:hang_forever',
        progress_goal_fn=lambda: 1,
        build_reconstruct_payload=lambda p, m: {},
        is_trainable=True,
        train_entry_point=f'{FIXTURES_MODULE}:train_and_save',
        build_train_payload=lambda i, o: {},
    )

    with pytest.raises(RuntimeError):
        adapter.save_model(Path('/tmp/x'))
