"""Unit tests for :func:`ptychodus.api.spawn.iterate_in_subprocess`.

These tests spawn real subprocesses, so they exercise the pickling, log
forwarding, sentinel/error paths and cleanup, but the work they run touches no
GPU framework. See ``tests/subprocess_child_fixtures.py`` and
``tests/subprocess_reconstructor_fixtures.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import sys
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy
import pytest

from ptychodus.api import spawn
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import NoOptions, ReconstructInput, ReconstructOutput
from ptychodus.api.spawn import ChildError, iterate_in_subprocess

# The spawned child resolves the fixtures module by name against the sys.path it
# inherits from this process, so the tests directory must be on it before any
# child is spawned. pytest's default import mode already does this; the insert
# keeps it true under other invocations.
_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import subprocess_child_fixtures as fixtures  # noqa: E402
import subprocess_reconstructor_fixtures as reconstructor_fixtures  # noqa: E402


def _minimal_product() -> Product:
    metadata = ProductMetadata(
        name='fake',
        comments='',
        detector_distance_m=1.0,
        photon_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )
    pixel_geometry = PixelGeometry(1.0e-9, 1.0e-9)
    return Product(
        metadata=metadata,
        probe_positions=ProbePositionSequence([]),
        probes=ProbeEnsemble(
            array=numpy.zeros((1, 1, 4, 4), dtype=numpy.complex64),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.zeros((1, 4, 4), dtype=numpy.complex64),
            layer_spacing_m=[],
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(0.0, 0.0),
        ),
        losses=[],
    )


def _minimal_reconstruct_input() -> ReconstructInput:
    return ReconstructInput(
        diffraction_patterns=numpy.zeros((1, 4, 4), dtype=numpy.float32),
        bad_pixels=numpy.zeros((4, 4), dtype=numpy.bool_),
        product=_minimal_product(),
    )


class _ProcessRecordingContext:
    """A spawn context that hands back the processes it was asked to build.

    Wraps rather than subclasses the real context: spawn pickles the Process
    object itself, and a locally defined subclass would not survive that.
    """

    def __init__(self, context: Any) -> None:
        self._context = context
        self.processes: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    def Process(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        process = self._context.Process(*args, **kwargs)
        self.processes.append(process)
        return process


def _record_processes(monkeypatch: pytest.MonkeyPatch) -> _ProcessRecordingContext:
    recorded = _ProcessRecordingContext(multiprocessing.get_context('spawn'))
    monkeypatch.setattr(spawn.multiprocessing, 'get_context', lambda method: recorded)
    return recorded


def test_items_stream_back_in_order() -> None:
    work = partial(reconstructor_fixtures.yield_outputs, _minimal_product(), 3)

    outputs = list(iterate_in_subprocess(work))

    assert [output.progress for output in outputs] == [1, 2, 3]


def test_a_bound_method_runs_on_a_copy_of_its_object() -> None:
    """The object crosses as itself, so what it yields comes back and what it does to itself does not."""
    reconstructor = reconstructor_fixtures.FakeTrainableReconstructor(num_outputs=2)
    work = partial(reconstructor.reconstruct, _minimal_reconstruct_input(), options=NoOptions())

    outputs = list(iterate_in_subprocess(work))

    assert [(output.progress, output.progress_goal) for output in outputs] == [(1, 2), (2, 2)]


def test_training_in_a_child_leaves_the_parent_object_untouched(tmp_path: Path) -> None:
    reconstructor = reconstructor_fixtures.FakeTrainableReconstructor()
    work = partial(reconstructor.train, tmp_path, tmp_path, options=NoOptions())

    outputs = list(iterate_in_subprocess(work))

    assert outputs[-1].trained_model == tmp_path / 'model.fake'
    assert outputs[-1].trained_model.read_bytes() == b'trained'
    assert not reconstructor.trained_in_this_process
    assert not reconstructor.is_model_loaded()


def test_child_exception_propagates_as_child_error() -> None:
    with pytest.raises(ChildError) as excinfo:
        list(iterate_in_subprocess(partial(fixtures.raise_immediately, 'boom')))

    assert excinfo.value.child_exception_type == 'ValueError'
    # The original exception is picklable, so the parent got it back.
    assert isinstance(excinfo.value.child_exception, ValueError)
    assert 'boom' in str(excinfo.value.child_exception)


def test_unpicklable_item_raises_child_error_rather_than_vanishing() -> None:
    """A child puts the object itself, so the put is where pickling fails.

    The failure must reach the parent: pickled on a feeder thread instead, the
    message would be dropped and the parent would see a stream with no output.
    """
    with pytest.raises(ChildError):
        list(iterate_in_subprocess(fixtures.yield_unpicklable))


def test_unpicklable_work_fails_in_this_process() -> None:
    def work() -> list[int]:
        return [1]

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        list(iterate_in_subprocess(work))


def test_an_item_is_captured_when_it_is_yielded() -> None:
    (array,) = list(iterate_in_subprocess(fixtures.yield_then_modify_in_place))

    assert not numpy.any(array)


def test_the_environment_reaches_the_work() -> None:
    name = 'PTYCHODUS_TEST_ITERATE_ENVIRONMENT'
    work = partial(fixtures.yield_environment, name)

    assert list(iterate_in_subprocess(work, environment={name: 'set'})) == ['set']


def test_hanging_child_is_terminated_on_iterator_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child sleeps for an hour, so only the cleanup path can end this test.

    Asserting the child is reaped matters: without it the test passes whenever it
    merely fails to hang, including if the process were left running and only
    collected when the interpreter exits.
    """
    recorded = _record_processes(monkeypatch)
    iterator = cast(
        Generator[ReconstructOutput, None, None],
        iterate_in_subprocess(
            partial(reconstructor_fixtures.yield_one_then_hang, _minimal_product()),
            terminate_grace_sec=1.0,
        ),
    )

    # Take the child's one output. That both spawns it and leaves the parent
    # generator suspended inside the context manager -- closing a generator that
    # was never advanced runs no cleanup, so this step is what arms the test.
    next(iterator)

    (process,) = recorded.processes
    assert process.is_alive()

    iterator.close()

    assert not process.is_alive()
    assert process.exitcode is not None


@pytest.mark.parametrize(
    'work',
    [
        partial(reconstructor_fixtures.yield_outputs, _minimal_product(), 1),
        partial(fixtures.raise_immediately, 'boom'),
    ],
)
def test_finished_child_exits_on_its_own(monkeypatch: pytest.MonkeyPatch, work: Any) -> None:
    """A child that has sent its sentinel is joined, not terminated.

    A SIGTERM landing in the child's exit cleanup leaves the semaphores it
    created registered with the resource tracker, which warns about them when
    the parent exits; a zero exit code shows the child was left to finish.
    """
    recorded = _record_processes(monkeypatch)

    try:
        list(iterate_in_subprocess(work))
    except ChildError:
        pass

    (process,) = recorded.processes
    assert process.exitcode == 0


def test_child_log_is_forwarded_to_parent_logger(caplog: pytest.LogCaptureFixture) -> None:
    work = partial(
        reconstructor_fixtures.log_then_yield, 'hello from the child', _minimal_product()
    )

    with caplog.at_level(logging.WARNING, logger=reconstructor_fixtures.__name__):
        outputs = list(iterate_in_subprocess(work))

    assert len(outputs) == 1
    assert any('hello from the child' in record.message for record in caplog.records)
