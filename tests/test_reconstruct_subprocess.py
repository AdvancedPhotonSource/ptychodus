"""The parts of the subprocess driver a launcher depends on.

Its contract with a launcher is the event stream and the exit code, and both are produced
by small helpers around the shared reconstruction tail rather than by the tail itself. These
tests pin down the stream's framing (strict JSON, one object per line, silent on a rank that
has no launcher listening) and which exit code each ending produces. How the options
document is resolved is shared with ptychodus-reconstruct and pinned in
test_reconstruct_common.py, and where the per-rank log lands in test_cli_logging.py. ``main()`` is never called:
it redirects file descriptors and installs signal handlers, neither of which a test process
can give back.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.product import LossValue  # noqa: E402
from ptychodus.api.reconstruct import ReconstructOutput  # noqa: E402

from ptychodus.cli import reconstruct_subprocess  # noqa: E402
from ptychodus.cli.reconstruct_subprocess import (  # noqa: E402
    EventStream,
    ExitCode,
    _finite_loss,
)


def _events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_each_event_is_one_line_of_strict_json() -> None:
    buffer = io.StringIO()
    stream = EventStream(buffer)

    stream.emit('started', pid=7)
    stream.emit('finished', state='succeeded', path=None)

    assert _events(buffer) == [
        {'event': 'started', 'pid': 7},
        {'event': 'finished', 'state': 'succeeded', 'path': None},
    ]


def test_a_non_finite_value_is_refused_rather_than_written() -> None:
    # Python would emit a bare ``Infinity`` literal, which parsers outside Python reject.
    # Failing here is what forces non-finite losses to be normalized before they arrive.
    stream = EventStream(io.StringIO())

    with pytest.raises(ValueError):
        stream.emit('progress', loss=float('inf'))


def test_a_process_with_no_stream_emits_nothing() -> None:
    # Secondary ranks run the same code; only one of them has a launcher reading.
    stream = EventStream(None)

    stream.emit('started')
    stream.emit_failure(RuntimeError('boom'))


def test_a_failure_finishes_with_its_type_and_traceback() -> None:
    buffer = io.StringIO()
    stream = EventStream(buffer)

    try:
        raise ValueError('bad options')
    except ValueError as exc:
        stream.emit_failure(exc, path=Path('/out/product.000003.h5'), progress=3, progress_goal=9)

    (event,) = _events(buffer)
    assert event['event'] == 'finished'
    assert event['state'] == 'failed'
    assert event['reason'] == 'ValueError'
    assert (event['progress'], event['progress_goal']) == (3, 9)
    assert event['path'] == '/out/product.000003.h5'
    assert 'bad options' in str(event['detail'])


def _output(progress: int, loss: float | None) -> ReconstructOutput:
    class _Product:
        losses = [] if loss is None else [LossValue(epoch=progress, value=loss)]

    return ReconstructOutput(product=_Product(), progress=progress)  # type: ignore[arg-type]


@pytest.mark.parametrize('loss', [float('inf'), float('-inf'), float('nan')])
def test_a_diverged_loss_becomes_null(loss: float) -> None:
    # A diverging reconstruction produces these readily, and they must not reach the JSON.
    assert _finite_loss(_output(1, loss)) is None


def test_a_missing_loss_becomes_null() -> None:
    assert _finite_loss(_output(1, None)) is None


def test_a_finite_loss_is_reported() -> None:
    assert _finite_loss(_output(1, 0.25)) == pytest.approx(0.25)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_directory=tmp_path,
        output_directory=tmp_path / 'out',
        options=None,
        index_filter='all',
        mmap_file=None,
    )


def test_unreadable_options_are_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Distinguished from a reconstruction failure so a launcher can tell "you gave me
    # nonsense" from "the reconstruction broke".
    (tmp_path / 'reconstruct_options.yaml').write_text('not: [a, document')
    buffer = io.StringIO()
    stream = EventStream(buffer)

    code = reconstruct_subprocess._reconstruct(
        stream, reconstruct_subprocess.CancellationToken(), _args(tmp_path), None
    )

    assert code is ExitCode.USAGE
    events = _events(buffer)
    assert [event['event'] for event in events] == ['finished']
    assert events[0]['state'] == 'failed'


def _stub_reconstruction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, product: object | None
) -> io.StringIO:
    """Replace everything between the options and the product with the minimum that runs."""
    monkeypatch.setattr(reconstruct_subprocess, 'load_trained_model', lambda *args: None)
    monkeypatch.setattr(
        reconstruct_subprocess, 'load_diffraction_data', lambda path, mmap_file=None: object()
    )
    monkeypatch.setattr(
        reconstruct_subprocess,
        'load_product',
        lambda path: argparse.Namespace(metadata=argparse.Namespace(far_field=True)),
    )

    class _Input:
        diffraction_patterns = numpy.zeros((3, 2, 2), dtype=numpy.float32)

    monkeypatch.setattr(
        reconstruct_subprocess,
        'pair_patterns_with_positions',
        lambda logger, data, prod, index_filter=None: _Input(),
    )
    monkeypatch.setattr(
        reconstruct_subprocess,
        'run_reconstruction',
        lambda *args, **kwargs: product,
    )
    return io.StringIO()


def test_a_completed_run_reports_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    buffer = _stub_reconstruction(monkeypatch, tmp_path, product=object())
    stream = EventStream(buffer)

    code = reconstruct_subprocess._reconstruct(
        stream, reconstruct_subprocess.CancellationToken(), _args(tmp_path), None
    )

    assert code is ExitCode.SUCCESS
    events = _events(buffer)
    assert [event['event'] for event in events] == ['started', 'finished']
    assert events[-1]['state'] == 'succeeded'
    assert events[-1]['reason'] is None


def test_a_run_that_produced_nothing_reports_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The tail returns None only for a cancel that landed before the first output.
    buffer = _stub_reconstruction(monkeypatch, tmp_path, product=None)
    stream = EventStream(buffer)
    cancellation = reconstruct_subprocess.CancellationToken()
    cancellation.cancel('SIGTERM')

    code = reconstruct_subprocess._reconstruct(stream, cancellation, _args(tmp_path), None)

    assert code is ExitCode.CANCELLED
    finished = _events(buffer)[-1]
    assert finished == {
        'event': 'finished',
        'state': 'cancelled',
        'reason': 'SIGTERM',
        'progress': 0,
        'progress_goal': 0,
        'path': None,
        'elapsed_s': pytest.approx(finished['elapsed_s']),
    }


def test_a_run_cancelled_part_way_reports_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    buffer = _stub_reconstruction(monkeypatch, tmp_path, product=object())
    stream = EventStream(buffer)
    cancellation = reconstruct_subprocess.CancellationToken()
    cancellation.cancel('SIGTERM')

    code = reconstruct_subprocess._reconstruct(stream, cancellation, _args(tmp_path), None)

    assert code is ExitCode.CANCELLED
    assert _events(buffer)[-1]['state'] == 'cancelled'
    assert _events(buffer)[-1]['reason'] == 'SIGTERM'
