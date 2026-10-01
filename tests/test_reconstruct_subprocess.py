"""The parts of the subprocess driver a launcher depends on.

Its contract with a launcher is the event stream and the exit code, and both are produced
by small helpers around the shared reconstruction tail rather than by the tail itself. These
tests pin down the stream's framing (strict JSON, one object per line, silent on a rank that
has no launcher listening), where the per-rank log lands, how the options file is resolved
when none was named, and which exit code each ending produces. ``main()`` is never called:
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
    log_file,
    resolve_ptychi_options_file,
)


def _events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_each_event_is_one_line_of_strict_json() -> None:
    buffer = io.StringIO()
    stream = EventStream(buffer)

    stream.emit('started', pid=7)
    stream.emit('finished', cancelled=False, path=None)

    assert _events(buffer) == [
        {'event': 'started', 'pid': 7},
        {'event': 'finished', 'cancelled': False, 'path': None},
    ]


def test_a_non_finite_value_is_refused_rather_than_written() -> None:
    # Python would emit a bare ``Infinity`` literal, which parsers outside Python reject.
    # Failing here is what forces non-finite losses to be normalized before they arrive.
    stream = EventStream(io.StringIO())

    with pytest.raises(ValueError):
        stream.emit('epoch', loss=float('inf'))


def test_a_process_with_no_stream_emits_nothing() -> None:
    # Secondary ranks run the same code; only one of them has a launcher reading.
    stream = EventStream(None)

    stream.emit('started')
    stream.emit_exception(RuntimeError('boom'))


def test_an_exception_event_carries_its_type_and_traceback() -> None:
    buffer = io.StringIO()
    stream = EventStream(buffer)

    try:
        raise ValueError('bad options')
    except ValueError as exc:
        stream.emit_exception(exc)

    (event,) = _events(buffer)
    assert event['event'] == 'error'
    assert event['type'] == 'ValueError'
    assert event['message'] == 'bad options'
    assert 'ValueError' in str(event['traceback'])


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


def test_the_main_rank_logs_to_the_plain_name(tmp_path: Path) -> None:
    assert log_file(tmp_path, 0).name == 'reconstruct.log'


def test_a_secondary_rank_logs_to_its_own_file(tmp_path: Path) -> None:
    # Every rank logs, and one shared name would have them truncating each other's.
    assert log_file(tmp_path, 1).name == 'reconstruct.rank01.log'
    assert log_file(tmp_path, 12).name == 'reconstruct.rank12.log'


def test_an_explicit_options_file_wins(tmp_path: Path) -> None:
    staged = tmp_path / 'ptychi_options.json'
    staged.write_text('{}')
    chosen = tmp_path / 'elsewhere.json'

    assert resolve_ptychi_options_file(chosen, tmp_path) == chosen


def test_the_stdin_sentinel_is_passed_through(tmp_path: Path) -> None:
    # ``-`` is not a path to probe for; resolving it against the input directory would
    # turn an in-memory handoff into a missing file.
    (tmp_path / 'ptychi_options.json').write_text('{}')

    assert resolve_ptychi_options_file(Path('-'), tmp_path) == Path('-')


def test_the_staged_options_are_used_when_present(tmp_path: Path) -> None:
    staged = tmp_path / 'ptychi_options.json'
    staged.write_text('{}')

    assert resolve_ptychi_options_file(None, tmp_path) == staged


def test_a_missing_staged_options_file_falls_back_rather_than_failing(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # No staging path writes that file today, so its absence is the ordinary case. An
    # error here would make the common invocation the failing one.
    with caplog.at_level('WARNING', logger='reconstruct_subprocess'):
        assert resolve_ptychi_options_file(None, tmp_path) is None

    assert any('stock LSQML' in record.getMessage() for record in caplog.records)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_directory=tmp_path,
        output_directory=tmp_path / 'out',
        ptychi_options_file=None,
        index_filter='all',
        num_sync_epochs=1,
        mmap_file=None,
    )


def test_unreadable_options_are_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Distinguished from a reconstruction failure so a launcher can tell "you gave me
    # nonsense" from "the reconstruction broke".
    (tmp_path / 'ptychi_options.json').write_text('not json')
    buffer = io.StringIO()
    stream = EventStream(buffer)

    code = reconstruct_subprocess._reconstruct(
        stream, reconstruct_subprocess.CancellationToken(), _args(tmp_path), None
    )

    assert code is ExitCode.USAGE
    assert [event['event'] for event in _events(buffer)] == ['error']


def _stub_reconstruction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, product: object | None
) -> io.StringIO:
    """Replace everything between the options and the product with the minimum that runs."""
    options = reconstruct_subprocess.load_ptychi_options(None)
    options.reconstructor_options.num_epochs = 2

    monkeypatch.setattr(reconstruct_subprocess, 'load_ptychi_options', lambda path: options)
    monkeypatch.setattr(
        reconstruct_subprocess, 'load_diffraction_data', lambda path, mmap_file=None: object()
    )
    monkeypatch.setattr(reconstruct_subprocess, 'load_product', lambda path: object())

    class _Input:
        diffraction_patterns = numpy.zeros((3, 2, 2), dtype=numpy.float32)

    monkeypatch.setattr(
        reconstruct_subprocess,
        'prepare_reconstruct_input',
        lambda data, prod, index_filter=None: _Input(),
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

    assert code is ExitCode.OK
    events = _events(buffer)
    assert [event['event'] for event in events] == ['started', 'finished']
    assert events[-1]['cancelled'] is False


def test_a_run_that_produced_nothing_reports_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The tail returns None only for a cancel that landed before the first epoch.
    buffer = _stub_reconstruction(monkeypatch, tmp_path, product=None)
    stream = EventStream(buffer)

    code = reconstruct_subprocess._reconstruct(
        stream, reconstruct_subprocess.CancellationToken(), _args(tmp_path), None
    )

    assert code is ExitCode.CANCELLED
    assert _events(buffer)[-1] == {
        'event': 'finished',
        'epoch': 0,
        'cancelled': True,
        'path': None,
        'elapsed_s': pytest.approx(_events(buffer)[-1]['elapsed_s']),
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
    assert _events(buffer)[-1]['cancelled'] is True
