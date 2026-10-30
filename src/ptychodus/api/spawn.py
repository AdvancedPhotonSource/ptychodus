"""Run GPU-touching code in a transient spawned subprocess.

The parent-side ptychodus process must NEVER acquire a GPU context (no
``torch.cuda.*`` / ``tf.config.experimental.*`` calls, no tensor-on-GPU
allocations, no ``ptychi.api.task.PtychographyTask`` construction, no
``tf.keras.Model`` fit, no Lightning ``Trainer`` construction). All GPU work
happens inside a child process started with ``spawn`` that lives only for the
duration of one call (reconstruct / train / enhance) and dies immediately
after. This gives each call a clean GPU context and fully releases GPU
memory between calls, which lets ptychodus mix reconstructor backends built
on different GPU frameworks without driver-state interference.

The parent MAY import a GPU framework if doing so is required to construct
picklable configuration objects the child needs — e.g. importing
``ptychi.api.options.*`` or ``ptycho_torch.config_params`` pulls torch in for
its type annotations, but no CUDA runtime is initialised until a tensor is
placed on a GPU. Keep this to the minimum needed for the payload, and prefer
importing from inside the payload builder over module scope so the cost lands
on the first call rather than at startup.

Wire protocol
-------------

Messages travel over a ``multiprocessing.SimpleQueue`` as tagged tuples. Because a
single child thread produces every message, ordering is preserved (a log line
arrives before the result it precedes).

A message is pickled by ``put`` itself, in the calling thread, before ``put``
returns -- unlike ``multiprocessing.Queue``, which pickles later in a feeder
thread. So a child puts the object it means to send, never pre-pickled bytes:
an array it goes on to modify in place has already been captured, and a message
that cannot be pickled raises at its own ``put`` and reaches the parent as a
:class:`ChildError` rather than vanishing from a background thread.

- ``(ChildMessage.LOG, levelno, logger_name, formatted_message)``
- ``(ChildMessage.ERROR, traceback_str, exception_type_name, pickled_exc_or_none)``
- ``None`` -- end-of-stream sentinel (always the final message)

Any other tag is passed through to the consumer verbatim. A consumer names
its own in an enum of its own, since what a child has to say about its work
is the consumer's vocabulary rather than the transport's.

Child entry points are addressed by ``'dotted.module.path:function_name'``
strings; the child imports the module lazily and calls
``function(payload, queue)``. The parent never imports the entry-point
module.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import pickle
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import SimpleQueue
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    'ChildError',
    'ChildMessage',
    'SubprocessLogHandler',
    'install_child_log_forwarder',
    'run_subprocess',
    'send_error',
]


class ChildMessage(StrEnum):
    """The two messages this module itself puts on the queue.

    A string enum rather than a pair of constants, so one place names every
    tag the transport owns and a reader need not go looking for the others.
    What travels is the value, and a child writing the bare string compares
    equal to the member -- which is what keeps an entry point that predates
    this class working unchanged.
    """

    LOG = 'log'
    ERROR = 'error'


class ChildError(RuntimeError):
    """Raised in the parent when the child subprocess reports an unhandled exception.

    The child's original traceback is available via ``child_traceback`` and,
    when the exception class is available parent-side, the original exception
    instance via ``child_exception``.
    """

    def __init__(
        self,
        message: str,
        *,
        child_traceback: str,
        child_exception_type: str,
        child_exception: BaseException | None,
    ) -> None:
        super().__init__(message)
        self.child_traceback = child_traceback
        self.child_exception_type = child_exception_type
        self.child_exception = child_exception


class SubprocessLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to the parent over a queue.

    Attach to the child's root logger so records from any GPU-framework logger
    (torch, ptychi, ptycho, ptychozoon, ...) propagate back to the parent's
    logging tree unchanged.
    """

    def __init__(self, result_queue: SimpleQueue[Any]) -> None:
        super().__init__()
        self._result_queue = result_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._result_queue.put(
                (ChildMessage.LOG, record.levelno, record.name, self.format(record))
            )
        except Exception:
            # Never let logging failures break the worker.
            pass


def install_child_log_forwarder(
    result_queue: SimpleQueue[Any], level: int = logging.INFO
) -> SubprocessLogHandler:
    """Install a queue-forwarding handler on the child's root logger.

    Returns the handler so it can be removed in a ``finally`` block. The
    formatter emits ``'name: message'`` because the parent-side dispatcher
    already tags each record with the child's logger name.
    """
    handler = SubprocessLogHandler(result_queue)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    return handler


def send_error(result_queue: SimpleQueue[Any], exc: BaseException) -> None:
    """Marshal an unhandled child-side exception onto the queue.

    Pickles the exception when possible so the parent can re-raise the
    original class; otherwise the parent falls back to :class:`ChildError`
    carrying the traceback text.

    The one message the transport pickles explicitly rather than leaving to
    the queue. An exception can pickle here and still fail to rebuild in the
    parent -- an ``__init__`` whose signature differs from ``args`` is the
    usual cause -- and a failure inside ``SimpleQueue.get`` would take the
    traceback down with it. Nested bytes let the parent drop only the
    exception.
    """
    tb = traceback.format_exc()
    try:
        pickled_exc: bytes | None = pickle.dumps(exc)
    except Exception:
        pickled_exc = None
    try:
        result_queue.put((ChildMessage.ERROR, tb, type(exc).__name__, pickled_exc))
    except Exception:
        # Queue might already be closed; nothing we can do.
        pass


def _child_main(
    entry_point: str,
    payload: Any,
    result_queue: SimpleQueue[Any],
    log_level: int,
    environment: Mapping[str, str],
) -> None:
    """Default target for the spawned :class:`multiprocessing.Process`.

    Runs in the child. Applies the environment, installs the log forwarder,
    imports the user entry point lazily (so GPU libraries load here, not in the
    parent), calls ``entry(payload, queue)``, and always sends the sentinel.
    """
    # Before anything else, and before the entry point is imported: a runtime
    # reads its visibility variable once, when it is first touched, and ignores
    # every later change to it.
    os.environ.update(environment)
    handler = install_child_log_forwarder(result_queue, level=log_level)
    try:
        module_path, _, func_name = entry_point.partition(':')
        if not func_name:
            raise ValueError(
                f"Entry point {entry_point!r} must be formatted as 'module.path:function_name'"
            )
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        func(payload, result_queue)
    except BaseException as exc:  # noqa: BLE001 - want to marshal every failure
        send_error(result_queue, exc)
    finally:
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            pass
        try:
            result_queue.put(None)
        except Exception:
            pass


@dataclass
class _RunState:
    """A spawned child process and the queue it reports on."""

    process: SpawnProcess
    queue: SimpleQueue[Any]


@contextmanager
def run_subprocess(
    entry_point: str,
    payload: Any,
    *,
    log_level: int | None = None,
    terminate_grace_sec: float = 10.0,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Iterator[tuple[Any, ...]]]:
    """Spawn a child process and yield an iterator over its queue messages.

    Usage::

        with run_subprocess('your.package.child_module:entry_function', payload) as events:
            for event in events:
                tag = event[0]
                ...

    Messages tagged :attr:`ChildMessage.LOG` are dispatched to the parent's
    logging tree by logger name and NOT re-yielded. Messages tagged
    :attr:`ChildMessage.ERROR` raise
    :class:`ChildError` (with the child's original exception attached when
    unpicklable-safe). ``None`` terminates the iterator normally.

    ``environment`` is applied in the child before its entry point is imported,
    which is what makes it the place to restrict the devices the child can see.

    On ``__exit__``, the child is terminated (SIGTERM), joined with
    ``terminate_grace_sec`` seconds of grace, then killed (SIGKILL) if it did
    not exit. This mirrors the ptychozoon shutdown discipline.
    """
    if log_level is None:
        log_level = logging.getLogger().getEffectiveLevel()

    ctx = multiprocessing.get_context('spawn')
    result_queue: SimpleQueue[Any] = ctx.SimpleQueue()
    process = ctx.Process(
        target=_child_main,
        args=(entry_point, payload, result_queue, log_level, dict(environment or {})),
    )
    process.start()

    state = _RunState(process=process, queue=result_queue)
    try:
        yield _iter_events(state)
    finally:
        _shutdown(state, terminate_grace_sec)


def _iter_events(state: _RunState) -> Iterator[tuple[Any, ...]]:
    while True:
        item = state.queue.get()
        if item is None:
            return

        tag = item[0]

        if tag == ChildMessage.LOG:
            _, levelno, logger_name, message = item
            logging.getLogger(logger_name).log(levelno, message)
            continue

        if tag == ChildMessage.ERROR:
            _, tb, type_name, pickled_exc = item
            child_exc: BaseException | None = None
            if pickled_exc is not None:
                try:
                    child_exc = pickle.loads(pickled_exc)
                except Exception:
                    child_exc = None
            raise ChildError(
                f'Subprocess raised {type_name}:\n{tb}',
                child_traceback=tb,
                child_exception_type=type_name,
                child_exception=child_exc,
            )

        yield item


def _shutdown(state: _RunState, grace_sec: float) -> None:
    process = state.process
    if process.is_alive():
        process.terminate()
    process.join(timeout=grace_sec)
    if process.is_alive():
        process.kill()
        process.join()
    try:
        state.queue.close()
    except Exception:
        pass
