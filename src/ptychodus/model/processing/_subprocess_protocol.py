"""Shared machinery for running GPU-touching code in transient spawned subprocesses.

The parent-side ptychodus process must NEVER acquire a GPU context (no
``torch.cuda.*`` / ``tf.config.experimental.*`` calls, no tensor-on-GPU
allocations, no ``ptychi.api.task.PtychographyTask`` construction, no
``tf.keras.Model`` fit, no Lightning ``Trainer`` construction). All GPU work
happens inside a freshly ``spawn``ed child process that lives only for the
duration of one call (reconstruct / train / enhance) and dies immediately
after. This gives each call a clean GPU context and fully releases GPU
memory between calls, which lets ptychodus mix reconstructor backends built
on different GPU frameworks without driver-state interference.

The parent MAY import a GPU framework at module load if doing so is required
to construct picklable configuration objects the child needs — e.g. importing
``ptychi.api.options.*`` pulls torch in for its type annotations, but no CUDA
runtime is initialised until a tensor is placed on a GPU. Keep this to the
minimum needed for the payload; when in doubt, do the translation child-side.

This module provides the reusable transport layer that ptychozoon,
:class:`SubprocessReconstructor`, and any future GPU-isolated consumer share.

Wire protocol
-------------

Messages travel over a ``multiprocessing.Queue`` as tagged tuples. Because a
single child thread produces every message, ordering is preserved (a log line
arrives before the result it precedes).

- ``('log', levelno, logger_name, formatted_message)`` -- child log record
- ``('error', traceback_str, exception_type_name, pickled_exc_or_none)``
- ``None`` -- end-of-stream sentinel (always the final message)

Any other tag is passed through to the consumer verbatim; consumers are free
to define their own tags (e.g. ``'output'``, ``'progress'``, ``'result'``,
``'settings_sync'``, ``'model_saved'``).

Child entry points are addressed by ``'dotted.module.path:function_name'``
strings; the child imports the module lazily and calls
``function(payload, queue)``. The parent never imports the entry-point
module.
"""

from __future__ import annotations

import importlib
import io
import logging
import multiprocessing
import pickle
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    'TAG_ERROR',
    'TAG_LOG',
    'ChildError',
    'SubprocessLogHandler',
    'dump_settings_registry_to_string',
    'install_child_log_forwarder',
    'load_settings_registry_from_string',
    'run_subprocess',
    'send_error',
]


TAG_LOG = 'log'
TAG_ERROR = 'error'


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

    def __init__(self, result_queue: 'Queue[Any]') -> None:
        super().__init__()
        self._result_queue = result_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._result_queue.put((TAG_LOG, record.levelno, record.name, self.format(record)))
        except Exception:
            # Never let logging failures break the worker.
            pass


def install_child_log_forwarder(
    result_queue: 'Queue[Any]', level: int = logging.INFO
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


def send_error(result_queue: 'Queue[Any]', exc: BaseException) -> None:
    """Marshal an unhandled child-side exception onto the queue.

    Pickles the exception when possible so the parent can re-raise the
    original class; otherwise the parent falls back to :class:`ChildError`
    carrying the traceback text.
    """
    tb = traceback.format_exc()
    try:
        pickled_exc: bytes | None = pickle.dumps(exc)
    except Exception:
        pickled_exc = None
    try:
        result_queue.put((TAG_ERROR, tb, type(exc).__name__, pickled_exc))
    except Exception:
        # Queue might already be closed; nothing we can do.
        pass


def _child_main(
    entry_point: str,
    payload: Any,
    result_queue: 'Queue[Any]',
    log_level: int,
) -> None:
    """Default target for the spawned :class:`multiprocessing.Process`.

    Runs in the child. Installs the log forwarder, imports the user entry
    point lazily (so GPU libraries load here, not in the parent), calls
    ``entry(payload, queue)``, and always sends the sentinel.
    """
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
    process: SpawnProcess
    queue: 'Queue[Any]'


@contextmanager
def run_subprocess(
    entry_point: str,
    payload: Any,
    *,
    log_level: int | None = None,
    terminate_grace_sec: float = 10.0,
) -> Iterator[Iterator[tuple[Any, ...]]]:
    """Spawn a child process and yield an iterator over its queue messages.

    Usage::

        with run_subprocess('ptychodus.model.ptychi._child:run_reconstruct', payload) as events:
            for event in events:
                tag = event[0]
                ...

    Messages tagged ``'log'`` are dispatched to the parent's logging tree by
    logger name and NOT re-yielded. Messages tagged ``'error'`` raise
    :class:`ChildError` (with the child's original exception attached when
    unpicklable-safe). ``None`` terminates the iterator normally.

    On ``__exit__``, the child is terminated (SIGTERM), joined with
    ``terminate_grace_sec`` seconds of grace, then killed (SIGKILL) if it did
    not exit. This mirrors the ptychozoon shutdown discipline.
    """
    if log_level is None:
        log_level = logging.getLogger().getEffectiveLevel()

    ctx = multiprocessing.get_context('spawn')
    result_queue: 'Queue[Any]' = ctx.Queue()
    process = ctx.Process(
        target=_child_main,
        args=(entry_point, payload, result_queue, log_level),
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

        if tag == TAG_LOG:
            _, levelno, logger_name, message = item
            logging.getLogger(logger_name).log(levelno, message)
            continue

        if tag == TAG_ERROR:
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
        state.queue.join_thread()
    except Exception:
        pass


def dump_settings_registry_to_string(registry: Any) -> str:
    """Serialize a :class:`SettingsRegistry` to an INI-format string.

    Uses the same field layout as :meth:`SettingsRegistry.save_settings` so
    the child can rehydrate a registry with the identical parameter values.
    """
    import configparser

    config = configparser.ConfigParser(interpolation=None)
    setattr(config, 'optionxform', lambda option: option)

    for group_name in registry:
        config.add_section(group_name)
        group = registry[group_name]
        for parameter_name, parameter in group.parameters().items():
            config.set(group_name, parameter_name, parameter.get_value_as_string())

    buf = io.StringIO()
    config.write(buf)
    return buf.getvalue()


def load_settings_registry_from_string(registry: Any, content: str) -> None:
    """Populate a :class:`SettingsRegistry` in place from an INI-format string.

    Only sets values on groups/parameters that already exist in ``registry``;
    unknown sections are silently ignored, matching
    :meth:`SettingsRegistry.open_settings`.
    """
    import configparser

    config = configparser.ConfigParser(interpolation=None)
    config.read_file(io.StringIO(content))

    for group_name in registry:
        try:
            group_config = config[group_name]
        except KeyError:
            continue
        group = registry[group_name]
        for parameter_name, parameter in group.parameters().items():
            try:
                value_string = group_config[parameter_name]
            except KeyError:
                pass
            else:
                parameter.set_value_from_string(value_string)
