"""Discipline gate for the GPU subprocess isolation refactor.

The invariant the refactor buys us is that the parent ptychodus process never
acquires a GPU context. Acquiring one is what pins driver state and GPU memory
to the long-lived parent, and it is what stops ptychodus from mixing backends
built on different frameworks. Context acquisition means things like
constructing ``ptychi.api.task.PtychographyTask``, placing a tensor on a
device, ``tf.keras.Model.fit``, constructing a Lightning ``Trainer``, or
allocating a CuPy array -- all of which must happen inside a freshly spawned
child that dies at end-of-call.

*Importing* a GPU framework in the parent is fine and expected. ``ptychi.api``
pulls torch in for its type annotations so the parent can build the picklable
``PtychographyTaskOptions`` the child needs, and no CUDA runtime is
initialised by that. See the invariant note at the top of
``ptychodus.model.processing._subprocess_protocol`` for the canonical
statement.

So this module checks two things, in one spawned probe process that imports
``ptychodus`` plus the main composition roots:

1. No module that would acquire a context merely by being imported (or that
   simply has no parent-side use) ended up in ``sys.modules``.
2. No framework that *is* allowed parent-side is holding a live GPU context.

Runs in a spawned subprocess so it is not tainted by whatever the pytest
worker previously loaded (e.g. earlier tests that touched a GPU-backed
plugin).
"""

from __future__ import annotations

from typing import Any
import multiprocessing
import sys
import textwrap

import pytest

# Frameworks the parent is allowed to import -- it needs them to build the
# picklable options/payload objects it hands to the child -- but which must
# never be holding a GPU context parent-side. Enforced by
# ``_gpu_context_offenders`` rather than by an import ban.
CONTEXT_CAPABLE_MODULES = (
    'torch',
    'tensorflow',
    'ptychi',  # covers ptychi.api etc.
    'lightning',
    'pytorch_lightning',
)

# Modules banned outright in the parent: importing them acquires a context, or
# they are child-side backend packages with no parent-side use at all.
CHILD_ONLY_MODULES = (
    'cupy',  # links and initialises the CUDA runtime at import; no non-invasive probe
    'ptycho',  # ptychopinn TensorFlow package; configures GPUs at import
    'ptycho_torch',  # ptychopinn_torch backend; child-side entry point only
)

# ``ptychozoon.data_structures`` and ``ptychozoon.settings`` are CPU-only --
# they pull in only numpy, dataclasses, enum, and typing. The parent-side
# fluorescence factory imports them so it can construct the ptychozoon
# payload directly instead of duplicating its fields. Any *other*
# ``ptychozoon.*`` module (notably ``vspi_enhance``) pulls in CuPy and is a
# regression if it lands in the parent's ``sys.modules``.
ALLOWED_PTYCHOZOON_MODULES = frozenset(
    {
        'ptychozoon',
        'ptychozoon.data_structures',
        'ptychozoon.settings',
    }
)


def _is_child_only(module_name: str) -> bool:
    if module_name == 'ptychozoon' or module_name.startswith('ptychozoon.'):
        return module_name not in ALLOWED_PTYCHOZOON_MODULES
    return any(module_name == m or module_name.startswith(m + '.') for m in CHILD_ONLY_MODULES)


def _gpu_context_offenders() -> tuple[list[str], list[str]]:
    """Report any live GPU context held by an already-imported framework.

    Returns ``(offenders, notes)``. Every probe below only *observes* state --
    none of them create a context as a side effect, so calling this cannot
    itself break the invariant. A probe that fails (framework internals moved
    between versions) lands in ``notes`` instead of silently passing.
    """
    offenders: list[str] = []
    notes: list[str] = []

    torch = sys.modules.get('torch')
    if torch is not None:
        for backend_name in ('cuda', 'xpu'):
            backend = getattr(torch, backend_name, None)
            is_initialized = getattr(backend, 'is_initialized', None)
            if is_initialized is None:
                continue  # e.g. torch.xpu is absent on older torch builds
            try:
                if is_initialized():
                    offenders.append(f'torch.{backend_name} context is initialized')
            except Exception as exc:  # noqa: BLE001
                notes.append(f'could not probe torch.{backend_name}: {type(exc).__name__}: {exc}')

    if 'tensorflow' in sys.modules:
        try:
            from tensorflow.python.eager import context as tf_context

            # context_safe() returns the eager context, or None when none has
            # been created -- unlike context(), it does not create one.
            if tf_context.context_safe() is not None:
                offenders.append('tensorflow eager context is initialized')
        except Exception as exc:  # noqa: BLE001
            notes.append(f'could not probe tensorflow eager context: {type(exc).__name__}: {exc}')

    return offenders, notes


def _check_in_subprocess(result_queue: multiprocessing.Queue[dict[str, list[str]] | str]) -> None:
    try:
        # Import the top-level package plus the main composition roots.
        import ptychodus  # noqa: F401
        import ptychodus.model.core  # noqa: F401
        import ptychodus.model.processing.subprocess_reconstructor  # noqa: F401
        import ptychodus.model.processing._subprocess_protocol  # noqa: F401

        context_offenders, notes = _gpu_context_offenders()
        result_queue.put(
            {
                'child_only': sorted(m for m in sys.modules if _is_child_only(m)),
                'context': context_offenders,
                'notes': notes,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(f'{type(exc).__name__}: {exc}')


@pytest.fixture(scope='module')
def probe_result() -> dict[str, list[str]]:
    """Run the import/context probe once and share it across both tests."""
    ctx = multiprocessing.get_context('spawn')
    result_queue: multiprocessing.Queue[dict[str, list[str]] | str] = ctx.Queue()
    process = ctx.Process(target=_check_in_subprocess, args=(result_queue,))
    process.start()
    process.join(timeout=60.0)

    assert not process.is_alive(), 'GPU-isolation probe subprocess did not exit in 60s.'

    result: Any = result_queue.get(timeout=1.0)

    if isinstance(result, str):
        raise AssertionError(f'Probe subprocess raised: {result}')

    return result


def test_parent_imports_no_child_only_modules(probe_result: dict[str, list[str]]) -> None:
    offenders = probe_result['child_only']
    assert offenders == [], textwrap.dedent(
        f"""\
        Parent ptychodus process imported modules that belong to a child only.
        Offending sys.modules entries: {offenders}

        These modules acquire a GPU context just by being imported, or are
        backend packages the parent has no reason to touch. Importing a GPU
        framework parent-side to build a picklable payload is fine -- see
        CONTEXT_CAPABLE_MODULES -- but these are not in that category. Wire the
        backend through SubprocessReconstructor with a child-side entry point
        instead.
        """
    )


def test_parent_holds_no_gpu_context(probe_result: dict[str, list[str]]) -> None:
    offenders = probe_result['context']
    notes = probe_result['notes']
    assert offenders == [], textwrap.dedent(
        f"""\
        Parent ptychodus process is holding a live GPU context.
        Offenders: {offenders}
        Probe notes: {notes or 'none'}

        Importing torch/tensorflow/ptychi in the parent is allowed; acquiring a
        context is not. Something at import time placed a tensor on a device,
        constructed a PtychographyTask/Trainer, or otherwise initialised the
        runtime. Move that work into a child entry point dispatched through
        SubprocessReconstructor (see
        ptychodus.model.processing._subprocess_protocol).
        """
    )
