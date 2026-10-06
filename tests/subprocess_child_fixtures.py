"""Child-side entry points used by ``test_subprocess_reconstructor``.

The ``spawn``-context child imports this module by dotted path, so it must be
resolvable on the child's ``sys.path``; the test module puts the tests
directory there before spawning. Not collected by pytest (the filename does
not match ``test_*.py``). Uses no GPU frameworks. Do not import from
application code.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

from ptychodus.api.reconstruct import ReconstructOutput, TrainOutput
from ptychodus.model.processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_SETTINGS_SYNC,
    TAG_TRAIN_OUTPUT,
)

logger = logging.getLogger(__name__)


def _set_attrs(module: ModuleType, **attrs: Any) -> None:
    """Attach attributes to a module.

    Via setattr because a ModuleType has no declared attributes, so assigning them
    directly is an attr-defined error on every line.
    """
    for name, value in attrs.items():
        setattr(module, name, value)


def _fake_module(name: str, **attrs: Any) -> ModuleType:
    """A stand-in module carrying `attrs`, for faking an absent optional package."""
    module = ModuleType(name)
    _set_attrs(module, **attrs)
    return module


class GroupingCallCapturedError(Exception):
    pass


def capture_ptychopinn_grouping_call(payload: Any, queue: Any) -> None:
    """Run the real child entry point with fake optional PtychoPINN modules."""
    grouping_call: dict[str, int] = {}

    def generate_grouped_data(n: int, *, nsamples: int, gridsize: int, **kwargs: int) -> object:
        grouping_call.update(N=n, K=kwargs['K'], nsamples=nsamples, gridsize=gridsize)
        return object()

    raw_data = SimpleNamespace(
        probeGuess=object(),
        generate_grouped_data=generate_grouped_data,
    )

    def stop_after_grouping(*_args: object, **_kwargs: object) -> None:
        raise GroupingCallCapturedError(grouping_call)

    ptycho = _fake_module('ptycho', __path__=[])
    config = _fake_module('ptycho.config.config', update_legacy_dict=lambda _cfg, _config: None)
    components = _fake_module(
        'ptycho.workflows.components', load_inference_bundle=lambda _path: (object(), {})
    )
    loader = _fake_module('ptycho.loader', load=stop_after_grouping)
    params = _fake_module('ptycho.params', cfg={})
    probe = _fake_module('ptycho.probe', set_probe_guess=lambda *_args: None)
    tf_helper = _fake_module('ptycho.tf_helper')

    for name, module in {
        'ptycho': ptycho,
        'ptycho.config': ModuleType('ptycho.config'),
        'ptycho.config.config': config,
        'ptycho.workflows': ModuleType('ptycho.workflows'),
        'ptycho.workflows.components': components,
        'ptycho.loader': loader,
        'ptycho.params': params,
        'ptycho.probe': probe,
        'ptycho.tf_helper': tf_helper,
    }.items():
        sys.modules[name] = module

    _set_attrs(ptycho, loader=loader, params=params, probe=probe, tf_helper=tf_helper)

    from ptychodus.model.ptychopinn import _subprocess

    _subprocess._create_raw_data = lambda parameters: raw_data
    _subprocess.run_reconstruct(payload, queue)


def yield_n_outputs(payload: Any, queue: Any) -> None:
    """Emit ``payload['n']`` :class:`ReconstructOutput`s with product=None."""
    n = int(payload['n'])
    for i in range(n):
        output = ReconstructOutput(product=payload['product'], progress=i + 1)
        queue.put((TAG_OUTPUT, pickle.dumps(output)))


def raise_immediately(payload: Any, queue: Any) -> None:
    """Raise a :class:`ValueError` before emitting anything."""
    raise ValueError(payload['message'])


def hang_forever(payload: Any, queue: Any) -> None:
    """Sleep so the parent must terminate us."""
    time.sleep(3600.0)


def emit_output_then_hang(payload: Any, queue: Any) -> None:
    """Emit one output, then sleep so the parent must terminate us.

    The output is what lets the parent reach its first ``yield``: a child that
    only sleeps leaves the parent generator un-started, and closing a generator
    that never ran executes no cleanup at all.
    """
    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product=payload['product'], progress=1))))
    time.sleep(3600.0)


def emit_log_then_output(payload: Any, queue: Any) -> None:
    """Log a message, then emit one output; parent must see log before output."""
    logger.warning(payload['log_message'])
    output = ReconstructOutput(product=payload['product'], progress=1)
    queue.put((TAG_OUTPUT, pickle.dumps(output)))


def emit_settings_sync_then_output(payload: Any, queue: Any) -> None:
    """Emit a settings-sync message, then an output."""
    queue.put((TAG_SETTINGS_SYNC, payload['settings']))
    output = ReconstructOutput(product=payload['product'], progress=1)
    queue.put((TAG_OUTPUT, pickle.dumps(output)))


def train_and_save(payload: Any, queue: Any) -> None:
    """Emit one :class:`TrainOutput` and a model-saved path."""
    queue.put((TAG_TRAIN_OUTPUT, pickle.dumps(TrainOutput(progress=1))))
    queue.put((TAG_MODEL_SAVED, payload['saved_path']))
