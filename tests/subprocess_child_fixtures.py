"""Child-side entry points used by ``test_subprocess_reconstructor``.

The ``spawn``-context child imports this module by dotted path, so it must be
resolvable on the child's ``sys.path``; the test module puts the tests
directory there before spawning. Not collected by pytest (the filename does
not match ``test_*.py``). Uses no GPU frameworks. Do not import from
application code.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from multiprocessing.queues import SimpleQueue
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy

from ptychodus.api.reconstruct import (
    ReconstructOutput,
    ReconstructorMessage,
    TrainOutput,
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


class ChildCallCapturedError(Exception):
    """Carries what a stubbed backend call received, back across the spawn."""


def _install_fake_ptycho_torch(**module_attrs: Any) -> None:
    """Put a minimal fake ``ptycho_torch`` (and ``torch``) on ``sys.modules``.

    ``module_attrs`` maps a dotted submodule name, with dots replaced by
    underscores, to the attributes that submodule should carry.
    """
    package = _fake_module('ptycho_torch', __path__=[])
    modules: dict[str, ModuleType] = {
        'ptycho_torch': package,
        'ptycho_torch.workflows': _fake_module('ptycho_torch.workflows', __path__=[]),
        # _resolve_device asks torch whether a device is present; say no, so the
        # fixture resolves to CPU without a real runtime.
        'torch': _fake_module('torch', cuda=SimpleNamespace(is_available=lambda: False)),
    }

    for dotted, attrs in module_attrs.items():
        name = dotted.replace('_DOT_', '.')
        modules[name] = _fake_module(name, **attrs)

    sys.modules.update(modules)


def capture_ptychopinn_reconstruct_call(payload: Any, queue: Any) -> None:
    """Run the real child inference entry point against a stubbed backend."""
    from dataclasses import dataclass

    @dataclass
    class FakeInferenceConfig:
        patch_weighting: str = 'probe'
        batch_size: int = 1

    loaded_model = SimpleNamespace(
        model_config=SimpleNamespace(mode='Unsupervised'),
        inference_config=FakeInferenceConfig(),
        data_config=object(),
        training_config=object(),
    )

    def reconstruct_from_arrays(model: Any, arrays: Any, **kwargs: Any) -> None:
        runtime_params = kwargs['runtime_params']
        raise ChildCallCapturedError(
            {
                'array_keys': sorted(arrays),
                'device': kwargs['device'],
                'num_workers': kwargs['num_workers'],
                'patch_weighting': runtime_params.inference_config.patch_weighting,
                'batch_size': runtime_params.inference_config.batch_size,
            }
        )

    _install_fake_ptycho_torch(
        ptycho_torch_DOT_inference={
            'ReconstructionRuntimeParams': SimpleNamespace,
            'reconstruct_from_arrays': reconstruct_from_arrays,
        },
        ptycho_torch_DOT_workflows_DOT_bundle_io={
            'load_inference_bundle_torch': lambda _dir: (loaded_model, {}),
        },
    )

    from ptychodus.plugins.ptychopinn import _subprocess

    _subprocess.run_reconstruct(payload, queue)


def capture_ptychopinn_train_call(payload: Any, queue: Any) -> None:
    """Run the real child training entry point against a stubbed backend."""

    class FakeExecutionRequest:
        def __init__(self, values: Any, explicit_fields: Any) -> None:
            self.values = dict(values)
            self.explicit_fields = set(explicit_fields)

    def train(dataset: Any, output_dir: Any, settings: Any, **kwargs: Any) -> None:
        execution_config = kwargs['execution_config']
        raise ChildCallCapturedError(
            {
                'dataset_name': str(dataset.name),
                'settings_is_mapping': isinstance(settings, Mapping),
                'has_nphotons': 'nphotons' in settings,
                'mode': settings.get('mode'),
                'execution_values': execution_config.values,
                'explicit_fields': sorted(execution_config.explicit_fields),
            }
        )

    _install_fake_ptycho_torch(
        ptycho_torch_DOT_execution_request={'ExecutionRequest': FakeExecutionRequest},
        ptycho_torch_DOT_train={'train': train},
    )

    from ptychodus.plugins.ptychopinn import _subprocess

    _subprocess.run_train(payload, queue)


def yield_n_outputs(payload: Any, queue: Any) -> None:
    """Emit ``payload.options['n']`` :class:`ReconstructOutput`s with product=None."""
    n = int(payload.options['n'])
    for i in range(n):
        output = ReconstructOutput(product=payload.options['product'], progress=i + 1)
        queue.put((ReconstructorMessage.OUTPUT, output))


def raise_immediately(payload: Any, queue: Any) -> None:
    """Raise a :class:`ValueError` before emitting anything."""
    raise ValueError(payload.options['message'])


def hang_forever(payload: Any, queue: Any) -> None:
    """Sleep so the parent must terminate us."""
    time.sleep(3600.0)


def emit_output_then_hang(payload: Any, queue: Any) -> None:
    """Emit one output, then sleep so the parent must terminate us.

    The output is what lets the parent reach its first ``yield``: a child that
    only sleeps leaves the parent generator un-started, and closing a generator
    that never ran executes no cleanup at all.
    """
    queue.put(
        (
            ReconstructorMessage.OUTPUT,
            ReconstructOutput(product=payload.options['product'], progress=1),
        )
    )
    time.sleep(3600.0)


def emit_log_then_output(payload: Any, queue: Any) -> None:
    """Log a message, then emit one output; parent must see log before output."""
    logger.warning(payload.options['log_message'])
    output = ReconstructOutput(product=payload.options['product'], progress=1)
    queue.put((ReconstructorMessage.OUTPUT, output))


def emit_settings_sync_then_output(payload: Any, queue: Any) -> None:
    """Emit a settings-sync message, then an output."""
    queue.put((ReconstructorMessage.SETTINGS_SYNC, payload.options['settings']))
    output = ReconstructOutput(product=payload.options['product'], progress=1)
    queue.put((ReconstructorMessage.OUTPUT, output))


def train_and_save(payload: Any, queue: Any) -> None:
    """Emit one :class:`TrainOutput` and a model-saved path."""
    queue.put((ReconstructorMessage.TRAIN_OUTPUT, TrainOutput(progress=1)))
    queue.put((ReconstructorMessage.MODEL_SAVED, payload.options['saved_path']))


def probe_cuda_uuids(_payload: Any, queue: SimpleQueue[Any]) -> None:
    """Report the UUID of every CUDA device visible to this child.

    A name does not identify a device when a host holds two of the same model,
    so proving a restriction reached the right one takes something unique.
    """
    import torch

    uuids = [
        str(torch.cuda.get_device_properties(index).uuid)
        for index in range(torch.cuda.device_count())
    ]
    queue.put((ReconstructorMessage.OUTPUT, uuids))


def emit_unpicklable(payload: Any, queue: Any) -> None:
    """Put a message that cannot be pickled; the put itself must raise."""
    queue.put((ReconstructorMessage.OUTPUT, lambda: None))


def emit_then_modify_in_place(payload: Any, queue: Any) -> None:
    """Put an array, then overwrite it; the parent must see what was put."""
    array = numpy.zeros(1 << 20)
    queue.put(('array', array))
    array += 1.0
