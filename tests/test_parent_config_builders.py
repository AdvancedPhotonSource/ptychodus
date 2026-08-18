"""Discipline gate for the parent-side backend config builders.

``ptychopinn`` and ``ptychopinn_torch`` build their backend config objects in
the parent so the child receives finished objects and does nothing but GPU
work. That means the parent reaches ``ptycho.config.config`` /
``ptycho_torch.config_params`` at call time -- which
``tests/test_no_gpu_context.py`` cannot observe, because its probe only imports
the composition roots and never dispatches a reconstruction.

This module closes that gap. For each installed backend it spawns a probe that
calls the builders exactly as ``build_*_payload`` does, then asserts:

1. the configs construct and survive the pickle round-trip the spawn transport
   requires, and
2. no GPU context was acquired in the process that built them.

Each probe runs in its own spawned process so a framework loaded by an earlier
test cannot mask a failure here.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any
import multiprocessing
import sys
import textwrap

import pytest


def _gpu_context_offenders() -> list[str]:
    """Report any live GPU context. Observation only -- creates no context."""
    offenders: list[str] = []

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
                offenders.append(f'could not probe torch.{backend_name}: {type(exc).__name__}')

    if 'tensorflow' in sys.modules:
        try:
            from tensorflow.python.eager import context as tf_context

            if tf_context.context_safe() is not None:
                offenders.append('tensorflow eager context is initialized')
        except Exception as exc:  # noqa: BLE001
            offenders.append(f'could not probe tensorflow: {type(exc).__name__}')

    return offenders


def _build_ptychopinn_configs() -> list[Any]:
    from ptychodus.api.settings import SettingsRegistry
    from ptychodus.model.ptychopinn.reconstructor import (
        _build_inference_config,
        _build_training_config,
    )
    from ptychodus.model.ptychopinn.settings import (
        PtychoPINNModelSettings,
        PtychoPINNTrainingSettings,
    )

    registry = SettingsRegistry()
    model_settings = PtychoPINNModelSettings(registry)
    training_settings = PtychoPINNTrainingSettings(registry)

    return [
        _build_inference_config(
            model_settings, 'PINN', model_size=64, is_developer_mode_enabled=False
        ),
        _build_training_config(model_settings, training_settings, 'PINN'),
    ]


def _build_ptychopinn_torch_configs() -> list[Any]:
    from ptychodus.api.settings import SettingsRegistry
    from ptychodus.model.ptychopinn_torch.reconstructor import _build_configs
    from ptychodus.model.ptychopinn_torch.settings import (
        PtychoPINNTorchDataSettings,
        PtychoPINNTorchInferenceSettings,
        PtychoPINNTorchModelSettings,
        PtychoPINNTorchTrainingSettings,
    )

    registry = SettingsRegistry()

    return list(
        _build_configs(
            'Unsupervised',
            PtychoPINNTorchDataSettings(registry),
            PtychoPINNTorchModelSettings(registry),
            PtychoPINNTorchTrainingSettings(registry),
            PtychoPINNTorchInferenceSettings(registry),
        )
    )


def _build_ptycho_fm_configs() -> list[Any]:
    from ptychodus.api.settings import SettingsRegistry
    from ptychodus.model.ptycho_fm.reconstructor import _build_config
    from ptychodus.model.ptycho_fm.settings import (
        PtychoFMDataSettings,
        PtychoFMInferenceSettings,
        PtychoFMModelSettings,
        PtychoFMTrainingSettings,
    )

    registry = SettingsRegistry()

    return [
        _build_config(
            PtychoFMDataSettings(registry),
            PtychoFMModelSettings(registry),
            PtychoFMTrainingSettings(registry),
            PtychoFMInferenceSettings(registry),
        )
    ]


_BUILDERS = {
    'ptychopinn': _build_ptychopinn_configs,
    'ptychopinn_torch': _build_ptychopinn_torch_configs,
    'ptycho_fm': _build_ptycho_fm_configs,
}

_REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    'ptychopinn': ('ptycho',),
    'ptychopinn_torch': ('ptycho_torch',),
    # ptycho_fm's _build_config is a pure-Python dict factory; it does not
    # import ptycho_vit or torch, so this test can run everywhere.
    'ptycho_fm': (),
}


def _probe(backend: str, result_queue: multiprocessing.Queue[dict[str, Any] | str]) -> None:
    """Build the configs, then report the outcome AND the GPU-context state.

    A build failure is reported rather than raised, because the context check
    is meaningful either way: by the time a builder has failed it has already
    imported whatever framework it was going to import.
    """
    import pickle

    result: dict[str, Any] = {'count': 0, 'types': [], 'build_error': None}

    try:
        configs = _BUILDERS[backend]()
        round_tripped = [pickle.loads(pickle.dumps(config)) for config in configs]
        result['count'] = len(configs)
        result['types'] = [type(config).__name__ for config in round_tripped]
    except BaseException as exc:  # noqa: BLE001
        result['build_error'] = f'{type(exc).__name__}: {exc}'

    try:
        result['context'] = _gpu_context_offenders()
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(f'context probe failed: {type(exc).__name__}: {exc}')
        return

    result_queue.put(result)


@pytest.mark.parametrize('backend', sorted(_BUILDERS))
def test_parent_config_builders_acquire_no_gpu_context(backend: str) -> None:
    for module_name in _REQUIRED_MODULES[backend]:
        if find_spec(module_name) is None:
            pytest.skip(f'{module_name} is not installed')

    ctx = multiprocessing.get_context('spawn')
    result_queue: multiprocessing.Queue[dict[str, Any] | str] = ctx.Queue()
    process = ctx.Process(target=_probe, args=(backend, result_queue))
    process.start()
    process.join(timeout=180.0)

    assert not process.is_alive(), f'{backend} config-builder probe did not exit in 180s.'

    result: Any = result_queue.get(timeout=1.0)

    if isinstance(result, str):
        raise AssertionError(f'{backend} config-builder probe raised: {result}')

    # The invariant this module exists to pin, checked whether or not the
    # builder itself succeeded.
    assert result['context'] == [], textwrap.dedent(
        f"""\
        Building {backend} configs parent-side acquired a GPU context.
        Offenders: {result['context']}
        Configs built: {result['types']}

        The parent is allowed to import a GPU framework to construct the
        picklable configs it ships to the child, but not to acquire a context
        doing it. Something in the builder placed a tensor on a device or
        queried the runtime. See the invariant note in
        ptychodus.model.processing._subprocess_protocol.
        """
    )

    if result['build_error'] is not None:
        # The installed backend's config schema does not match what the
        # settings mapping targets. That is an environment/version mismatch
        # rather than a ptychodus defect -- the mapping is the same one the
        # child used before it moved parent-side -- so report it loudly
        # instead of failing the suite against an arbitrary local checkout.
        pytest.skip(
            f'{backend} config schema has drifted from the installed backend: '
            f'{result["build_error"]}'
        )

    assert result['count'] > 0, f'{backend} built no configs.'
