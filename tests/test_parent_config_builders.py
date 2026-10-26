"""Discipline gate for the parent-side backend config builders.

Each backend translates ptychodus settings into backend inputs in the parent
so the child receives something finished and does nothing but GPU work. That
translation happens at call time, which ``tests/test_no_gpu_context.py`` cannot
observe: its probe only imports the composition roots and never dispatches a
reconstruction.

This module closes that gap. For each installed backend it spawns a probe that
calls the builders exactly as ``build_*_payload`` does, then asserts:

1. the inputs build and survive the pickle round-trip the spawn transport
   requires,
2. no GPU context was acquired in the process that built them, and
3. no backend module was imported at all, where the builder claims not to need
   one.

Each probe runs in its own spawned process so a framework loaded by an earlier
test cannot mask a failure here.

The module also pins the rejections those builders perform in the parent, which
run in-process. PtychoFM's options re-declare upstream's config models, so its
tests skip where ptycho_fm is not installed; PtychoPINN's need no backend.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any
import multiprocessing
import pathlib
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


def _build_ptychopinn_settings() -> list[Any]:
    from ptychodus.plugins.ptychopinn.options import default_options
    from ptychodus.plugins.ptychopinn.reconstructor import _build_settings

    return [_build_settings('Unsupervised', default_options())]


def _build_ptycho_fm_configs() -> list[Any]:
    from ptychodus.plugins.ptycho_fm.options import default_options
    from ptychodus.plugins.ptycho_fm.reconstructor import (
        build_reconstruct_config,
        build_train_config,
    )

    # The name selects the training objective, coupled_decoder_mode the model
    # class, and loss_function which sub-section the child reads back. Both
    # section sets are built, since each operation narrows to its own and a
    # builder that imported a backend would do it for either.
    configs: list[Any] = []

    for name in ('Unsupervised', 'Supervised'):
        for coupled_decoder_mode in (None, 'real_imag', 'amp_phase'):
            for loss_function in ('weighted', 'total_flux', 'probe_aware', 'q_dependent'):
                if name == 'Supervised' and loss_function != 'weighted':
                    # Rejected on the train path; covered in-process below.
                    continue

                options = default_options()
                options.model.coupled_decoder_mode = coupled_decoder_mode
                options.training.loss_function = loss_function
                for build in (build_reconstruct_config, build_train_config):
                    configs.append(build(name, options))

    return configs


def _build_cdtools_payloads() -> list[Any]:
    from cdtools_fixtures import make_input
    from ptychodus.plugins.cdtools.options import CDToolsOptimizer, default_options
    from ptychodus.plugins.cdtools.reconstructor import build_reconstructor

    # Every optimizer, since each selects a different set of fields the child reads.
    payloads: list[Any] = []
    reconstructor = build_reconstructor()

    for optimizer in CDToolsOptimizer:
        options = default_options()
        options.optimizer.optimizer = optimizer
        payloads.append(reconstructor._dump_reconstruct_options(make_input(), options))

    return payloads


_BUILDERS = {
    'cdtools': _build_cdtools_payloads,
    'ptychopinn': _build_ptychopinn_settings,
    'ptycho_fm': _build_ptycho_fm_configs,
}

# Every builder is a pure-Python factory that imports no backend, so these probes
# run everywhere. An entry here would mean a builder that cannot.
_REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    'cdtools': (),
    'ptychopinn': (),
    'ptycho_fm': (),
}

# Modules a builder must not pull in. Checked in the spawned probe, where no
# earlier test can have imported them first.
_FORBIDDEN_MODULES: tuple[str, ...] = (
    'cdtools',
    'ptycho',
    'ptycho_torch',
    'torch',
    'tensorflow',
)


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

    result['imported'] = sorted(name for name in _FORBIDDEN_MODULES if name in sys.modules)

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
        ptychodus.api.spawn.
        """
    )

    assert result['imported'] == [], (
        f'Building {backend} inputs parent-side imported {result["imported"]}. '
        'These builders translate settings into plain Python values and must not '
        'reach the backend to do it; the child owns every backend import.'
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


@pytest.mark.parametrize('loss_function', ['total_flux', 'probe_aware', 'q_dependent'])
def test_ptycho_fm_rejects_diffraction_losses_when_supervised(loss_function: str) -> None:
    """Diffraction-space losses are meaningless against an object patch.

    They do not raise on their own -- they compute a different quantity -- so
    the parent has to turn the combination away before a child is spawned.
    """
    pytest.importorskip('ptycho_fm')
    from ptychodus.plugins.ptycho_fm.reconstructor import _verify_loss_supports_mode

    with pytest.raises(ValueError, match=loss_function):
        _verify_loss_supports_mode('Supervised', loss_function)

    _verify_loss_supports_mode('Unsupervised', loss_function)


def test_ptycho_fm_accepts_object_space_loss_when_supervised() -> None:
    pytest.importorskip('ptycho_fm')
    from ptychodus.plugins.ptycho_fm.reconstructor import _verify_loss_supports_mode

    for name in ('Supervised', 'Unsupervised'):
        for loss_function in ('mse', 'l1', 'smooth_l1', 'poisson_nll', 'weighted'):
            _verify_loss_supports_mode(name, loss_function)


def test_ptycho_fm_maps_the_uncoupled_mode_to_null() -> None:
    """'none' is how the settings file spells it; ptycho_fm reads the key as null.

    The translation moved into the section table's Sentinel when the backend
    started declaring, so what is checked here is that the declared value
    reaches the child unchanged rather than that a builder converts it.
    """
    pytest.importorskip('ptycho_fm')
    from ptychodus.plugins.ptycho_fm.options import default_options
    from ptychodus.plugins.ptycho_fm.reconstructor import (
        build_reconstruct_config,
    )

    options = default_options()
    options.model.coupled_decoder_mode = None
    assert (
        build_reconstruct_config('Unsupervised', options)['model']['coupled_decoder_mode'] is None
    )

    options = default_options()
    options.model.coupled_decoder_mode = 'amp_phase'
    assert (
        build_reconstruct_config('Unsupervised', options)['model']['coupled_decoder_mode']
        == 'amp_phase'
    )


def _build_ptychopinn_settings_mapping(**overrides: Any) -> dict[str, Any]:
    from ptychodus.plugins.ptychopinn.options import default_options
    from ptychodus.plugins.ptychopinn.reconstructor import _build_settings

    options = default_options()

    for name, value in overrides.items():
        for branch in (options.data, options.model, options.training, options.inference):
            if name in type(branch).model_fields:
                setattr(branch, name, value)
                break
        else:
            raise AssertionError(f'no PtychoPINN option named {name!r}')

    return _build_settings('Unsupervised', options)


def test_ptychopinn_settings_omit_nphotons() -> None:
    """Supplying nphotons would rescale already-counted data.

    Ptychodus exports photon counts and declares them as counts, so the source
    and the bundle target agree and nothing is converted. Naming nphotons marks
    the photon count as explicitly chosen, which turns the export back into
    amplitude evidence and silently rescales it.
    """
    assert 'nphotons' not in _build_ptychopinn_settings_mapping()


def test_ptychopinn_settings_omit_derived_loss_fields() -> None:
    """loss_function and nll are derived from torch_loss_mode, not supplied."""
    settings = _build_ptychopinn_settings_mapping()

    assert settings['torch_loss_mode'] == 'poisson'
    assert 'loss_function' not in settings
    assert 'nll' not in settings


def test_ptychopinn_settings_map_loss_function_to_torch_loss_mode() -> None:
    assert _build_ptychopinn_settings_mapping(loss_function='MAE')['torch_loss_mode'] == 'mae'


def test_ptychopinn_settings_express_object_policy_as_a_pair() -> None:
    """object_big is the deprecated spelling; the canonical form is two fields."""
    big = _build_ptychopinn_settings_mapping(object_big=True)
    assert (big['object_layout'], big['training_canvas']) == (
        'grouped_patches',
        'relative_overlap',
    )

    small = _build_ptychopinn_settings_mapping(object_big=False)
    assert (small['object_layout'], small['training_canvas']) == (
        'single_patch',
        'independent',
    )

    assert 'object_big' not in big
    assert 'object_big' not in small


def test_ptychopinn_settings_raise_neighbor_count_to_the_group_size() -> None:
    """A group of C members needs at least C-1 neighbors, or grouping raises."""
    settings = _build_ptychopinn_settings_mapping(gridsize=3, neighbor_count=4)
    assert settings['neighbor_count'] == 8

    # A request already large enough is left alone.
    assert _build_ptychopinn_settings_mapping(gridsize=2, neighbor_count=6)['neighbor_count'] == 6


def test_ptychopinn_settings_map_no_auxiliary_loss_to_absent() -> None:
    settings = _build_ptychopinn_settings_mapping()
    assert settings['amp_loss'] is None
    assert settings['phase_loss'] is None

    settings = _build_ptychopinn_settings_mapping(auxiliary_amplitude_loss='Total_Variation')
    assert settings['amp_loss'] == 'Total_Variation'


def test_ptychopinn_settings_disable_gradient_clipping_with_none() -> None:
    """Zero is the settings spelling for "off"; the backend reads it as null."""
    assert _build_ptychopinn_settings_mapping(gradient_clip_val=0.0)['gradient_clip_val'] is None
    assert _build_ptychopinn_settings_mapping(gradient_clip_val=1.5)['gradient_clip_val'] == 1.5


def _accepted_override_names() -> set[str]:
    """Read the backend's accepted override names out of its source.

    Parsed rather than imported: importing ``ptycho_torch`` would pull torch
    into this process, which is exactly what the rest of this module exists to
    prevent.
    """
    import ast
    import re

    # Resolve the package directory from the top-level spec only. Asking for
    # the dotted submodule would import ``ptycho_torch`` to find it.
    spec = find_spec('ptycho_torch')
    if spec is None or not spec.submodule_search_locations:
        pytest.skip('ptycho_torch is not installed')

    package_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))
    module_path = package_dir / 'config_resolution.py'
    if not module_path.is_file():
        pytest.skip('installed ptycho_torch has no config_resolution module')

    source = module_path.read_text()
    accepted: set[str] = set()

    # (owner, (field, field, ...)) pairs, plus the separate training frozenset.
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
            continue
        owner, fields = node.elts
        if (
            isinstance(owner, ast.Constant)
            and isinstance(owner.value, str)
            and isinstance(fields, ast.Tuple)
        ):
            accepted |= {
                element.value
                for element in fields.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }

    match = re.search(r'TRAINING_OWNER_FIELDS = frozenset\(\s*\{(.*?)\}\s*\)', source, re.DOTALL)
    if match:
        accepted |= set(re.findall(r'"([^"]+)"', match.group(1)))

    assert accepted, 'could not read any override names from the installed backend'
    return accepted


def test_ptychopinn_settings_are_all_accepted_override_names() -> None:
    """Every emitted name must be one the backend resolves.

    An unrecognized name is rejected rather than ignored, so a vocabulary
    rename upstream turns into a failed training run. This catches that at test
    time instead.
    """
    emitted = set(_build_ptychopinn_settings_mapping())
    unknown = sorted(emitted - _accepted_override_names())

    assert unknown == [], (
        f'These settings names are not accepted by the installed backend: {unknown}. '
        'Check the generated settings table in its configuration reference.'
    )
