"""The PtychoFM overlay must reproduce the settings layout it replaces.

The keys and defaults are checked against the golden settings fixture, because
a missing key orphans a saved setting silently and a slipped default changes
what a run computes while every key still matches. And the config the child
receives is checked against the dict ``_build_config`` hands it today.

That second one is the property specific to this backend: upstream resolves
pretrained encoder sizes, validation batch size and dataset normalization from a
field having *no value*, so the declaration may set only what ptychodus surfaces
or freezes. ``model_dump(exclude_unset=True)`` is what preserves that, and it is
only correct if nothing else was set along the way.

The frozen half needs its own guard here, because it is nowhere else. The
architecture has no settings group, so the golden fixture does not see it; a
slipped ``embed_dim`` would change every run with no test to notice.
``_FROZEN`` is that guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json
import subprocess
import sys

import pytest

pytest.importorskip('ptycho_fm')

from ptychodus.model.settings import SettingsRegistry  # noqa: E402
from ptychodus.model.processing._ptycho_fm_sections import sections  # noqa: E402
from ptychodus.model.processing.flatten import flatten_options  # noqa: E402
from ptychodus.model.processing.generate import (  # noqa: E402
    apply_options,
    populate_parameter_group,
    restore_options,
)
from ptychodus.model.processing.parameter_store import OptionsSection  # noqa: E402
from ptychodus.plugins.ptycho_fm.options import default_options  # noqa: E402


def _golden() -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    current = ''

    for line in (Path(__file__).parent / 'data' / 'settings_defaults.ini').read_text().splitlines():
        if line.startswith('['):
            current = line.strip('[]')
            groups[current] = {}
        elif ' = ' in line and current:
            key, value = line.split(' = ', 1)
            groups[current][key] = value

    return groups


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_group_reproduces_its_persisted_keys(section: OptionsSection) -> None:
    root = getattr(default_options(), section.root_path)
    produced = {flat.ini_key for flat in flatten_options(root, section.policy)}
    assert produced == set(_golden()[section.group_name])


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_declared_default_matches_the_persisted_one(section: OptionsSection) -> None:
    root = getattr(default_options(), section.root_path)
    expected = _golden()[section.group_name]

    for flat in flatten_options(root, section.policy):
        assert flat.spelling.render(flat.default) == expected[flat.ini_key], flat.ini_key


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_key_carries_display_text(section: OptionsSection) -> None:
    """Upstream declares no title or description, so the table has to supply it."""
    root = getattr(default_options(), section.root_path)
    unlabelled = [f.ini_key for f in flatten_options(root, section.policy) if not f.label]
    assert not unlabelled


def _flat_by_key() -> dict[str, Any]:
    options = default_options()
    return {
        flat.ini_key: flat
        for section in sections()
        for flat in flatten_options(getattr(options, section.root_path), section.policy)
    }


# The three losses ptychodus offers, each defined on both objectives and taking
# no parameters of its own. Upstream declares eight more; adding one back here
# means persisting the sub-branch that configures it.
_EXPECTED_CHOICES: dict[str, tuple[str, ...]] = {
    'LossFunction': ('mse', 'l1', 'smooth_l1'),
}


def test_every_choice_field_offers_what_the_panel_offered() -> None:
    """Upstream types every choice as a bare str, so the members come from the table."""
    flat = _flat_by_key()
    assert {key: tuple(flat[key].choices) for key in _EXPECTED_CHOICES} == _EXPECTED_CHOICES


def test_no_other_key_claims_a_choice_list() -> None:
    """A stray entry would render a free value as a combo box of two."""
    with_choices = {key for key, flat in _flat_by_key().items() if flat.choices}
    assert with_choices == set(_EXPECTED_CHOICES)


class TestRoundTrip:
    def _build(self) -> tuple[SettingsRegistry, Any]:
        registry = SettingsRegistry()
        options = default_options()

        for section in sections():
            populate_parameter_group(
                registry.create_group(section.group_name),
                getattr(options, section.root_path),
                section.policy,
            )

        return registry, options

    def test_an_edit_reaches_the_restored_branch(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'PtychoFMTraining')
        registry['PtychoFMTraining'].parameters()['Epochs'].set_value(77)
        restored = restore_options(
            options.training, registry['PtychoFMTraining'].parameters(), section.policy
        )
        assert restored.epochs == 77

    def test_no_cap_on_files_persists_as_zero(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'PtychoFMData')
        parameters = registry['PtychoFMData'].parameters()
        assert parameters['MaxFiles'].get_value() == 0
        assert restore_options(options.data, parameters, section.policy).max_files is None

        parameters['MaxFiles'].set_value(12)
        assert restore_options(options.data, parameters, section.policy).max_files == 12


def _restored_config() -> Any:
    """The declaration carrying what the generated parameters hold at defaults."""
    registry = SettingsRegistry()
    options = default_options()
    bound = {
        section.root_path: populate_parameter_group(
            registry.create_group(section.group_name),
            getattr(options, section.root_path),
            section.policy,
        )
        for section in sections()
    }
    restored = copy.deepcopy(options)

    for section in sections():
        apply_options(
            getattr(restored, section.root_path), bound[section.root_path], section.policy
        )

    return restored


# What ptychodus sets without offering, and the values it sets. Everything
# under `model` is here because a .pth names no architecture, so the sizes a
# checkpoint was trained with are not a user's to vary; the four under `data`
# because they size the dataloader and the array cache to the machine.
#
# The values matter as much as the names. No settings group holds them, so the
# golden fixture never sees them, and this is the only place a changed one is
# noticed. Changing a value under `model` changes which checkpoints this build
# can load, so a diff here should be deliberate.
_FROZEN: dict[str, dict[str, Any]] = {
    'data': {
        'num_workers': 4,
        'prefetch_factor': 2,
        'cache_object': False,
        'cache_memory_budget_mb': 512,
    },
    'model': {
        'encoder_type': 'custom',
        'coupled_decoder_mode': 'amp_phase',
        'encoder': {
            'img_size': 256,
            'patch_size': 16,
            'embed_dim': 512,
            'depth': 12,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'use_cls_token': False,
            'dropout': 0.1,
            'attn_dropout': 0.0,
            'timm_model_name': 'vit_large_patch32_224',
        },
        'decoder': {
            'base_channels': 64,
            'latent_dim': 512,
            'num_stages': 4,
            'use_batchnorm': True,
            'dropout': 0.1,
        },
        'init': {'enabled': False, 'method': 'trunc_normal'},
    },
}


def _surfaced() -> dict[str, set[str]]:
    """The first path segment of every key each section persists."""
    options = _restored_config()
    return {
        section.root_path: {
            flat.path.split('.')[0]
            for flat in flatten_options(getattr(options, section.root_path), section.policy)
        }
        for section in sections()
    }


def test_only_the_fields_ptychodus_surfaces_or_freezes_are_set() -> None:
    """Anything else set here would be emitted, overriding a runtime-resolved default."""
    restored = _restored_config()
    surfaced = _surfaced()

    assert set(restored.model_fields_set) == set(surfaced) | set(_FROZEN)

    for root_path in set(surfaced) | set(_FROZEN):
        expected = surfaced.get(root_path, set()) | set(_FROZEN.get(root_path, {}))
        assert getattr(restored, root_path).model_fields_set == expected, root_path


def test_the_frozen_fields_hold_the_values_declared_for_them() -> None:
    """The architecture's only guard: no settings group holds it to compare against."""
    restored = _restored_config()

    assert restored.model_dump(exclude_unset=True, include={'model'})['model'] == _FROZEN['model']

    for name, value in _FROZEN['data'].items():
        assert getattr(restored.data, name) == value, name


def _dumps(name: str) -> tuple[Any, Any]:
    """The two payload builders one reconstructor was built with.

    Reached through the reconstructor rather than called directly, because what
    each operation sends is the thing worth pinning and the builders are
    closures over the mode.
    """
    from ptychodus.plugins.ptycho_fm.reconstructor import build_reconstructor

    built = build_reconstructor(name)
    return built._dump_reconstruct_options, built._dump_train_options


def test_the_config_the_child_receives_is_what_the_declaration_sets() -> None:
    """Only the surfaced fields, plus the two that are not settings at all.

    The hand-built dict this once compared against went with the settings
    classes it read. What remains worth pinning is the shape: upstream decides
    pretrained encoder sizes, validation batch size and dataset normalization
    from a field having no value, so a key appearing here that nobody chose
    would silently override one of those.
    """
    _, dump_train = _dumps('Unsupervised')
    config = dump_train(_restored_config())
    declared = _restored_config().model_dump(exclude_unset=True)

    # Neither is a setting: the objective follows from which reconstructor was
    # selected, and the encoder takes one channel in every configuration.
    assert config['training'].pop('mode') == 'unsupervised'
    assert config['model']['encoder'].pop('in_channels') == 1
    assert config == {branch: declared[branch] for branch in config}


def test_each_operation_is_sent_only_the_branches_it_reads() -> None:
    """Training values do not travel to an inference run, nor the reverse.

    Narrowing here is hygiene rather than a fix -- the child ignored what it
    did not read either way -- but a payload that changed when an unrelated
    setting was edited is one nobody can diff.
    """
    dump_reconstruct, dump_train = _dumps('Unsupervised')
    options = _restored_config()

    assert set(dump_reconstruct(None, options)) == {'data', 'inference', 'model'}
    assert set(dump_train(options)) == {'data', 'model', 'training'}


def test_a_surviving_branch_is_spelled_exactly_as_it_would_be_whole() -> None:
    """Narrowing intersects with `exclude_unset` rather than overriding it."""
    dump_reconstruct, _ = _dumps('Unsupervised')
    options = _restored_config()
    narrowed = dump_reconstruct(None, options)
    declared = options.model_dump(exclude_unset=True)

    assert narrowed['data'] == declared['data']
    assert narrowed['inference'] == declared['inference']


def test_the_objective_is_the_only_thing_the_mode_changes() -> None:
    supervised = _dumps('Supervised')[1](_restored_config())
    unsupervised = _dumps('Unsupervised')[1](_restored_config())

    assert supervised['training'].pop('mode') == 'supervised'
    assert unsupervised['training'].pop('mode') == 'unsupervised'
    assert supervised == unsupervised


# Run in a bare interpreter rather than a spawned multiprocessing child: a
# spawned child re-imports this module, which has already read the declaration.
_IMPORT_PROBE = """
import json, sys
from pydantic import BaseModel

# pydantic discovers its plugins when a model class is first created, not when
# it is imported, and one of them drags in logfire and the OpenTelemetry stack.
# ptychodus has long since created a model by the time it reads a declaration,
# so warm that up before measuring or the delta reports several hundred modules
# that are not PtychoFM's.
type('Warm', (BaseModel,), {'__annotations__': {'x': int}})

before = set(sys.modules)
import ptycho_fm.config
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def test_reading_the_declaration_costs_two_modules() -> None:
    """What makes an overlay on upstream possible here and not for PtychoPINN."""
    probe = subprocess.run(
        [sys.executable, '-c', _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=120.0,
        check=True,
    )
    assert json.loads(probe.stdout) == ['ptycho_fm', 'ptycho_fm.config']
