"""The PtychoFM overlay must reproduce the settings layout it replaces.

Three contracts, each with its own oracle. The keys and defaults are checked
against the golden settings fixture, because a missing key orphans a saved
setting silently and a slipped default changes what a run computes while every
key still matches. The choice lists are checked against ``PtychoFMEnumerators``,
the hand-kept lists the policy table replaces. And the config the child receives
is checked against the dict ``_build_config`` hands it today.

That last one is the property specific to this backend: upstream resolves
pretrained encoder sizes, validation batch size and dataset normalization from a
field having *no value*, so the declaration may set only what ptychodus
surfaces. ``model_dump(exclude_unset=True)`` is what preserves that, and it is
only correct if nothing else was set along the way.
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

from ptychodus.api.parameters import IntegerParameter, RealParameter  # noqa: E402
from ptychodus.api.settings import SettingsRegistry  # noqa: E402
from ptychodus.model.processing._ptycho_fm_sections import sections  # noqa: E402
from ptychodus.model.processing.flatten import flatten_options, render_value  # noqa: E402
from ptychodus.model.processing.generate import (  # noqa: E402
    apply_options,
    populate_parameter_group,
    restore_options,
)
from ptychodus.model.processing.parameter_store import OptionsSection  # noqa: E402
from ptychodus.plugins.ptycho_fm import settings as ptycho_fm_settings  # noqa: E402
from ptychodus.plugins.ptycho_fm.enums import PtychoFMEnumerators  # noqa: E402
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
        assert render_value(flat.default, section.policy) == expected[flat.ini_key], flat.ini_key


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


def _expected_choices() -> dict[str, tuple[str, ...]]:
    enumerators = PtychoFMEnumerators()
    loss_types = tuple(enumerators.get_loss_types())
    return {
        'encoder_type': tuple(enumerators.get_encoder_types()),
        'coupled_decoder_mode': tuple(enumerators.get_coupled_decoder_modes()),
        'init_method': tuple(enumerators.get_init_methods()),
        'loss_function': tuple(enumerators.get_loss_functions()),
        'weighted_loss_type': loss_types,
        'probe_aware_loss_type': loss_types,
        'q_dependent_loss_type': loss_types,
    }


def test_every_choice_field_offers_what_the_panel_offered() -> None:
    """Upstream types every choice as a bare str, so the members come from the table."""
    flat = _flat_by_key()
    assert {key: tuple(flat[key].choices) for key in _expected_choices()} == _expected_choices()


def test_no_other_key_claims_a_choice_list() -> None:
    """A stray entry would render a free value as a combo box of two."""
    with_choices = {key for key, flat in _flat_by_key().items() if flat.choices}
    assert with_choices == set(_expected_choices())


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
        registry['PtychoFMTraining'].parameters()['epochs'].set_value(77)
        restored = restore_options(
            options.training, registry['PtychoFMTraining'].parameters(), section.policy
        )
        assert restored.epochs == 77

    def test_no_cap_on_files_persists_as_zero(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'PtychoFMData')
        parameters = registry['PtychoFMData'].parameters()
        assert parameters['max_files'].get_value() == 0
        assert restore_options(options.data, parameters, section.policy).max_files is None

        parameters['max_files'].set_value(12)
        assert restore_options(options.data, parameters, section.policy).max_files == 12

    def test_the_uncoupled_decoder_persists_as_none(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'PtychoFMModel')
        parameters = registry['PtychoFMModel'].parameters()
        assert parameters['coupled_decoder_mode'].get_value() == 'amp_phase'

        parameters['coupled_decoder_mode'].set_value('none')
        restored = restore_options(options.model, parameters, section.policy)
        assert restored.coupled_decoder_mode is None


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


def test_only_the_fields_ptychodus_surfaces_are_set() -> None:
    """Anything else set here would be emitted, overriding a runtime-resolved default."""
    restored = _restored_config()
    surfaced = {
        section.root_path: {
            flat.path.split('.')[0]
            for flat in flatten_options(getattr(restored, section.root_path), section.policy)
        }
        for section in sections()
    }

    assert set(restored.model_fields_set) == set(surfaced)

    for root_path, names in surfaced.items():
        assert getattr(restored, root_path).model_fields_set == names, root_path


def test_the_dump_reproduces_the_config_built_by_hand() -> None:
    """The nested shape and every value the child reads, from the declaration instead."""
    from ptychodus.plugins.ptycho_fm.reconstructor import _build_config
    from ptychodus.plugins.ptycho_fm.settings import (
        PtychoFMDataSettings,
        PtychoFMInferenceSettings,
        PtychoFMModelSettings,
        PtychoFMTrainingSettings,
    )

    registry = SettingsRegistry()
    by_hand = _build_config(
        'Unsupervised',
        PtychoFMDataSettings(registry),
        PtychoFMModelSettings(registry),
        PtychoFMTrainingSettings(registry),
        PtychoFMInferenceSettings(registry),
    )

    # Neither is a setting: the objective follows from which reconstructor was
    # selected, and the encoder takes one channel in every configuration.
    assert by_hand['training'].pop('mode') == 'unsupervised'
    assert by_hand['model']['encoder'].pop('in_channels') == 1

    assert _restored_config().model_dump(exclude_unset=True) == by_hand


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


def _settings_bounds() -> dict[tuple[str, str], tuple[Any, Any]]:
    """The bounds the hand-written settings classes carry, read off live parameters.

    The oracle for bounds, as settings_defaults.ini is the oracle for keys and
    defaults: an INI records values, not the ranges they are held to. Read from
    constructed parameters rather than parsed out of the source, which costs
    nothing because the settings classes import no part of PtychoFM. Retires
    with the classes it reads.
    """
    registry = SettingsRegistry()

    for name, cls in vars(ptycho_fm_settings).items():
        if isinstance(cls, type) and name.endswith('Settings'):
            cls(registry)

    bounds: dict[tuple[str, str], tuple[Any, Any]] = {}

    for group_name in registry:
        for key, parameter in registry[group_name].parameters().items():
            if not isinstance(parameter, IntegerParameter | RealParameter):
                continue

            minimum = parameter.get_minimum()
            maximum = parameter.get_maximum()

            if minimum is not None or maximum is not None:
                bounds[group_name, key] = (minimum, maximum)

    return bounds


def test_every_bound_the_settings_classes_carry_survives_generation() -> None:
    """A bound lost here is a range the GUI cannot build a slider or spin box from.

    Upstream annotates none of its 165 fields, so every one of these comes from
    the policy -- except the inference branch, which is ptychodus's own and
    states its bounds inline.
    """
    options = default_options()
    produced = {
        (section.group_name, flat.ini_key): flat
        for section in sections()
        for flat in flatten_options(getattr(options, section.root_path), section.policy)
    }

    for (group, key), expected in sorted(_settings_bounds().items()):
        flat = produced[group, key]
        assert (flat.minimum, flat.maximum) == expected, f'{group}/{key}'
