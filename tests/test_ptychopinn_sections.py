"""PtychoPINN's declaration must reproduce the settings layout it replaces.

Both halves matter. The keys are the external contract -- a missing one orphans
a saved setting silently -- and the values are hand-transcribed from the
settings classes, so a slipped default would change what a reconstruction runs
with while every key still matched.

Needs no backend installed: the declaration is ptychodus's own precisely
because importing PtychoPINN's would pull torch into the parent. That is also
why it is read through `model/declared.py` rather than through the flattener.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.declared import declare_in, describe, restore
from ptychodus.model.processing.core import _BRANCHES_BY_LIBRARY
from ptychodus.model.processing.parameter_store import OptionsBranch
from ptychodus.plugins.ptychopinn.options import Accelerator, PtychoPINNOptions

_BRANCHES = _BRANCHES_BY_LIBRARY['PtychoPINN']


def _described(branch: OptionsBranch) -> list:
    return list(describe(getattr(PtychoPINNOptions(), branch.root_path), prefix=branch.prefix))


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


@pytest.mark.parametrize('branch', _BRANCHES, ids=lambda b: b.group_name)
def test_every_group_reproduces_its_persisted_keys(branch: OptionsBranch) -> None:
    produced = {description.key for description in _described(branch)}
    assert produced == set(_golden()[branch.group_name])


@pytest.mark.parametrize('branch', _BRANCHES, ids=lambda b: b.group_name)
def test_every_declared_default_matches_the_persisted_one(branch: OptionsBranch) -> None:
    registry = SettingsRegistry()
    group = registry.create_group(branch.group_name)
    expected = _golden()[branch.group_name]

    for key, parameter in declare_in(group, _described(branch)).items():
        assert parameter.get_value_as_string() == expected[key], key


@pytest.mark.parametrize('branch', _BRANCHES, ids=lambda b: b.group_name)
def test_every_key_carries_display_text(branch: OptionsBranch) -> None:
    """The declaration states its own, so a blank label means a missing Field title."""
    unlabelled = [d.key for d in _described(branch) if not d.label]
    assert not unlabelled


class TestRoundTrip:
    def _build(self) -> SettingsRegistry:
        registry = SettingsRegistry()

        for branch in _BRANCHES:
            declare_in(registry.create_group(branch.group_name), _described(branch))

        return registry

    def test_an_edit_reaches_the_restored_branch(self) -> None:
        registry = self._build()
        branch = next(b for b in _BRANCHES if b.group_name == 'PtychoPINNTraining')
        parameters = registry['PtychoPINNTraining'].parameters()
        parameters['Epochs'].set_value(77)

        restored = restore(PtychoPINNOptions().training, _described(branch), parameters)

        assert restored.epochs == 77

    def test_enums_persist_by_value_not_member_name(self) -> None:
        """PtychoPINN writes 'auto', where pty-chi would write the member name."""
        registry = self._build()
        assert registry['PtychoPINNTraining'].parameters()['Accelerator'].get_value() == 'auto'
        assert Accelerator.AUTO.name == 'AUTO'

    def test_a_bound_clamps_a_value_loaded_from_file(self) -> None:
        registry = self._build()
        parameter = registry['PtychoPINNData'].parameters()['XUpperBound']
        parameter.set_value_from_string('5.0')
        assert parameter.get_value() == 1.0
