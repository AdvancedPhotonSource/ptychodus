"""PtychoPINN's declaration must reproduce the settings layout it replaces.

Both halves matter. The keys are the external contract -- a missing one orphans
a saved setting silently -- and the values are hand-transcribed from the
settings classes, so a slipped default would change what a reconstruction runs
with while every key still matched.

Needs no backend installed: the declaration is ptychodus's own precisely
because importing PtychoPINN's would pull torch into the parent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing._ptychopinn_sections import sections
from ptychodus.model.processing.flatten import flatten_options
from ptychodus.model.processing.generate import populate_parameter_group, restore_options
from ptychodus.model.processing.parameter_store import OptionsSection
from ptychodus.plugins.ptychopinn.options import Accelerator, PtychoPINNOptions


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
    options = PtychoPINNOptions()
    root = getattr(options, section.root_path)
    produced = {flat.ini_key for flat in flatten_options(root, section.policy)}
    assert produced == set(_golden()[section.group_name])


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_declared_default_matches_the_persisted_one(section: OptionsSection) -> None:
    options = PtychoPINNOptions()
    root = getattr(options, section.root_path)
    expected = _golden()[section.group_name]

    for flat in flatten_options(root, section.policy):
        assert flat.spelling.render(flat.default) == expected[flat.ini_key], flat.ini_key


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_key_carries_display_text(section: OptionsSection) -> None:
    """The declaration states its own, so a blank label means a missing Field title."""
    options = PtychoPINNOptions()
    root = getattr(options, section.root_path)
    unlabelled = [f.ini_key for f in flatten_options(root, section.policy) if not f.label]
    assert not unlabelled


class TestRoundTrip:
    def _build(self) -> tuple[SettingsRegistry, PtychoPINNOptions]:
        registry = SettingsRegistry()
        options = PtychoPINNOptions()

        for section in sections():
            populate_parameter_group(
                registry.create_group(section.group_name),
                getattr(options, section.root_path),
                section.policy,
            )

        return registry, options

    def test_an_edit_reaches_the_restored_branch(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'PtychoPINNTraining')
        registry['PtychoPINNTraining'].parameters()['epochs'].set_value(77)
        restored = restore_options(
            options.training, registry['PtychoPINNTraining'].parameters(), section.policy
        )
        assert restored.epochs == 77

    def test_enums_persist_by_value_not_member_name(self) -> None:
        """PtychoPINN writes 'auto', where pty-chi would write the member name."""
        registry, _ = self._build()
        assert registry['PtychoPINNTraining'].parameters()['accelerator'].get_value() == 'auto'
        assert Accelerator.AUTO.name == 'AUTO'

    def test_a_bound_clamps_a_value_loaded_from_file(self) -> None:
        registry, _ = self._build()
        parameter = registry['PtychoPINNData'].parameters()['x_upper_bound']
        parameter.set_value_from_string('5.0')
        assert parameter.get_value() == 1.0
