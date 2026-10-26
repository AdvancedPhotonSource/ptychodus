"""cdtools' declaration must reproduce the settings layout it persists.

The keys are the external contract -- a missing one orphans a saved setting silently --
and the golden values pin the defaults a run starts from. Needs no backend installed:
the declaration is ptychodus's own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing._cdtools_sections import sections
from ptychodus.model.processing.flatten import flatten_options
from ptychodus.model.processing.generate import populate_parameter_group, restore_options
from ptychodus.model.processing.parameter_store import OptionsSection
from ptychodus.plugins.cdtools.options import (
    CDToolsOptimizer,
    CDToolsOptions,
    ScheduleOptions,
    StageOptions,
)


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
    root = getattr(CDToolsOptions(), section.root_path)
    produced = {flat.ini_key for flat in flatten_options(root, section.policy)}
    assert produced == set(_golden()[section.group_name])


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_declared_default_matches_the_persisted_one(section: OptionsSection) -> None:
    root = getattr(CDToolsOptions(), section.root_path)
    expected = _golden()[section.group_name]

    for flat in flatten_options(root, section.policy):
        assert flat.spelling.render(flat.default) == expected[flat.ini_key], flat.ini_key


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_key_carries_display_text(section: OptionsSection) -> None:
    root = getattr(CDToolsOptions(), section.root_path)
    unlabelled = [f.ini_key for f in flatten_options(root, section.policy) if not f.label]
    assert not unlabelled


def test_the_sections_cover_every_declared_branch() -> None:
    assert {section.root_path for section in sections()} == set(CDToolsOptions.model_fields)


def test_each_stage_persists_under_its_own_prefix() -> None:
    section = next(s for s in sections() if s.group_name == 'CDToolsSchedule')
    keys = {f.ini_key for f in flatten_options(CDToolsOptions().schedule, section.policy)}
    assert keys == {
        f'Stage{number}{field}'
        for number in (1, 2, 3)
        for field in ('Epochs', 'LearningRate', 'BatchSize')
    }


def test_a_stage_with_no_epochs_is_skipped() -> None:
    schedule = ScheduleOptions(
        stage1=StageOptions(epochs=4),
        stage2=StageOptions(epochs=0),
        stage3=StageOptions(epochs=2, learning_rate=0.5),
    )
    assert [stage.epochs for stage in schedule.stages()] == [4, 2]
    assert schedule.stages()[1].learning_rate == 0.5


class TestRoundTrip:
    def _build(self) -> tuple[SettingsRegistry, CDToolsOptions]:
        registry = SettingsRegistry()
        options = CDToolsOptions()

        for section in sections():
            populate_parameter_group(
                registry.create_group(section.group_name),
                getattr(options, section.root_path),
                section.policy,
            )

        return registry, options

    def test_an_edit_reaches_the_restored_stage(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'CDToolsSchedule')
        parameters = registry['CDToolsSchedule'].parameters()
        parameters['Stage2Epochs'].set_value(7)
        restored = restore_options(options.schedule, parameters, section.policy)
        assert restored.stage2.epochs == 7
        assert restored.stage1.epochs == options.schedule.stage1.epochs

    def test_enums_persist_by_value(self) -> None:
        registry, _ = self._build()
        optimizer = registry['CDToolsOptimizer'].parameters()['Optimizer']
        assert optimizer.get_value() == 'Adam'
        assert registry['CDToolsFancyPtycho'].parameters()['Loss'].get_value() == 'amplitude_mse'

    def test_an_optimizer_choice_restores_as_its_member(self) -> None:
        registry, options = self._build()
        section = next(s for s in sections() if s.group_name == 'CDToolsOptimizer')
        parameters = registry['CDToolsOptimizer'].parameters()
        parameters['Optimizer'].set_value('LBFGS')
        restored = restore_options(options.optimizer, parameters, section.policy)
        assert restored.optimizer is CDToolsOptimizer.LBFGS

    def test_a_bound_clamps_a_value_loaded_from_file(self) -> None:
        registry, _ = self._build()
        parameter = registry['CDToolsOptimizer'].parameters()['AdamBeta1']
        parameter.set_value_from_string('2.0')
        assert parameter.get_value() == 0.999999
