"""cdtools' declaration must reproduce the settings layout it persists.

The keys are the external contract -- a missing one orphans a saved setting silently --
and the golden values pin the defaults a run starts from. Needs no backend installed:
the declaration is ptychodus's own, which is also why it is read through
`model/declared.py` rather than through the flattener.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.declared import declare_in, describe, restore
from ptychodus.model.processing.core import _BRANCHES_BY_LIBRARY
from ptychodus.model.processing.parameter_store import OptionsBranch
from ptychodus.plugins.cdtools.options import (
    CDToolsOptimizer,
    CDToolsOptions,
    ScheduleOptions,
    StageOptions,
)

_BRANCHES = _BRANCHES_BY_LIBRARY['cdtools']


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


def _described(branch: OptionsBranch) -> list:
    root = CDToolsOptions()

    for name in branch.root_path.split('.'):
        root = getattr(root, name)

    return list(describe(root, prefix=branch.prefix))


def _ids(branch: OptionsBranch) -> str:
    return f'{branch.group_name}:{branch.root_path}'


@pytest.mark.parametrize('branch', _BRANCHES, ids=_ids)
def test_every_branch_lands_in_the_group_that_persists_it(branch: OptionsBranch) -> None:
    produced = {description.key for description in _described(branch)}
    assert produced <= set(_golden()[branch.group_name])


def test_the_branches_together_reproduce_every_persisted_key() -> None:
    """Separately each branch is a subset; together they must be the whole group."""
    produced: dict[str, set[str]] = {}

    for branch in _BRANCHES:
        keys = produced.setdefault(branch.group_name, set())
        keys.update(description.key for description in _described(branch))

    golden = _golden()
    assert produced == {name: set(golden[name]) for name in produced}


@pytest.mark.parametrize('branch', _BRANCHES, ids=_ids)
def test_every_declared_default_matches_the_persisted_one(branch: OptionsBranch) -> None:
    registry = SettingsRegistry()
    group = registry.create_group(branch.group_name)
    expected = _golden()[branch.group_name]

    for key, parameter in declare_in(group, _described(branch)).items():
        assert parameter.get_value_as_string() == expected[key], key


@pytest.mark.parametrize('branch', _BRANCHES, ids=_ids)
def test_every_key_carries_display_text(branch: OptionsBranch) -> None:
    unlabelled = [d.key for d in _described(branch) if not d.label]
    assert not unlabelled


def test_the_branches_cover_every_declared_branch() -> None:
    roots = {branch.root_path.split('.')[0] for branch in _BRANCHES}
    assert roots == set(CDToolsOptions.model_fields)


def test_each_stage_persists_under_its_own_prefix() -> None:
    keys = {
        description.key
        for branch in _BRANCHES
        if branch.group_name == 'CDToolsSchedule'
        for description in _described(branch)
    }
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
    def _build(self) -> SettingsRegistry:
        registry = SettingsRegistry()
        groups: dict[str, object] = {}

        for branch in _BRANCHES:
            group = groups.get(branch.group_name)

            if group is None:
                group = registry.create_group(branch.group_name)
                groups[branch.group_name] = group

            declare_in(group, _described(branch))  # type: ignore[arg-type]

        return registry

    def test_an_edit_reaches_the_restored_stage(self) -> None:
        registry = self._build()
        branch = next(b for b in _BRANCHES if b.root_path == 'schedule.stage2')
        parameters = registry['CDToolsSchedule'].parameters()
        parameters['Stage2Epochs'].set_value(7)

        restored = restore(CDToolsOptions().schedule.stage2, _described(branch), parameters)

        assert restored.epochs == 7
        assert restored.learning_rate == CDToolsOptions().schedule.stage2.learning_rate

    def test_enums_persist_by_value(self) -> None:
        registry = self._build()
        optimizer = registry['CDToolsOptimizer'].parameters()['Optimizer']
        assert optimizer.get_value() == 'Adam'
        assert registry['CDToolsFancyPtycho'].parameters()['Loss'].get_value() == 'amplitude_mse'

    def test_an_optimizer_choice_restores_as_its_member(self) -> None:
        registry = self._build()
        branch = next(b for b in _BRANCHES if b.root_path == 'optimizer')
        parameters = registry['CDToolsOptimizer'].parameters()
        parameters['Optimizer'].set_value('LBFGS')

        restored = restore(CDToolsOptions().optimizer, _described(branch), parameters)

        assert restored.optimizer is CDToolsOptimizer.LBFGS

    def test_a_bound_clamps_a_value_loaded_from_file(self) -> None:
        registry = self._build()
        parameter = registry['CDToolsOptimizer'].parameters()['AdamBeta1']
        parameter.set_value_from_string('2.0')
        assert parameter.get_value() == 0.999999
