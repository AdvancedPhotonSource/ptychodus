"""The pty-chi section table must reproduce the persisted settings layout exactly.

Every key ptychodus writes today has to come back out of the declaration, for
every algorithm. A missing one orphans a saved setting silently -- the parameter
falls back to its default and a run proceeds with a value nobody chose -- and a
spurious one invents a key no settings file has.

Driven across all eight algorithm option classes rather than one, because
pty-chi subclasses every branch per algorithm and a table that satisfies LSQML
can still be wrong for PIE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pytest

pytest.importorskip('ptychi')

from ptychodus.model.processing import _ptychi_sections  # noqa: E402
from ptychodus.model.processing._ptychi_sections import sections_for  # noqa: E402
from ptychodus.model.processing.flatten import flatten_options  # noqa: E402
from ptychodus.plugins.ptychi.algorithms import task_options_classes  # noqa: E402
from ptychodus.plugins.ptychi.options import default_options  # noqa: E402

ALGORITHMS = sorted(task_options_classes())


def _declared(algorithm: str) -> Any:
    """What the reconstructor for *algorithm* declares, wrapper and all."""
    return default_options(task_options_classes()[algorithm])


def _golden() -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    current = ''

    for line in (Path(__file__).parent / 'data' / 'settings_defaults.ini').read_text().splitlines():
        if line.startswith('['):
            current = line.strip('[]')
            groups[current] = set()
        elif ' = ' in line and current:
            groups[current].add(line.split(' = ')[0])

    return groups


def _resolve(root: Any, path: str) -> Any:
    for name in path.split('.') if path else ():
        root = getattr(root, name)

    return root


def _produced(algorithm: str) -> dict[str, set[str]]:
    options = _declared(algorithm)
    keys: dict[str, set[str]] = {}

    for section in sections_for(algorithm, options):
        root = _resolve(options, section.root_path)
        keys.setdefault(section.group_name, set()).update(
            flat.ini_key for flat in flatten_options(root, section.policy)
        )

    return keys


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_every_group_reproduces_its_persisted_keys(algorithm: str) -> None:
    golden = _golden()

    for group, produced in sorted(_produced(algorithm).items()):
        assert produced == golden[group], f'{algorithm}/{group}'


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_no_persisted_group_is_left_unfed(algorithm: str) -> None:
    """Every PtyChi* group a settings file carries must have a section behind it."""
    produced = set(_produced(algorithm))
    shared = {'PtyChi', 'PtyChiObject', 'PtyChiProbe', 'PtyChiProbePosition', 'PtyChiOPR'}
    assert shared <= produced


def test_the_table_covers_every_algorithm() -> None:
    """A new algorithm must not silently reach the GUI with no settings behind it."""
    assert len(ALGORITHMS) == 8

    for algorithm in ALGORITHMS:
        assert sections_for(algorithm, _declared(algorithm))


# A plan's triple is drawn as one composite row labelled 'Plan:', so the three
# keys behind it carry no text of their own.
_PLAN_SUFFIXES = ('Start', 'Stop', 'Stride')


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_every_key_a_panel_shows_has_a_label(algorithm: str) -> None:
    """An upstream declaration carries no text, so a missing entry renders blank."""
    options = _declared(algorithm)
    unlabelled = []

    for section in sections_for(algorithm, options):
        for flat in flatten_options(_resolve(options, section.root_path), section.policy):
            if not flat.ini_key.endswith(_PLAN_SUFFIXES) and not flat.label:
                unlabelled.append(f'{section.group_name}.{flat.ini_key}')

    assert not unlabelled


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_plan_keys_are_left_to_the_composite_that_draws_them(algorithm: str) -> None:
    options = _declared(algorithm)

    for section in sections_for(algorithm, options):
        for flat in flatten_options(_resolve(options, section.root_path), section.policy):
            if flat.ini_key.endswith(_PLAN_SUFFIXES):
                assert not flat.label, flat.ini_key


def test_no_overlay_entry_names_a_field_no_branch_declares() -> None:
    """A dead entry is silent: its text simply never reaches a key.

    Two survived the port from the panels because pty-chi names the field `rho`
    on each branch while the panels called them probe rho and probe position
    rho, and the per-branch tables already carried the right text.
    """
    consulted: set[str] = set()

    for algorithm in ALGORITHMS:
        options = _declared(algorithm)

        for section in sections_for(algorithm, options):
            for flat in flatten_options(_resolve(options, section.root_path), section.policy):
                consulted.add(flat.path)

                if flat.structural_role:
                    consulted.add(flat.structural_role)

    declared: set[str] = set()

    for name, table in vars(_ptychi_sections).items():
        if name.endswith('_OVERLAYS') and isinstance(table, dict):
            declared |= set(table)

    assert declared - consulted == set()


def _settings_bounds() -> dict[tuple[str, str], tuple[Any, Any]]:
    """The bounds the hand-written settings classes carried, frozen as a fixture.

    The oracle for bounds, as settings_defaults.ini is the oracle for keys and
    defaults: an INI records values, not the ranges they are held to, so that
    fixture cannot carry these. Read off the live parameters those classes
    built, and written down here when they were deleted -- a range is not
    recoverable from anything still in the tree, so freezing it is the only
    way the guarantee outlives its source.
    """
    frozen = json.loads(
        (Path(__file__).parent / 'data' / 'ptychi_settings_bounds.json').read_text()
    )

    return {
        (group, key): (bounds[0], bounds[1])
        for group, keys in frozen.items()
        for key, bounds in keys.items()
    }


def test_every_bound_the_settings_classes_carry_survives_generation() -> None:
    """A bound lost here is a range the GUI cannot build a slider or spin box from.

    Asserted one way only. Upstream annotates bounds the settings classes never
    had, and those are a gain; a bound that moved, or went missing, is not.
    """
    produced: dict[tuple[str, str], Any] = {}

    for algorithm in ALGORITHMS:
        options = _declared(algorithm)

        for section in sections_for(algorithm, options):
            for flat in flatten_options(_resolve(options, section.root_path), section.policy):
                produced.setdefault((section.group_name, flat.ini_key), flat)

    for (group, key), expected in sorted(_settings_bounds().items()):
        flat = produced[group, key]
        assert (flat.minimum, flat.maximum) == expected, f'{group}/{key}'
