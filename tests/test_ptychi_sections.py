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

import pytest

pytest.importorskip('ptychi')

from ptychodus.model.processing._ptychi_sections import sections_for  # noqa: E402
from ptychodus.model.processing.flatten import flatten_options  # noqa: E402
from ptychodus.model.ptychi.algorithms import task_options_classes  # noqa: E402

# Not derived from the declaration: NumSyncEpochs chunks the epoch loop for
# progress reporting and has no pty-chi equivalent, and UseDevices is device
# selection. Both stay in a ptychodus-owned settings group.
PTYCHODUS_OWNED = frozenset({'NumSyncEpochs', 'UseDevices'})

ALGORITHMS = sorted(task_options_classes())


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
    options = task_options_classes()[algorithm]()
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
        expected = golden[group] - PTYCHODUS_OWNED
        assert produced == expected, f'{algorithm}/{group}'


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
        assert sections_for(algorithm, task_options_classes()[algorithm]())
