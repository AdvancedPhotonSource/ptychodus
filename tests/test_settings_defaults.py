"""Golden fixture pinning every settings group's sections, keys and default values.

The INI layout is an external contract: saved settings files, batch runs and the
reconstruction drivers all depend on the exact section and key names. A rename
orphans the old key silently -- ``SettingsRegistry`` warns, but the parameter
still falls back to its default and a run proceeds with values nobody chose.
Across 500-plus parameters that surface is far too large to review by eye, so it
is pinned here instead.

Settings classes are discovered rather than listed, so a new one is covered the
day it is added. Sections and keys are emitted sorted: ordering depends on
construction order, which is not part of the contract and would otherwise make
the fixture churn for no reason.

Regenerate after an intentional change, then read the diff::

    python tests/test_settings_defaults.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast
import importlib
import inspect
import pkgutil

from ptychodus.api.settings import SettingsRegistry

GOLDEN_PATH = Path(__file__).parent / 'data' / 'settings_defaults.ini'

# ScratchDirectory defaults under the user's home, so the literal value differs per
# machine. Substituting a token keeps the key and its shape pinned without making
# the fixture unportable.
_HOME_TOKEN = '${HOME}'


# Both trees, because a reconstructor backend's settings live under plugins/
# once it has moved there and under model/ until it does. Walking only one
# would drop a whole backend's groups from the fixture and read as an
# intentional removal rather than as a package that moved.
_SETTINGS_ROOTS = ('ptychodus.model.', 'ptychodus.plugins.')


def _build_registry() -> SettingsRegistry:
    """Instantiate every discoverable ``*Settings`` class against one registry."""
    registry = SettingsRegistry()
    constructed: set[type] = set()

    for prefix in _SETTINGS_ROOTS:
        package = importlib.import_module(prefix.rstrip('.'))
        root = Path(next(iter(package.__path__)))
        _construct_settings_under(root, prefix, registry, constructed)

    return registry


def _construct_settings_under(
    root: Path, prefix: str, registry: SettingsRegistry, constructed: set[type]
) -> None:
    for info in pkgutil.walk_packages([str(root)], prefix):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            # Matches how plugins are loaded: a module whose optional
            # dependency is absent is skipped, not fatal.
            continue

        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and name.endswith('Settings')
                and obj.__module__ == info.name
                and obj not in constructed
            ):
                constructed.add(obj)
                # isclass narrows only to type[object], which takes no arguments; every
                # *Settings class was verified to accept exactly a registry.
                cast(Callable[[SettingsRegistry], object], obj)(registry)


def render_settings() -> str:
    """Render the full settings tree as deterministic INI text."""
    registry = _build_registry()
    home = str(Path.home())
    lines: list[str] = []

    for group_name in sorted(registry):
        lines.append(f'[{group_name}]')
        parameters = registry[group_name].parameters()

        for key in sorted(parameters):
            value = parameters[key].get_value_as_string().replace(home, _HOME_TOKEN)
            lines.append(f'{key} = {value}')

        lines.append('')

    return '\n'.join(lines)


def test_settings_defaults_match_golden() -> None:
    assert GOLDEN_PATH.is_file(), (
        f'Missing {GOLDEN_PATH.name}; regenerate with "python {Path(__file__).name}".'
    )
    assert render_settings() == GOLDEN_PATH.read_text(), (
        'Settings sections, keys or defaults changed. If intended, regenerate with '
        f'"python tests/{Path(__file__).name}" and review the diff -- a renamed key '
        'orphans every saved settings file that still uses the old one.'
    )


def test_golden_covers_the_whole_surface() -> None:
    """A discovery bug that found nothing would make the fixture vacuously true."""
    text = GOLDEN_PATH.read_text()
    assert text.count('\n[') + text.startswith('[') >= 30
    assert sum(1 for line in text.splitlines() if ' = ' in line) >= 500


if __name__ == '__main__':
    GOLDEN_PATH.write_text(render_settings())
    print(f'Wrote {GOLDEN_PATH}')
