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

import pytest

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing import ProcessingAlgorithmParameter, ProcessingParameterStore
from ptychodus.model.processing.core import _SECTIONS_BY_LIBRARY

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
    """Build every settings group ptychodus would create, however it creates it."""
    registry = SettingsRegistry()
    constructed: set[type] = set()

    for prefix in _SETTINGS_ROOTS:
        package = importlib.import_module(prefix.rstrip('.'))
        root = Path(next(iter(package.__path__)))
        _construct_settings_under(root, prefix, registry, constructed)

    _register_declared_sections(registry)
    return registry


def absent_backends() -> list[str]:
    """Reconstructor libraries that registered but offer nothing on this machine.

    A backend that has crossed over has no ``*Settings`` class to discover --
    its groups come from what it declares, and it declares nothing when the
    package it wraps is not installed. So the groups such a backend
    contributes are genuinely absent here, and the fixture cannot pin them.
    """
    absent: list[str] = []

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        library = plugin.strategy

        if library.name in _SECTIONS_BY_LIBRARY and not list(library):
            absent.append(library.name)

    return absent


def _register_declared_sections(registry: SettingsRegistry) -> None:
    """Add the groups generated from each backend's declaration."""
    store = ProcessingParameterStore(registry)

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        library = plugin.strategy
        sections = _SECTIONS_BY_LIBRARY.get(library.name)

        if sections is None:
            continue

        for reconstructor in library:
            store.register(
                ProcessingAlgorithmParameter.generate_key(library.name, reconstructor.name),
                reconstructor,
                sections(),
            )


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

    absent = absent_backends()

    if absent:
        # A backend that has crossed over contributes no groups unless the
        # package it wraps is installed, so on a bare install those sections
        # genuinely do not exist and the fixture cannot pin them here. Compare
        # what this machine can see, and say what it could not.
        pytest.skip(f'Not installed, so their settings are absent: {", ".join(sorted(absent))}.')

    assert render_settings() == GOLDEN_PATH.read_text(), (
        'Settings sections, keys or defaults changed. If intended, regenerate with '
        f'"python tests/{Path(__file__).name}" and review the diff -- a renamed key '
        'orphans every saved settings file that still uses the old one.'
    )


def test_every_installed_backend_still_contributes_its_groups() -> None:
    """Runs everywhere, including where the golden comparison is skipped.

    The comparison above pins values; this pins that a backend which *is*
    installed did not silently stop producing settings -- the failure a
    crossover would cause, and the one a skip would otherwise hide.
    """
    rendered = render_settings()

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        library = plugin.strategy

        if library.name in _SECTIONS_BY_LIBRARY and list(library):
            for section in _SECTIONS_BY_LIBRARY[library.name]():
                assert f'[{section.group_name}]' in rendered, section.group_name


def test_golden_covers_the_whole_surface() -> None:
    """A discovery bug that found nothing would make the fixture vacuously true."""
    text = GOLDEN_PATH.read_text()
    assert text.count('\n[') + text.startswith('[') >= 30
    assert sum(1 for line in text.splitlines() if ' = ' in line) >= 500


if __name__ == '__main__':
    GOLDEN_PATH.write_text(render_settings())
    print(f'Wrote {GOLDEN_PATH}')


def test_every_backend_named_in_the_section_table_is_a_registered_library() -> None:
    """The table is keyed on a display name, so a rename would drop its settings.

    Nothing else would notice: a backend whose key no longer matches simply
    registers no groups, which reads exactly like one that has not crossed
    over yet.
    """
    registered = {
        plugin.strategy.name for plugin in PluginRegistry.load_plugins().reconstructor_libraries
    }
    assert set(_SECTIONS_BY_LIBRARY) <= registered
