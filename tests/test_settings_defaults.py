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

from collections.abc import Callable, Container, Iterable, Mapping
from pathlib import Path
from typing import Any, cast
import importlib
import inspect
import pkgutil
import warnings

from ptychodus.api.plugins import PluginCatalog, PluginRegistry
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.declared import declare_in, describe
from ptychodus.model.fluorescence import ENHANCER_LIBRARY
from ptychodus.model.fluorescence.core import _BRANCHES_BY_ENHANCER
from ptychodus.model.processing import ProcessingParameterStore
from ptychodus.model.processing.core import _BRANCHES_BY_LIBRARY, _SECTIONS_BY_LIBRARY

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
    _register_generator_parameters(registry)
    return registry


# Which group each kind of generator declares into. A generator is a plugin and
# its parameters are created by the factory that offers it, so unlike a
# *Settings class there is nothing here for the walk above to discover. Naming
# the group is what keeps those keys in the fixture; without this they would
# read as deleted the day the generators stopped being hand-written.
_GENERATORS_BY_GROUP: Mapping[str, Callable[[PluginRegistry], PluginCatalog[Any]]] = {
    'ProbePositions': lambda plugins: plugins.probe_position_generators,
    'Probe': lambda plugins: plugins.probe_generators,
    'Object': lambda plugins: plugins.object_generators,
}


def _register_generator_parameters(registry: SettingsRegistry) -> None:
    """Add the parameters every registered generator declares."""
    plugins = PluginRegistry.load_plugins()

    for group_name, choose in _GENERATORS_BY_GROUP.items():
        group = registry[group_name]

        for plugin in choose(plugins):
            declare_in(group, describe(plugin.strategy.get_default_options(), expand_num=True))


def absent_backends() -> list[str]:
    """Reconstructor libraries that registered but offer nothing on this machine.

    A backend that has crossed over has no ``*Settings`` class to discover --
    its groups come from what it declares, and it declares nothing when the
    package it wraps is not installed. So the groups such a backend
    contributes are genuinely absent here, and the fixture cannot pin them.
    """
    absent: list[str] = []

    crossed_over = set(_SECTIONS_BY_LIBRARY) | set(_BRANCHES_BY_LIBRARY)

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        if plugin.display_name in crossed_over and not list(plugin.strategy):
            absent.append(plugin.display_name)

    return absent


def _register_declared_sections(registry: SettingsRegistry) -> None:
    """Add the groups generated from each backend's declaration.

    Both ways a backend reaches the store, since the groups a backend
    contributes are the same contract whichever way it got there.
    """
    store = ProcessingParameterStore(registry)

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        sections = _SECTIONS_BY_LIBRARY.get(plugin.display_name)
        branches = _BRANCHES_BY_LIBRARY.get(plugin.display_name)

        for reconstructor in plugin.strategy:
            if branches is not None:
                store.register_declared(
                    plugin.display_name,
                    reconstructor.name,
                    reconstructor.get_default_options(),
                    branches,
                )
            elif sections is not None:
                store.register(
                    plugin.display_name,
                    reconstructor,
                    sections(reconstructor.name, reconstructor.get_default_options()),
                )

    # Fluorescence enhancers reach the store the same way, under the groups their
    # table names.
    for enhancer in PluginRegistry.load_plugins().fluorescence_enhancers:
        enhancer_branches = _BRANCHES_BY_ENHANCER.get(enhancer.simple_name)

        if enhancer_branches is not None:
            store.register_declared(
                ENHANCER_LIBRARY,
                enhancer.simple_name,
                enhancer.strategy.get_default_options(),
                enhancer_branches,
            )


def absent_enhancers() -> list[str]:
    """Enhancers the table names that did not register on this machine.

    An optional enhancer registers only where the package it wraps is installed, so
    the groups it contributes are genuinely absent elsewhere. Unlike a reconstructor
    backend, the table names its groups whether or not it registered.
    """
    registered = {
        plugin.simple_name for plugin in PluginRegistry.load_plugins().fluorescence_enhancers
    }
    return sorted(set(_BRANCHES_BY_ENHANCER) - registered)


def enhancer_groups(enhancers: Iterable[str]) -> set[str]:
    """The groups the table names for *enhancers*."""
    return {
        branch.group_name for name in enhancers for branch in _BRANCHES_BY_ENHANCER.get(name, ())
    }


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


# The prefix every settings group of a crossed-over backend carries. Named
# here because the backend itself cannot say when its package is absent: its
# section table is answered per reconstructor, and an absent backend offers
# none -- which is exactly the case this fixture needs the answer for.
# ``test_each_backends_group_prefix_covers_what_it_declares`` checks each entry
# against the backend wherever the backend is installed.
_GROUP_PREFIX_BY_BACKEND = {
    'PtychoFM': 'PtychoFM',
    'PtychoPINN': 'PtychoPINN',
    'cdtools': 'CDTools',
    'pty-chi': 'PtyChi',
}


def declared_groups(backends: Iterable[str], text: str) -> set[str]:
    """The groups in *text* belonging to the named backends."""
    prefixes = tuple(
        prefix for name, prefix in _GROUP_PREFIX_BY_BACKEND.items() if name in set(backends)
    )

    if not prefixes:
        return set()

    return {
        line.strip('[]')
        for line in text.split('\n')
        if line.startswith('[') and line.strip('[]').startswith(prefixes)
    }


def generated_groups(backend: str) -> set[str]:
    """Every group *backend* generates, asked of each reconstructor it offers.

    Asked per reconstructor because a section table may answer differently for
    each: pty-chi files every algorithm's additions under a group of that
    algorithm's own. A backend reading its own declaration answers the same
    way for all of them. Empty for a backend whose package is absent.
    """
    groups: set[str] = set()

    for branch in _BRANCHES_BY_LIBRARY.get(backend, ()):
        groups.add(branch.group_name)

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        sections = _SECTIONS_BY_LIBRARY.get(plugin.display_name)

        if sections is None or plugin.display_name != backend:
            continue

        for reconstructor in plugin.strategy:
            for section in sections(reconstructor.name, reconstructor.get_default_options()):
                groups.add(section.group_name)

    return groups


def _without_groups(text: str, group_names: Container[str]) -> str:
    """The same INI text with the named groups dropped whole.

    Split on the separator rather than by lines, so the trailing newline is
    the last element and survives the round trip: a group's block is its
    header, its keys and the blank line after them, and dropping the final
    group has to take that blank line with it.
    """
    lines: list[str] = []
    keep = True

    for line in text.split('\n'):
        if line.startswith('['):
            keep = line.strip('[]') not in group_names

        if keep:
            lines.append(line)

    return '\n'.join(lines)


def test_settings_defaults_match_golden() -> None:
    assert GOLDEN_PATH.is_file(), (
        f'Missing {GOLDEN_PATH.name}; regenerate with "python {Path(__file__).name}".'
    )

    # A backend that has crossed over contributes no groups unless the package
    # it wraps is installed, so on a bare install its sections genuinely do not
    # exist here. Elide exactly those from the golden text and compare
    # everything else: skipping outright would stop pinning every other
    # subsystem too, which is far more surface than the backend costs.
    golden = GOLDEN_PATH.read_text()
    absent = absent_backends() + absent_enhancers()
    unpinned = declared_groups(absent_backends(), golden) | enhancer_groups(absent_enhancers())
    expected = _without_groups(golden, unpinned)

    if absent:
        warnings.warn(
            f'Not installed, so {len(unpinned)} of their settings groups went '
            f'uncompared: {", ".join(sorted(absent))}.',
            stacklevel=1,
        )

    assert render_settings() == expected, (
        'Settings sections, keys or defaults changed. If intended, regenerate with '
        f'"python tests/{Path(__file__).name}" and review the diff -- a renamed key '
        'orphans every saved settings file that still uses the old one.'
    )


def test_every_installed_backend_still_contributes_its_groups() -> None:
    """Complements the comparison above, which elides what it cannot see.

    That one pins values; this pins that a backend which *is* installed did
    not silently stop producing settings -- the failure a crossover would
    cause, and the one an elision would otherwise absorb.
    """
    rendered = render_settings()

    for plugin in PluginRegistry.load_plugins().reconstructor_libraries:
        if plugin.display_name in _SECTIONS_BY_LIBRARY and list(plugin.strategy):
            for group_name in generated_groups(plugin.display_name):
                assert f'[{group_name}]' in rendered, group_name


def test_each_backends_group_prefix_covers_what_it_declares() -> None:
    """The prefix table must agree with the backends that can be asked.

    It exists for the backends that cannot -- an absent one generates no
    groups to read a prefix off -- so it is checked here against every
    backend this machine does have, which is the only place it can be.
    """
    golden = GOLDEN_PATH.read_text()

    for backend in _GROUP_PREFIX_BY_BACKEND:
        generated = generated_groups(backend)

        if generated:
            assert declared_groups([backend], golden) == generated, backend


def test_the_prefix_table_names_every_crossed_over_backend() -> None:
    """A backend missing from the table would be compared against a golden
    that cannot contain it, so a bare install would fail rather than elide.

    Both ways a backend reaches the store: which of the two it uses says how
    its declaration is read, not whether its groups need eliding.
    """
    assert set(_GROUP_PREFIX_BY_BACKEND) == set(_SECTIONS_BY_LIBRARY) | set(_BRANCHES_BY_LIBRARY)


def test_golden_covers_the_whole_surface() -> None:
    """A discovery bug that found nothing would make the fixture vacuously true."""
    text = GOLDEN_PATH.read_text()
    assert text.count('\n[') + text.startswith('[') >= 30
    assert sum(1 for line in text.splitlines() if ' = ' in line) >= 500


if __name__ == '__main__':
    # Refuse rather than quietly drop what this machine cannot build: a
    # regeneration run without every backend installed would delete those
    # groups from the fixture, and the deletion reads as an intended removal.
    missing = absent_backends() + absent_enhancers()

    if missing:
        dropped = declared_groups(absent_backends(), GOLDEN_PATH.read_text()) | enhancer_groups(
            absent_enhancers()
        )
        raise SystemExit(
            f'Refusing to regenerate: {", ".join(sorted(missing))} '
            f'{"is" if len(missing) == 1 else "are"} not installed, so '
            f'{len(dropped)} settings groups would be dropped.'
        )

    GOLDEN_PATH.write_text(render_settings())
    print(f'Wrote {GOLDEN_PATH}')


def test_every_backend_named_in_the_section_table_is_a_registered_library() -> None:
    """The table is keyed on a display name, so a rename would drop its settings.

    Nothing else would notice: a backend whose key no longer matches simply
    registers no groups, which reads exactly like one that has not crossed
    over yet.
    """
    registered = {
        plugin.display_name for plugin in PluginRegistry.load_plugins().reconstructor_libraries
    }
    assert set(_SECTIONS_BY_LIBRARY) <= registered


def test_every_registered_enhancer_names_its_settings_groups() -> None:
    """The table is keyed on a registered name, so a rename would drop its settings.

    An enhancer whose name no longer matches registers no groups and runs on its
    defaults, which reads exactly like one that declares nothing.
    """
    registered = {
        plugin.simple_name for plugin in PluginRegistry.load_plugins().fluorescence_enhancers
    }
    assert registered <= set(_BRANCHES_BY_ENHANCER)
