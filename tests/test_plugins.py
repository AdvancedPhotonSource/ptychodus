"""Unit tests for ptychodus.api.plugins: the plugin catalogs and PluginRegistry.

A catalog is a sorted, read-only collection once registration is over, and holds no
selection. These tests pin name resolution across both name spaces, strategy lookup,
the warnings registration emits, and the freeze that ends registration. Selection over
a catalog is covered in test_plugin_chooser.py, and the settings binding layered over
that in test_plugin_chooser_parameter.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging

import pytest

from ptychodus.api.files import FileHandler
from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginList, PluginRegistry


def test_empty_catalog_is_falsy() -> None:
    catalog: PluginList[str] = PluginList()

    assert not catalog
    assert len(catalog) == 0


def test_catalog_is_sorted_by_display_name() -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('z', display_name='Zeta')
    catalog.register_plugin('a', display_name='Alpha')
    catalog.register_plugin('m', display_name='Middle')

    assert [plugin.display_name for plugin in catalog] == ['Alpha', 'Middle', 'Zeta']
    assert catalog[0].strategy == 'a'


def test_find_plugin_matches_either_name_space_case_insensitively() -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('s', display_name='Richardson-Lucy')

    by_display = catalog.find_plugin('richardson-lucy')
    by_simple = catalog.find_plugin('RICHARDSONLUCY')

    assert by_display is not None
    assert by_simple is by_display
    assert by_display.simple_name == 'RichardsonLucy'


def test_find_plugin_returns_none_for_unknown_name() -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('s', display_name='Richardson-Lucy')

    assert catalog.find_plugin('Nonexistent') is None


def test_get_strategy_by_name_resolves_either_name_space_case_insensitively() -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('rl-strategy', display_name='Richardson-Lucy')

    assert catalog.get_strategy_by_name('richardson-lucy') == 'rl-strategy'
    assert catalog.get_strategy_by_name('RICHARDSONLUCY') == 'rl-strategy'


def test_get_strategy_by_name_raises_lookup_error_with_registered_names_in_message() -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('a', display_name='Alpha')
    catalog.register_plugin('b', display_name='Beta')

    with pytest.raises(LookupError) as info:
        catalog.get_strategy_by_name('missing')

    message = str(info.value)
    assert 'missing' in message
    assert 'Alpha' in message
    assert 'Beta' in message


def test_get_strategy_by_name_empty_catalog_raises_lookup_error() -> None:
    catalog: PluginList[str] = PluginList()

    with pytest.raises(LookupError):
        catalog.get_strategy_by_name('anything')


def test_register_duplicate_simple_name_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A shadowed registration is announced rather than failing silently.

    Settings persist the simple name and lookup returns the first match, so the
    second registration under a taken name can never be selected from settings.
    Registration still succeeds: plugin loading must never be fatal.
    """
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('h5', display_name='Example Files (*.h5)', simple_name='example')

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        catalog.register_plugin('mat', display_name='Example Files (*.mat)', simple_name='example')

    assert 'Example Files (*.h5)' in caplog.text
    assert 'Example Files (*.mat)' in caplog.text

    assert catalog.get_strategy_by_name('example') == 'h5'


def test_register_distinct_simple_names_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('h5', display_name='Example Files (*.h5)', simple_name='example')

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        catalog.register_plugin(
            'mat', display_name='Example Files (*.mat)', simple_name='example_mat'
        )

    assert 'Duplicate plugin simple name' not in caplog.text


def test_a_frozen_catalog_refuses_registration(caplog: pytest.LogCaptureFixture) -> None:
    """Read-only after registration, but a late registration warns rather than raises."""
    catalog: PluginList[str] = PluginList()
    catalog.register_plugin('a', display_name='Alpha')
    catalog.freeze()

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        catalog.register_plugin('b', display_name='Beta')

    assert 'frozen' in caplog.text
    assert [plugin.display_name for plugin in catalog] == ['Alpha']


def test_load_plugins_freezes_every_catalog() -> None:
    registry = PluginRegistry.load_plugins()
    open_catalogs: list[str] = []

    for attribute, catalog in vars(registry).items():
        if isinstance(catalog, PluginList):
            before = len(catalog)
            catalog.register_plugin(object(), display_name='Late Registration')

            if len(catalog) != before:
                open_catalogs.append(attribute)

    assert not open_catalogs


def test_a_directly_built_registry_accepts_registration() -> None:
    """Only load_plugins freezes, so a test can still assemble a registry by hand."""
    registry = PluginRegistry()
    registry.kb_mirrors.register_plugin(object(), display_name='Mirrors')  # type: ignore[arg-type]

    assert len(registry.kb_mirrors) == 1


def test_load_plugins_has_no_duplicate_simple_names() -> None:
    """Every catalog resolves each simple name to exactly one plugin.

    A duplicate makes the later registration unreachable from settings, which is
    how the two fold_slice probe-position readers once shadowed each other.
    """
    registry = PluginRegistry.load_plugins()
    duplicates: list[str] = []

    for attribute in dir(registry):
        if attribute.startswith('_'):
            continue

        catalog = getattr(registry, attribute)

        if not isinstance(catalog, PluginList):
            continue

        counter = Counter(plugin.simple_name.casefold() for plugin in catalog)
        duplicates.extend(f'{attribute}: {name}' for name, count in counter.items() if count > 1)

    assert not duplicates


class _ExampleObjectFileReader(ObjectFileReader):
    def get_file_description(self) -> str:
        return 'Example Files'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> Object:
        raise NotImplementedError


def test_register_plugin_derives_display_name_from_a_file_handler() -> None:
    """A file handler names itself, so the registration need not repeat the filter."""
    catalog: PluginList[ObjectFileReader] = PluginList()
    catalog.register_plugin(_ExampleObjectFileReader(), simple_name='example')

    assert catalog[0].display_name == 'Example Files (*.h5 *.hdf5)'


def test_register_plugin_skips_a_non_file_handler_with_no_display_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing to register under, but plugin loading must never be fatal."""
    catalog: PluginList[str] = PluginList()

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        catalog.register_plugin('strategy', simple_name='strategy')

    assert 'not a FileHandler' in caplog.text
    assert not catalog


def test_register_plugin_ignores_a_display_name_override_for_a_file_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file handler is named by its own filter; an override would reopen the drift."""
    catalog: PluginList[ObjectFileReader] = PluginList()

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        catalog.register_plugin(
            _ExampleObjectFileReader(), display_name='Something Else', simple_name='example'
        )

    assert 'Ignoring display name' in caplog.text
    assert catalog[0].display_name == 'Example Files (*.h5 *.hdf5)'


def test_every_registered_file_handler_describes_itself() -> None:
    """A file catalog holds only FileHandlers, and each one owns the name it shows.

    The registered display name is the Qt filter and the key a settings file resolves
    against, so a handler whose description and extensions drift from it would move a
    dialog entry and break that round trip.
    """
    registry = PluginRegistry.load_plugins()
    problems: list[str] = []

    for attribute in dir(registry):
        if attribute.startswith('_') or not attribute.endswith(('_readers', '_writers')):
            continue

        catalog = getattr(registry, attribute)

        if not isinstance(catalog, PluginList):
            continue

        for plugin in catalog:
            strategy = plugin.strategy
            where = f'{attribute}: {plugin.simple_name}'

            if not isinstance(strategy, FileHandler):
                problems.append(f'{where} is a {type(strategy).__name__}, not a FileHandler')
                continue

            extensions = strategy.get_file_extensions()

            if not extensions:
                problems.append(f'{where} declares no extensions')

            problems.extend(
                f'{where} extension "{extension}" has no leading dot'
                for extension in sorted(extensions)
                if not extension.startswith('.')
            )

            if plugin.display_name != strategy.get_file_filter():
                problems.append(
                    f'{where} shows "{plugin.display_name}" '
                    f'but builds "{strategy.get_file_filter()}"'
                )

    assert not problems
