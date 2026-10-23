"""Unit tests for ptychodus.api.plugins.PluginChooser and PluginRegistry.

The chooser is a pure registry plus a tracked selection, and knows nothing about
settings persistence. These tests pin that half: name resolution across both name
spaces, selection and truthiness on an empty chooser, strategy lookup, and the
duplicate-simple-name warning that registration emits. The settings binding layered
over a chooser is covered in test_plugin_chooser_parameter.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging

import pytest

from ptychodus.api.files import FileHandler
from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginChooser, PluginRegistry


def test_get_current_plugin_empty_raises_lookup_error() -> None:
    chooser: PluginChooser[str] = PluginChooser()

    with pytest.raises(LookupError):
        chooser.get_current_plugin()


def test_get_current_plugin_returns_registered() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('strategy-a', display_name='Strategy A')

    plugin = chooser.get_current_plugin()

    assert plugin.strategy == 'strategy-a'
    assert plugin.display_name == 'Strategy A'


def test_empty_chooser_is_falsy() -> None:
    chooser: PluginChooser[str] = PluginChooser()

    assert not chooser


def test_populated_chooser_is_truthy() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('s', display_name='S')

    assert chooser


def test_find_plugin_matches_either_name_space_case_insensitively() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('s', display_name='Richardson-Lucy')

    by_display = chooser.find_plugin('richardson-lucy')
    by_simple = chooser.find_plugin('RICHARDSONLUCY')

    assert by_display is not None
    assert by_simple is by_display
    assert by_display.simple_name == 'RichardsonLucy'


def test_find_plugin_returns_none_for_unknown_name() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('s', display_name='Richardson-Lucy')

    assert chooser.find_plugin('Nonexistent') is None


def test_get_strategy_by_name_resolves_either_name_space_case_insensitively() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('rl-strategy', display_name='Richardson-Lucy')

    assert chooser.get_strategy_by_name('richardson-lucy') == 'rl-strategy'
    assert chooser.get_strategy_by_name('RICHARDSONLUCY') == 'rl-strategy'


def test_get_strategy_by_name_raises_lookup_error_with_registered_names_in_message() -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('a', display_name='Alpha')
    chooser.register_plugin('b', display_name='Beta')

    with pytest.raises(LookupError) as info:
        chooser.get_strategy_by_name('missing')

    message = str(info.value)
    assert 'missing' in message
    assert 'Alpha' in message
    assert 'Beta' in message


def test_get_strategy_by_name_empty_chooser_raises_lookup_error() -> None:
    chooser: PluginChooser[str] = PluginChooser()

    with pytest.raises(LookupError):
        chooser.get_strategy_by_name('anything')


def test_register_duplicate_simple_name_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A shadowed registration is announced rather than failing silently.

    Settings persist the simple name and lookup returns the first match, so the
    second registration under a taken name can never be selected from settings.
    Registration still succeeds: plugin loading must never be fatal.
    """
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('h5', display_name='Example Files (*.h5)', simple_name='example')

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        chooser.register_plugin('mat', display_name='Example Files (*.mat)', simple_name='example')

    assert 'Example Files (*.h5)' in caplog.text
    assert 'Example Files (*.mat)' in caplog.text

    assert chooser.get_strategy_by_name('example') == 'h5'


def test_register_distinct_simple_names_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('h5', display_name='Example Files (*.h5)', simple_name='example')

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        chooser.register_plugin(
            'mat', display_name='Example Files (*.mat)', simple_name='example_mat'
        )

    assert 'Duplicate plugin simple name' not in caplog.text


def test_load_plugins_has_no_duplicate_simple_names() -> None:
    """Every chooser resolves each simple name to exactly one plugin.

    A duplicate makes the later registration unreachable from settings, which is
    how the two fold_slice probe-position readers once shadowed each other.
    """
    registry = PluginRegistry.load_plugins()
    duplicates: list[str] = []

    for attribute in dir(registry):
        if attribute.startswith('_'):
            continue

        chooser = getattr(registry, attribute)

        if not isinstance(chooser, PluginChooser):
            continue

        counter = Counter(plugin.simple_name.casefold() for plugin in chooser)
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
    chooser: PluginChooser[ObjectFileReader] = PluginChooser()
    chooser.register_plugin(_ExampleObjectFileReader(), simple_name='example')

    assert chooser.get_current_plugin().display_name == 'Example Files (*.h5 *.hdf5)'


def test_register_plugin_skips_a_non_file_handler_with_no_display_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing to register under, but plugin loading must never be fatal."""
    chooser: PluginChooser[str] = PluginChooser()

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        chooser.register_plugin('strategy', simple_name='strategy')

    assert 'not a FileHandler' in caplog.text
    assert not chooser


def test_register_plugin_ignores_a_display_name_override_for_a_file_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file handler is named by its own filter; an override would reopen the drift."""
    chooser: PluginChooser[ObjectFileReader] = PluginChooser()

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        chooser.register_plugin(
            _ExampleObjectFileReader(), display_name='Something Else', simple_name='example'
        )

    assert 'Ignoring display name' in caplog.text
    assert chooser.get_current_plugin().display_name == 'Example Files (*.h5 *.hdf5)'


def test_every_registered_file_handler_describes_itself() -> None:
    """A file chooser holds only FileHandlers, and each one owns the name it shows.

    The registered display name is the Qt filter and the key a settings file resolves
    against, so a handler whose description and extensions drift from it would move a
    dialog entry and break that round trip.
    """
    registry = PluginRegistry.load_plugins()
    problems: list[str] = []

    for attribute in dir(registry):
        if attribute.startswith('_') or not attribute.endswith(('_readers', '_writers')):
            continue

        chooser = getattr(registry, attribute)

        if not isinstance(chooser, PluginChooser):
            continue

        for plugin in chooser:
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
