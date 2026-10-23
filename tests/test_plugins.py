"""Unit tests for ptychodus.api.plugins.PluginChooser and PluginRegistry.

The chooser is a pure registry plus a tracked selection, and knows nothing about
settings persistence. These tests pin that half: name resolution across both name
spaces, selection and truthiness on an empty chooser, strategy lookup, and the
duplicate-simple-name warning that registration emits. The settings binding layered
over a chooser is covered in test_plugin_chooser_parameter.py.
"""

from __future__ import annotations

from collections import Counter
import logging

import pytest

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
