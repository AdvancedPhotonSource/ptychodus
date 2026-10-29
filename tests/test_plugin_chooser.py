"""Unit tests for ptychodus.model.plugins.PluginChooser.

A chooser is an observable selection over a read-only catalog, and knows nothing
about settings persistence. These tests pin the selection: where it starts, how it
moves by either name space, and when it notifies. Lookup itself belongs to the
catalog and is covered in test_plugins.py.
"""

from __future__ import annotations

import logging

import pytest

from ptychodus.api.plugins import PluginList
from ptychodus.model.observer import Observable, Observer
from ptychodus.model.plugins import PluginChooser


class _Counter(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def _chooser(*display_names: str) -> PluginChooser[str]:
    catalog = PluginList[str]()

    for display_name in display_names:
        catalog.register_plugin(display_name.lower(), display_name=display_name)

    catalog.freeze()
    return PluginChooser(catalog)


def test_get_current_plugin_empty_raises_lookup_error() -> None:
    with pytest.raises(LookupError):
        _chooser().get_current_plugin()


def test_empty_chooser_is_falsy() -> None:
    assert not _chooser()


def test_populated_chooser_is_truthy() -> None:
    assert _chooser('S')


def test_selection_starts_at_the_first_plugin_by_display_name() -> None:
    plugin = _chooser('Zeta', 'Alpha').get_current_plugin()

    assert plugin.display_name == 'Alpha'
    assert plugin.strategy == 'alpha'


def test_set_current_plugin_resolves_either_name_space() -> None:
    chooser = _chooser('Identity', 'Richardson-Lucy')

    chooser.set_current_plugin('RICHARDSONLUCY')
    assert chooser.get_current_plugin().display_name == 'Richardson-Lucy'

    chooser.set_current_plugin('identity')
    assert chooser.get_current_plugin().display_name == 'Identity'


def test_moving_the_selection_notifies_once() -> None:
    chooser = _chooser('Alpha', 'Beta')
    counter = _Counter()
    chooser.add_observer(counter)

    chooser.set_current_plugin('Beta')

    assert counter.count == 1


def test_selecting_the_current_plugin_is_silent() -> None:
    chooser = _chooser('Alpha', 'Beta')
    counter = _Counter()
    chooser.add_observer(counter)

    chooser.set_current_plugin('Alpha')

    assert counter.count == 0


def test_unknown_name_holds_the_selection_but_notifies(caplog: pytest.LogCaptureFixture) -> None:
    """A bound view resynchronizes to the selection the chooser actually holds."""
    chooser = _chooser('Alpha', 'Beta')
    counter = _Counter()
    chooser.add_observer(counter)

    with caplog.at_level(logging.WARNING, logger='ptychodus.model.plugins'):
        chooser.set_current_plugin('Nonexistent')

    assert 'Invalid plugin name' in caplog.text
    assert chooser.get_current_plugin().display_name == 'Alpha'
    assert counter.count == 1


def test_iteration_and_lookup_follow_the_catalog() -> None:
    chooser = _chooser('Beta', 'Alpha')

    assert [plugin.display_name for plugin in chooser] == ['Alpha', 'Beta']
    assert chooser.get_strategy_by_name('BETA') == 'beta'
    assert chooser.find_plugin('Gamma') is None


def test_two_choosers_over_one_catalog_select_independently() -> None:
    """Why ModelCore wraps each registry catalog once and shares the chooser."""
    catalog = PluginList[str]()
    catalog.register_plugin('a', display_name='Alpha')
    catalog.register_plugin('b', display_name='Beta')
    first = PluginChooser(catalog)
    second = PluginChooser(catalog)

    first.set_current_plugin('Beta')

    assert second.get_current_plugin().display_name == 'Alpha'
