"""Unit tests for ptychodus.api.plugins.PluginChooser."""

from __future__ import annotations

import pytest

from ptychodus.api.plugins import PluginChooser


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
