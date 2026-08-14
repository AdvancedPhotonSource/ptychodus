"""Unit tests for ptychodus.api.plugins.PluginChooser and its settings binding.

The chooser is a pure registry plus a tracked selection; PluginChooserParameter
is the only thing that knows about settings persistence. These tests pin the
selection semantics that the two classes agree on at that seam:

- a name resolves against either name space, but only the canonical simple name
  is ever written back to settings, including when it resolves to index 0;
- an unrecognized name holds the selection *and* the persisted value, but still
  notifies so a bound view resynchronizes;
- registering a plugin re-sorts the list without repointing the selection.
"""

from __future__ import annotations

import pytest

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import ParameterGroup, StringParameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter


class _Counter(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def _make_settings(default: str) -> StringParameter:
    group = ParameterGroup()
    return group.create_string_parameter('FileType', default)


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


def test_binding_normalizes_a_persisted_display_name() -> None:
    settings = _make_settings('Richardson-Lucy')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('i', display_name='Identity')
    chooser.register_plugin('r', display_name='Richardson-Lucy')

    PluginChooserParameter(chooser, settings)

    assert chooser.get_current_plugin().simple_name == 'RichardsonLucy'
    assert settings.get_value() == 'RichardsonLucy'


def test_binding_leaves_a_persisted_simple_name_alone() -> None:
    settings = _make_settings('RichardsonLucy')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('i', display_name='Identity')
    chooser.register_plugin('r', display_name='Richardson-Lucy')

    PluginChooserParameter(chooser, settings)

    assert settings.get_value() == 'RichardsonLucy'


def test_binding_normalizes_even_at_index_zero() -> None:
    """The selection does not move, but the persisted value must still be canonical.

    'Alpha-Plugin' sorts first, so binding to it leaves _current_index at 0. An
    earlier implementation gated the write-back on the index changing, which left
    a display name sitting in the INI forever.
    """
    settings = _make_settings('Alpha-Plugin')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('a', display_name='Alpha-Plugin')
    chooser.register_plugin('z', display_name='Zeta-Plugin')

    PluginChooserParameter(chooser, settings)

    assert chooser.get_current_plugin().simple_name == 'AlphaPlugin'
    assert settings.get_value() == 'AlphaPlugin'


def test_unknown_name_holds_selection_and_setting_but_notifies() -> None:
    """A name can be unresolvable because an optional-dependency plugin is missing.

    Overwriting the setting would discard the user's choice for good, so only the
    view is resynchronized.
    """
    settings = _make_settings('Identity')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('i', display_name='Identity')
    chooser.register_plugin('w', display_name='Wiener')
    chooser_parameter = PluginChooserParameter(chooser, settings)

    settings.set_value('MissingPlugin')
    counter = _Counter()
    chooser_parameter.add_observer(counter)
    chooser.set_current_plugin('MissingPlugin')

    assert chooser.get_current_plugin().display_name == 'Identity'
    assert settings.get_value() == 'MissingPlugin'
    assert counter.count == 1


def test_registration_does_not_repoint_the_selection() -> None:
    """register_plugin re-sorts by display name; the selected plugin must survive it."""
    settings = _make_settings('Zeta')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('m', display_name='Middle')
    chooser.register_plugin('z', display_name='Zeta')
    PluginChooserParameter(chooser, settings)

    chooser.register_plugin('a', display_name='Alpha')

    assert chooser.get_current_plugin().display_name == 'Zeta'
    assert settings.get_value() == 'Zeta'


def test_registration_does_not_clobber_an_unresolved_setting() -> None:
    settings = _make_settings('MissingPlugin')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('m', display_name='Middle')
    PluginChooserParameter(chooser, settings)

    chooser.register_plugin('a', display_name='Alpha')

    assert settings.get_value() == 'MissingPlugin'


def test_late_registration_settles_an_empty_chooser_onto_the_persisted_name() -> None:
    """Binding before plugins are registered must still honor the persisted choice."""
    settings = _make_settings('Zeta')
    chooser: PluginChooser[str] = PluginChooser()
    chooser_parameter = PluginChooserParameter(chooser, settings)

    chooser.register_plugin('a', display_name='Alpha')
    chooser.register_plugin('z', display_name='Zeta')

    assert chooser_parameter.get_value() == 'Zeta'
    assert settings.get_value() == 'Zeta'


def test_two_adapters_over_one_chooser_both_track_it() -> None:
    """The single-binding slot that synchronize_with_parameter had is gone.

    A second binding used to silently replace the first and leave it diverging;
    now each adapter owns its own settings parameter and both stay in step.
    """
    settings_a = _make_settings('Alpha')
    settings_b = _make_settings('Alpha')
    chooser: PluginChooser[str] = PluginChooser()
    chooser.register_plugin('a', display_name='Alpha')
    chooser.register_plugin('z', display_name='Zeta')

    parameter_a = PluginChooserParameter(chooser, settings_a)
    parameter_b = PluginChooserParameter(chooser, settings_b)
    parameter_a.set_value('Zeta')

    assert parameter_b.get_value() == 'Zeta'
    assert settings_a.get_value() == 'Zeta'
    assert settings_b.get_value() == 'Zeta'
