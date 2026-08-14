"""Tests for PluginChooserParameter, the display-name view of a PluginChooser.

A PluginChooser carries two name spaces: the human-readable ``display_name``
shown in the GUI and the ``simple_name`` persisted to settings. A combo box
bound directly to the settings parameter therefore mis-restores at startup,
because it would look up a simple name among display-name items.
PluginChooserParameter exists to close that gap, and it is also the sole owner
of the settings binding, so these tests pin the round trip in both directions
and confirm persistence still stores simple names.

No Qt is required — this is pure model-layer behavior.
"""

from __future__ import annotations

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import ParameterGroup, StringParameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter

# Display names deliberately chosen so that simple_name != display_name for
# most of them: re.sub(r'\W+', '', ...) strips the space and the hyphen.
DISPLAY_NAMES = ['Identity', 'Richardson-Lucy', 'Unsupervised Wiener', 'Wiener']


class _Counter(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def _make_settings(default: str) -> StringParameter:
    group = ParameterGroup()
    return group.create_string_parameter('DeconvolutionStrategy', default)


def _build(
    default: str = 'Richardson-Lucy',
) -> tuple[PluginChooserParameter[str], StringParameter]:
    settings = _make_settings(default)
    chooser = PluginChooser[str]()

    for display_name in DISPLAY_NAMES:
        chooser.register_plugin(display_name, display_name=display_name)

    return PluginChooserParameter(chooser, settings), settings


def test_value_is_display_name_while_settings_hold_simple_name() -> None:
    chooser_parameter, settings = _build()

    assert chooser_parameter.get_value() == 'Richardson-Lucy'
    assert settings.get_value() == 'RichardsonLucy'


def test_restores_display_name_from_persisted_simple_name() -> None:
    """The regression this adapter exists to prevent.

    Reading back a simple name from the INI must still surface the display name,
    otherwise a combo box populated with display names silently falls back to
    its first item.
    """
    chooser_parameter, _ = _build(default='UnsupervisedWiener')

    assert chooser_parameter.get_value() == 'Unsupervised Wiener'


def test_set_value_by_display_name_moves_chooser_and_settles() -> None:
    chooser_parameter, settings = _build()
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Unsupervised Wiener')

    assert chooser_parameter.get_value() == 'Unsupervised Wiener'
    assert settings.get_value() == 'UnsupervisedWiener'
    # One notification, not a cascade: the chooser's index guard breaks the loop.
    assert counter.count == 1


def test_selecting_the_current_plugin_is_silent() -> None:
    chooser_parameter, _ = _build()
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Richardson-Lucy')

    assert counter.count == 0


def test_set_value_honors_notify_false() -> None:
    chooser_parameter, _ = _build()
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Wiener', notify=False)

    assert chooser_parameter.get_value() == 'Wiener'
    assert counter.count == 0

    # The suppression must not leak into the next assignment.
    chooser_parameter.set_value('Identity')
    assert counter.count == 1


def test_notify_false_still_persists() -> None:
    """Suppression is a view concern; it must never skip the settings write-back."""
    chooser_parameter, settings = _build()

    chooser_parameter.set_value('Unsupervised Wiener', notify=False)

    assert settings.get_value() == 'UnsupervisedWiener'


def test_external_chooser_change_notifies() -> None:
    """A selection made elsewhere (e.g. batch mode) must reach GUI observers."""
    chooser_parameter, _ = _build()
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Wiener')

    assert chooser_parameter.get_value() == 'Wiener'
    assert counter.count == 1


def test_settings_change_moves_the_selection() -> None:
    """Loading an INI mutates the settings parameter; the selection must follow."""
    chooser_parameter, settings = _build()

    settings.set_value('Wiener')

    assert chooser_parameter.get_value() == 'Wiener'


def test_string_conversion_round_trips_display_names() -> None:
    chooser_parameter, _ = _build()

    chooser_parameter.set_value_from_string('Unsupervised Wiener')

    assert chooser_parameter.get_value_as_string() == 'Unsupervised Wiener'


def test_unknown_name_leaves_selection_unchanged() -> None:
    chooser_parameter, _ = _build()

    chooser_parameter.set_value('Nonexistent Strategy')

    assert chooser_parameter.get_value() == 'Richardson-Lucy'


def test_choices_are_display_names_in_chooser_order() -> None:
    chooser_parameter, _ = _build()

    assert list(chooser_parameter.choices()) == DISPLAY_NAMES


def test_get_strategy_returns_the_selected_plugin_strategy() -> None:
    chooser_parameter, _ = _build()

    chooser_parameter.set_value('Wiener')

    assert chooser_parameter.get_strategy() == 'Wiener'


def test_copy_is_an_unbound_view() -> None:
    """A copy tracks the same chooser but must not write to the original's settings."""
    chooser_parameter, settings = _build()
    copied = chooser_parameter.copy()

    copied.set_value('Wiener')

    assert chooser_parameter.get_value() == 'Wiener'
    # The original adapter is still bound, so it persists the change it observed.
    assert settings.get_value() == 'Wiener'
