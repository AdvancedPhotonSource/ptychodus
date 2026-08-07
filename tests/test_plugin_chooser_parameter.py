"""Tests for PluginChooserParameter, the display-name view of a PluginChooser.

A PluginChooser carries two name spaces: the human-readable ``display_name``
shown in the GUI and the ``simple_name`` persisted to settings. A combo box
bound directly to the settings parameter therefore mis-restores at startup,
because it would look up a simple name among display-name items.
PluginChooserParameter exists to close that gap, so these tests pin the round
trip in both directions and confirm persistence still stores simple names.

No Qt is required — this is pure model-layer behavior.
"""

from __future__ import annotations

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import ParameterGroup, StringParameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter

# Display names deliberately chosen so that simple_name != display_name for
# most of them: re.sub(r'\W+', '', ...) strips the space and the hyphen.
DISPLAY_NAMES = ['Identity', 'Richardson-Lucy', 'Unsupervised Wiener', 'Wiener']


class _Counter(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def _build(default: str = 'Richardson-Lucy') -> tuple[PluginChooser[str], StringParameter]:
    group = ParameterGroup()
    parameter = group.create_string_parameter('DeconvolutionStrategy', default)
    chooser = PluginChooser[str]()

    for display_name in DISPLAY_NAMES:
        chooser.register_plugin(display_name, display_name=display_name)

    chooser.synchronize_with_parameter(parameter)
    return chooser, parameter


def test_value_is_display_name_while_settings_hold_simple_name() -> None:
    chooser, parameter = _build()
    chooser_parameter = PluginChooserParameter(chooser)

    assert chooser_parameter.get_value() == 'Richardson-Lucy'
    assert parameter.get_value() == 'RichardsonLucy'


def test_restores_display_name_from_persisted_simple_name() -> None:
    """The regression this adapter exists to prevent.

    Reading back a simple name from the INI must still surface the display name,
    otherwise a combo box populated with display names silently falls back to
    its first item.
    """
    chooser, _ = _build(default='UnsupervisedWiener')
    chooser_parameter = PluginChooserParameter(chooser)

    assert chooser_parameter.get_value() == 'Unsupervised Wiener'


def test_set_value_by_display_name_moves_chooser_and_settles() -> None:
    chooser, parameter = _build()
    chooser_parameter = PluginChooserParameter(chooser)
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Unsupervised Wiener')

    assert chooser.get_current_plugin().display_name == 'Unsupervised Wiener'
    assert chooser_parameter.get_value() == 'Unsupervised Wiener'
    assert parameter.get_value() == 'UnsupervisedWiener'
    # One notification, not a cascade: the chooser's index guard breaks the loop.
    assert counter.count == 1


def test_selecting_the_current_plugin_is_silent() -> None:
    chooser, _ = _build()
    chooser_parameter = PluginChooserParameter(chooser)
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Richardson-Lucy')

    assert counter.count == 0


def test_set_value_honors_notify_false() -> None:
    chooser, _ = _build()
    chooser_parameter = PluginChooserParameter(chooser)
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser_parameter.set_value('Wiener', notify=False)

    assert chooser_parameter.get_value() == 'Wiener'
    assert counter.count == 0

    # The suppression must not leak into the next assignment.
    chooser_parameter.set_value('Identity')
    assert counter.count == 1


def test_external_chooser_change_notifies() -> None:
    """A selection made elsewhere (e.g. batch mode) must reach GUI observers."""
    chooser, _ = _build()
    chooser_parameter = PluginChooserParameter(chooser)
    counter = _Counter()
    chooser_parameter.add_observer(counter)

    chooser.set_current_plugin('Wiener')

    assert chooser_parameter.get_value() == 'Wiener'
    assert counter.count == 1


def test_string_conversion_round_trips_display_names() -> None:
    chooser, _ = _build()
    chooser_parameter = PluginChooserParameter(chooser)

    chooser_parameter.set_value_from_string('Unsupervised Wiener')

    assert chooser_parameter.get_value_as_string() == 'Unsupervised Wiener'


def test_unknown_name_leaves_selection_unchanged() -> None:
    chooser, _ = _build()
    chooser_parameter = PluginChooserParameter(chooser)

    chooser_parameter.set_value('Nonexistent Strategy')

    assert chooser_parameter.get_value() == 'Richardson-Lucy'
