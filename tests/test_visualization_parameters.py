"""Tests for the model/visualization parameters built on PluginChooserParameter.

ScalarTransformationParameter, ColormapParameter, and
CylindricalColorModelParameter were each a hand-written copy of the same
chooser-to-Parameter adapter, and each silently dropped the ``notify`` keyword
that ``Parameter.set_value`` declares. They are now thin subclasses of
PluginChooserParameter, so these tests pin the contract they used to break plus
the accessors callers rely on.

No Qt is required — this is pure model-layer behavior.
"""

from __future__ import annotations

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.visualization import CylindricalColorModel, ScalarTransformation
from ptychodus.model.visualization.color_model import CylindricalColorModelParameter
from ptychodus.model.visualization.colormap import ColormapParameter
from ptychodus.model.visualization.transformation import ScalarTransformationParameter


class _Counter(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def test_transformation_defaults_to_identity() -> None:
    parameter = ScalarTransformationParameter()

    assert parameter.get_value() == 'Identity'
    assert parameter.get_strategy() is ScalarTransformation.IDENTITY


def test_transformation_set_value_honors_notify_false() -> None:
    parameter = ScalarTransformationParameter()
    counter = _Counter()
    parameter.add_observer(counter)

    parameter.set_value('Square Root', notify=False)

    assert parameter.get_value() == 'Square Root'
    assert parameter.get_strategy() is ScalarTransformation.SQRT
    assert counter.count == 0

    # The suppression must not leak into the next assignment.
    parameter.set_value('Natural Logarithm')
    assert counter.count == 1


def test_transformation_resolves_simple_names() -> None:
    """Simple names differ from display names here ('log2' vs 'Logarithm (Base 2)')."""
    parameter = ScalarTransformationParameter()

    parameter.set_value('log2')

    assert parameter.get_value() == 'Logarithm (Base 2)'
    assert parameter.get_strategy() is ScalarTransformation.LOG2


def test_transformation_copy_is_independent() -> None:
    parameter = ScalarTransformationParameter()
    parameter.set_value('Square Root')

    copied = parameter.copy()
    copied.set_value('Identity')

    assert parameter.get_value() == 'Square Root'
    assert copied.get_value() == 'Identity'


def test_colormap_defaults_by_cyclicity() -> None:
    assert ColormapParameter(is_cyclic=False).get_value() == 'gray'
    assert ColormapParameter(is_cyclic=True).get_value() == 'colorwheel'


def test_colormap_choices_are_non_empty_and_contain_the_default() -> None:
    parameter = ColormapParameter(is_cyclic=False)

    choices = list(parameter.choices())

    assert 'gray' in choices
    assert len(choices) > 1


def test_colormap_copy_preserves_cyclicity() -> None:
    parameter = ColormapParameter(is_cyclic=True)

    copied = parameter.copy()

    assert copied.get_value() == 'colorwheel'


def test_color_model_default_resolves_the_simple_name() -> None:
    """The default is given as 'HSV-V', a simple name; the value space is display names."""
    parameter = CylindricalColorModelParameter()

    assert parameter.get_strategy() is CylindricalColorModel.HSV_VALUE
    assert parameter.get_value() == 'HSV Value'


def test_color_model_set_value_honors_notify_false() -> None:
    parameter = CylindricalColorModelParameter()
    counter = _Counter()
    parameter.add_observer(counter)

    parameter.set_value('HLS Lightness', notify=False)

    assert parameter.get_strategy() is CylindricalColorModel.HLS_LIGHTNESS
    assert counter.count == 0

    # The suppression must not leak into the next assignment.
    parameter.set_value('HSV Alpha')
    assert counter.count == 1
