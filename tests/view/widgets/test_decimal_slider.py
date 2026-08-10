from __future__ import annotations
from decimal import Decimal

import pytest

from PyQt5.QtCore import Qt

from ptychodus.api.geometry import Interval
from ptychodus.view.widgets import DecimalSlider


@pytest.fixture
def slider(qapp) -> DecimalSlider:
    del qapp  # fixture presence ensures QApplication exists
    return DecimalSlider.create_instance(Qt.Orientation.Horizontal)


def _capture(widget: DecimalSlider) -> list[Decimal]:
    emissions: list[Decimal] = []
    widget.value_changed.connect(emissions.append)
    return emissions


def test_default_state(slider: DecimalSlider) -> None:
    assert slider.get_value() == Decimal('0.5')


def test_set_value_stores_input(slider: DecimalSlider) -> None:
    slider.set_value(Decimal('0.25'))
    assert slider.get_value() == Decimal('0.25')


def test_set_value_clamps_below_minimum(slider: DecimalSlider) -> None:
    slider.set_value(Decimal('-5'))
    assert slider.get_value() == Decimal(0)


def test_set_value_clamps_above_maximum(slider: DecimalSlider) -> None:
    slider.set_value(Decimal('5'))
    assert slider.get_value() == Decimal(1)


def test_set_value_no_op_when_unchanged(slider: DecimalSlider) -> None:
    slider.set_value(Decimal('0.25'))
    emissions = _capture(slider)
    slider.set_value(Decimal('0.25'))
    assert emissions == []


def test_set_value_emits_on_change(slider: DecimalSlider) -> None:
    emissions = _capture(slider)
    slider.set_value(Decimal('0.25'))
    assert emissions == [Decimal('0.25')]


def test_set_value_and_range_rejects_inverted_bounds(slider: DecimalSlider) -> None:
    with pytest.raises(ValueError, match='maximum <= minimum'):
        slider.set_value_and_range(Decimal(0), Interval[Decimal](Decimal(1), Decimal(0)))


def test_set_value_and_range_rejects_degenerate_bounds(slider: DecimalSlider) -> None:
    with pytest.raises(ValueError, match='maximum <= minimum'):
        slider.set_value_and_range(Decimal(0), Interval[Decimal](Decimal(0), Decimal(0)))


def test_set_value_and_range_updates_bounds(slider: DecimalSlider) -> None:
    slider.set_value_and_range(Decimal('5'), Interval[Decimal](Decimal(0), Decimal(10)))
    assert slider.get_value() == Decimal('5')


def test_block_value_changed_signal_suppresses_emission(slider: DecimalSlider) -> None:
    emissions = _capture(slider)
    slider.set_value_and_range(
        Decimal('7'),
        Interval[Decimal](Decimal(0), Decimal(10)),
        block_value_changed_signal=True,
    )
    assert emissions == []
    assert slider.get_value() == Decimal('7')


def test_no_emit_on_bounds_only_change(slider: DecimalSlider) -> None:
    """`set_value_and_range` must not fire `value_changed` when only bounds move."""
    slider.set_value(Decimal('0.5'))
    emissions = _capture(slider)
    slider.set_value_and_range(Decimal('0.5'), Interval[Decimal](Decimal(0), Decimal(2)))
    assert emissions == []


def test_emit_on_value_change_with_bounds_change(slider: DecimalSlider) -> None:
    emissions = _capture(slider)
    slider.set_value_and_range(Decimal('1.5'), Interval[Decimal](Decimal(0), Decimal(2)))
    assert emissions == [Decimal('1.5')]
