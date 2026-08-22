from __future__ import annotations
from decimal import Decimal

import pytest

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest

from ptychodus.api.geometry import Interval
from ptychodus.view.image import DecimalRangeSlider, Handle


def _iv(lower, upper) -> Interval[Decimal]:
    return Interval[Decimal](Decimal(str(lower)), Decimal(str(upper)))


@pytest.fixture
def slider(qapp) -> DecimalRangeSlider:
    del qapp
    widget = DecimalRangeSlider.create_instance(Qt.Orientation.Horizontal)
    widget.resize(400, 40)
    widget.show()
    return widget


def _capture(widget: DecimalRangeSlider) -> list[Interval]:
    emissions: list[Interval] = []
    widget.selection_changed.connect(emissions.append)
    return emissions


def _endpoints(interval: Interval) -> tuple[Decimal, Decimal]:
    return interval.lower, interval.upper


def test_default_state(slider: DecimalRangeSlider) -> None:
    assert _endpoints(slider.get_bounds()) == (Decimal(0), Decimal(1))
    assert _endpoints(slider.get_selection()) == (Decimal(0), Decimal(1))


def test_default_construction_emits_no_signal(qapp) -> None:
    del qapp
    emissions: list[Interval] = []
    widget = DecimalRangeSlider.create_instance(Qt.Orientation.Horizontal)
    widget.selection_changed.connect(emissions.append)
    assert emissions == []


def test_only_horizontal_supported(qapp) -> None:
    del qapp
    with pytest.raises(NotImplementedError):
        DecimalRangeSlider.create_instance(Qt.Orientation.Vertical)


def test_set_selection_clamps_into_bounds(slider: DecimalRangeSlider) -> None:
    emissions = _capture(slider)
    slider.set_selection(_iv(-5, 5))
    assert _endpoints(slider.get_selection()) == (Decimal(0), Decimal(1))
    assert len(emissions) == 0  # selection was already (0, 1); clamped result matches


def test_set_selection_clamped_change_emits_once(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    emissions = _capture(slider)
    slider.set_selection(_iv(-5, 5))
    assert len(emissions) == 1
    assert _endpoints(emissions[0]) == (Decimal(0), Decimal(1))


def test_set_selection_rejects_inverted_interval(slider: DecimalRangeSlider) -> None:
    with pytest.raises(ValueError, match='upper < lower'):
        slider.set_selection(_iv('0.7', '0.3'))


def test_set_selection_no_op_when_unchanged(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    emissions = _capture(slider)
    slider.set_selection(_iv('0.3', '0.7'))
    assert emissions == []


def test_set_selection_and_bounds_widens_bounds_and_updates_selection(
    slider: DecimalRangeSlider,
) -> None:
    slider.set_selection_and_bounds(_iv(-5, 5), _iv(-10, 10))
    assert _endpoints(slider.get_bounds()) == (Decimal(-10), Decimal(10))
    assert _endpoints(slider.get_selection()) == (Decimal(-5), Decimal(5))


def test_set_selection_and_bounds_rejects_inverted_bounds(slider: DecimalRangeSlider) -> None:
    with pytest.raises(ValueError, match='maximum <= minimum'):
        slider.set_selection_and_bounds(_iv(0, 1), _iv(1, 0))


def test_set_selection_and_bounds_rejects_degenerate_bounds(slider: DecimalRangeSlider) -> None:
    with pytest.raises(ValueError, match='maximum <= minimum'):
        slider.set_selection_and_bounds(_iv(0, 0), _iv(0, 0))


def test_block_signal_suppresses_emission(slider: DecimalRangeSlider) -> None:
    emissions = _capture(slider)
    slider.set_selection_and_bounds(_iv(-5, 5), _iv(-10, 10), block_signal=True)
    assert emissions == []
    assert _endpoints(slider.get_selection()) == (Decimal(-5), Decimal(5))


def test_no_emit_on_bounds_only_change(slider: DecimalRangeSlider) -> None:
    """Widen bounds while keeping the selection: no signal."""
    slider.set_selection(_iv('0.25', '0.75'))
    emissions = _capture(slider)
    slider.set_selection_and_bounds(_iv('0.25', '0.75'), _iv(-2, 2))
    assert emissions == []


def test_signal_payload_is_interval_with_ordered_endpoints(slider: DecimalRangeSlider) -> None:
    emissions = _capture(slider)
    slider.set_selection(_iv('0.3', '0.7'))
    assert len(emissions) == 1
    payload = emissions[0]
    assert isinstance(payload, Interval)
    assert payload.lower <= payload.upper


def _tick(bounds: Interval[Decimal], num_ticks: int = 1000) -> Decimal:
    return (bounds.upper - bounds.lower) / Decimal(num_ticks)


def test_keyboard_arrow_moves_focused_handle_one_tick(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    emissions = _capture(slider)
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    step = _tick(slider.get_bounds())
    assert slider.get_selection().lower == Decimal('0.3') + step
    assert len(emissions) == 1


def test_keyboard_pageup_moves_ten_ticks(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_PageUp)
    step = _tick(slider.get_bounds())
    assert slider.get_selection().lower == Decimal('0.3') + 10 * step


def test_keyboard_home_focused_lower_goes_to_bounds_lower(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert slider.get_selection().lower == Decimal(0)
    assert slider.get_selection().upper == Decimal('0.7')


def test_keyboard_end_focused_lower_stops_at_upper(slider: DecimalRangeSlider) -> None:
    """End on the lower handle must not push past the upper handle."""
    slider.set_selection(_iv('0.3', '0.7'))
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert slider.get_selection().lower == Decimal('0.7')
    assert slider.get_selection().upper == Decimal('0.7')


def test_keyboard_handle_switch_via_bracket_keys(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.3', '0.7'))
    emissions = _capture(slider)
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_BracketRight)
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert slider.get_selection().upper == Decimal(1)
    QTest.keyClick(slider, Qt.Key.Key_BracketLeft)
    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert slider.get_selection().lower == Decimal(0)
    # bracket keys alone do not emit — only the End/Home keys after do
    assert len(emissions) == 2


def test_arrow_on_lower_stops_at_upper(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.5', '0.5'))
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    assert slider.get_selection().lower == Decimal('0.5')
    assert slider.get_selection().upper == Decimal('0.5')


def _paint_area(widget: DecimalRangeSlider):
    return widget._paint_area  # noqa: SLF001 - test accesses internal paint surface


def test_mouse_press_selects_lower_handle(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.25', '0.75'))
    pa = _paint_area(slider)
    x = pa._handle_x(Handle.LOWER)  # noqa: SLF001
    y = pa._groove_y()  # noqa: SLF001
    QTest.mousePress(pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
    assert slider._active_handle is Handle.LOWER  # noqa: SLF001
    QTest.mouseRelease(pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
    assert slider._active_handle is None  # noqa: SLF001


def test_mouse_press_selects_upper_handle(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.25', '0.75'))
    pa = _paint_area(slider)
    x = pa._handle_x(Handle.UPPER)  # noqa: SLF001
    y = pa._groove_y()  # noqa: SLF001
    QTest.mousePress(pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
    assert slider._active_handle is Handle.UPPER  # noqa: SLF001


def test_mouse_press_stacked_handles_tiebreak_by_side(slider: DecimalRangeSlider) -> None:
    slider.set_selection(_iv('0.5', '0.5'))
    pa = _paint_area(slider)
    x = pa._handle_x(Handle.LOWER)  # noqa: SLF001
    y = pa._groove_y()  # noqa: SLF001
    # click just to the left of the stacked pair -> lower
    QTest.mousePress(
        pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x - 2, y)
    )
    assert slider._active_handle is Handle.LOWER  # noqa: SLF001
    QTest.mouseRelease(
        pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x - 2, y)
    )
    # click just to the right of the stacked pair -> upper
    QTest.mousePress(
        pa, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x + 2, y)
    )
    assert slider._active_handle is Handle.UPPER  # noqa: SLF001
