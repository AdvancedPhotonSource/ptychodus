from __future__ import annotations
from decimal import Decimal
from enum import Enum

import numpy

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from ptychodus.api.geometry import Interval


class Handle(Enum):
    LOWER = 'lower'
    UPPER = 'upper'


class _RangeSliderPaintArea(QWidget):
    """Interactive paint surface for `DecimalRangeSlider`.

    Owns paint and mouse handling; all state and the keyboard/signal surface
    live on the outer `DecimalRangeSlider`.
    """

    _HANDLE_RADIUS = 7
    _GROOVE_HEIGHT = 4
    _TICK_HEIGHT = 5
    _TICK_GAP = 2
    _HIT_TOLERANCE = 2

    def __init__(self, owner: DecimalRangeSlider) -> None:
        super().__init__(owner)
        self._owner = owner
        height = 2 * self._HANDLE_RADIUS + self._TICK_GAP + self._TICK_HEIGHT + 4
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _groove_x_range(self) -> tuple[int, int]:
        pad = self._HANDLE_RADIUS + 1
        return pad, self.width() - pad - 1

    def _groove_y(self) -> int:
        return self._HANDLE_RADIUS + 1

    def _handle_x(self, handle: Handle) -> int:
        selection = self._owner._selection
        value = selection.lower if handle is Handle.LOWER else selection.upper
        return self._value_to_x(value)

    def _value_to_x(self, value: Decimal) -> int:
        bounds = self._owner._bounds
        left, right = self._groove_x_range()
        span = bounds.upper - bounds.lower
        if span == 0:
            return left
        alpha = (value - bounds.lower) / span
        alpha_f = max(0.0, min(1.0, float(alpha)))
        return int(numpy.rint(left + alpha_f * (right - left)))

    def _x_to_value(self, x: int) -> Decimal:
        bounds = self._owner._bounds
        left, right = self._groove_x_range()
        span_px = right - left
        if span_px <= 0:
            return bounds.lower
        alpha_f = max(0.0, min(1.0, (x - left) / span_px))
        num_ticks = self._owner._num_ticks
        tick = int(numpy.rint(alpha_f * num_ticks))
        alpha = Decimal(tick) / Decimal(num_ticks)
        return bounds.lower + alpha * (bounds.upper - bounds.lower)

    def _hit_test(self, pos: QPoint) -> Handle | None:
        lower_x = self._handle_x(Handle.LOWER)
        upper_x = self._handle_x(Handle.UPPER)
        y_center = self._groove_y()
        reach = self._HANDLE_RADIUS + self._HIT_TOLERANCE

        def within(hx: int) -> bool:
            return (pos.x() - hx) ** 2 + (pos.y() - y_center) ** 2 <= reach * reach

        lower_hit = within(lower_x)
        upper_hit = within(upper_x)

        if lower_hit and upper_hit:
            return Handle.LOWER if pos.x() <= lower_x else Handle.UPPER
        if lower_hit:
            return Handle.LOWER
        if upper_hit:
            return Handle.UPPER
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        handle = self._hit_test(event.pos())
        if handle is None:
            super().mousePressEvent(event)
            return
        self._owner._active_handle = handle
        self._owner._focused_handle = handle
        self._owner.setFocus(Qt.FocusReason.MouseFocusReason)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._owner._active_handle is None:
            super().mouseMoveEvent(event)
            return
        value = self._x_to_value(event.pos().x())
        self._owner._drive_handle(self._owner._active_handle, value)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._owner._active_handle is not None:
            self._owner._active_handle = None
            self.update()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()

        left, right = self._groove_x_range()
        y = self._groove_y()
        groove_rect = QRect(left, y - self._GROOVE_HEIGHT // 2, right - left, self._GROOVE_HEIGHT)
        painter.setPen(QPen(palette.dark().color(), 1))
        painter.setBrush(QBrush(palette.mid().color()))
        painter.drawRoundedRect(groove_rect, 2, 2)

        lower_x = self._handle_x(Handle.LOWER)
        upper_x = self._handle_x(Handle.UPPER)
        if upper_x > lower_x:
            fill_rect = QRect(
                lower_x, y - self._GROOVE_HEIGHT // 2, upper_x - lower_x, self._GROOVE_HEIGHT
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(palette.highlight().color()))
            painter.drawRect(fill_rect)

        tick_top = y + self._GROOVE_HEIGHT // 2 + self._TICK_GAP
        tick_bottom = tick_top + self._TICK_HEIGHT
        painter.setPen(QPen(palette.dark().color(), 1))
        for i in range(11):
            tx = int(numpy.rint(left + (i / 10.0) * (right - left)))
            painter.drawLine(tx, tick_top, tx, tick_bottom)

        has_focus = self._owner.hasFocus()
        for handle, hx in ((Handle.LOWER, lower_x), (Handle.UPPER, upper_x)):
            focused = has_focus and handle is self._owner._focused_handle
            painter.setPen(
                QPen(palette.highlight().color(), 2) if focused else QPen(palette.dark().color(), 1)
            )
            painter.setBrush(QBrush(palette.button().color()))
            painter.drawEllipse(QPoint(hx, y), self._HANDLE_RADIUS, self._HANDLE_RADIUS)

        painter.end()


class DecimalRangeSlider(QWidget):
    """Two-handle range slider over `Interval[Decimal]`.

    Public surface parallels the behaviors of :class:`DecimalSlider`
    (clamping, edge-triggered signal, `ValueError` on inverted bounds) but
    uses range-slider names: `get_selection` / `set_selection` /
    `set_selection_and_bounds` and the `selection_changed` signal.
    """

    selection_changed = pyqtSignal(Interval)

    def __init__(
        self,
        parent: QWidget | None,
        *,
        num_ticks: int,
    ) -> None:
        super().__init__(parent)
        self._num_ticks = num_ticks
        self._bounds = Interval[Decimal](Decimal(0), Decimal(1))
        self._selection = Interval[Decimal](Decimal(0), Decimal(1))
        self._active_handle: Handle | None = None
        self._focused_handle: Handle = Handle.LOWER

        self._min_label = QLabel()
        self._max_label = QLabel()
        self._paint_area = _RangeSliderPaintArea(self)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._min_label)
        layout.addWidget(self._paint_area, stretch=1)
        layout.addWidget(self._max_label)
        self.setLayout(layout)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_labels()

    @classmethod
    def create_instance(
        cls,
        orientation: Qt.Orientation,
        parent: QWidget | None = None,
        *,
        num_ticks: int = 1000,
    ) -> DecimalRangeSlider:
        if orientation != Qt.Orientation.Horizontal:
            raise NotImplementedError('DecimalRangeSlider only supports horizontal orientation.')
        return cls(parent, num_ticks=num_ticks)

    def get_selection(self) -> Interval[Decimal]:
        return Interval[Decimal](self._selection.lower, self._selection.upper)

    def get_bounds(self) -> Interval[Decimal]:
        return Interval[Decimal](self._bounds.lower, self._bounds.upper)

    def set_selection(self, selection: Interval[Decimal]) -> None:
        if selection.upper < selection.lower:
            raise ValueError(f'upper < lower ({selection.upper} < {selection.lower})')
        new_lower = self._bounds.clamp(selection.lower)
        new_upper = self._bounds.clamp(selection.upper)
        if self._apply_selection(new_lower, new_upper):
            self._emit_selection_changed()

    def set_selection_and_bounds(
        self,
        selection: Interval[Decimal],
        bounds: Interval[Decimal],
        block_signal: bool = False,
    ) -> None:
        if bounds.upper <= bounds.lower:
            raise ValueError(f'maximum <= minimum ({bounds.upper} <= {bounds.lower})')
        if selection.upper < selection.lower:
            raise ValueError(f'upper < lower ({selection.upper} < {selection.lower})')

        self._bounds = Interval[Decimal](bounds.lower, bounds.upper)
        new_lower = self._bounds.clamp(selection.lower)
        new_upper = self._bounds.clamp(selection.upper)
        selection_changed = self._apply_selection(new_lower, new_upper)

        self._paint_area.update()

        if selection_changed and not block_signal:
            self._emit_selection_changed()

    def _apply_selection(self, new_lower: Decimal, new_upper: Decimal) -> bool:
        if new_upper < new_lower:
            new_upper = new_lower
        changed = new_lower != self._selection.lower or new_upper != self._selection.upper
        if changed:
            self._selection = Interval[Decimal](new_lower, new_upper)
            self._update_labels()
            self._paint_area.update()
        return changed

    def _drive_handle(self, handle: Handle, candidate: Decimal) -> None:
        candidate = self._bounds.clamp(candidate)
        if handle is Handle.LOWER:
            new_lower = min(candidate, self._selection.upper)
            new_upper = self._selection.upper
        else:
            new_lower = self._selection.lower
            new_upper = max(candidate, self._selection.lower)
        if self._apply_selection(new_lower, new_upper):
            self._emit_selection_changed()

    def _tick_step(self) -> Decimal:
        span = self._bounds.upper - self._bounds.lower
        if self._num_ticks <= 0 or span == 0:
            return Decimal(0)
        return span / Decimal(self._num_ticks)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()

        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            new_focus = Handle.LOWER if key == Qt.Key.Key_BracketLeft else Handle.UPPER
            if new_focus is not self._focused_handle:
                self._focused_handle = new_focus
                self._paint_area.update()
            event.accept()
            return

        step = self._tick_step()
        current = (
            self._selection.lower if self._focused_handle is Handle.LOWER else self._selection.upper
        )

        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            candidate = current - step
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            candidate = current + step
        elif key == Qt.Key.Key_PageDown:
            candidate = current - 10 * step
        elif key == Qt.Key.Key_PageUp:
            candidate = current + 10 * step
        elif key == Qt.Key.Key_Home:
            candidate = (
                self._bounds.lower
                if self._focused_handle is Handle.LOWER
                else self._selection.lower
            )
        elif key == Qt.Key.Key_End:
            candidate = (
                self._selection.upper
                if self._focused_handle is Handle.LOWER
                else self._bounds.upper
            )
        else:
            super().keyPressEvent(event)
            return

        self._drive_handle(self._focused_handle, candidate)
        event.accept()

    def focusInEvent(self, event) -> None:  # noqa: N802, ANN001
        super().focusInEvent(event)
        self._paint_area.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802, ANN001
        super().focusOutEvent(event)
        self._paint_area.update()

    def _update_labels(self) -> None:
        self._min_label.setText(f'{self._selection.lower:.3f}')
        self._max_label.setText(f'{self._selection.upper:.3f}')

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.get_selection())
