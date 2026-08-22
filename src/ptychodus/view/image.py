from __future__ import annotations
from collections.abc import Iterator
from decimal import Decimal
from enum import Enum

import numpy

from PyQt5.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QIcon,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.geometry import Interval
from ptychodus.api.typing import RealArrayType

from .visualization import VisualizationView
from .widgets import DecimalLineEdit


class BottomTitledGroupBox(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: bottom center;
            }""")


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


class ImageDisplayRangeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.button_box = QDialogButtonBox()
        self.min_value_line_edit = DecimalLineEdit.create_instance()
        self.max_value_line_edit = DecimalLineEdit.create_instance()

        self.setWindowTitle('Set Display Range')
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        layout = QFormLayout()
        layout.addRow('Minimum Displayed Value:', self.min_value_line_edit)
        layout.addRow('Maximum Displayed Value:', self.max_value_line_edit)
        layout.addRow(self.button_box)
        self.setLayout(layout)


class ImageToolsGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None = None, *, add_fourier_tool: bool = False) -> None:
        super().__init__('Tools', parent)
        self.home_button = QToolButton()
        self.save_button = QToolButton()
        self.move_button = QToolButton()
        self.fourier_button = QToolButton()
        self.ruler_button = QToolButton()
        self.rectangle_button = QToolButton()
        self.line_cut_button = QToolButton()

        self.home_button.setIcon(QIcon(':/icons/home'))
        self.home_button.setIconSize(QSize(32, 32))
        self.home_button.setToolTip('Home')

        self.save_button.setIcon(QIcon(':/icons/save'))
        self.save_button.setIconSize(QSize(32, 32))
        self.save_button.setToolTip('Save Image')

        self.move_button.setIcon(QIcon(':/icons/move'))
        self.move_button.setIconSize(QSize(32, 32))
        self.move_button.setToolTip('Move')

        self.fourier_button.setIcon(QIcon(':/icons/fourier'))
        self.fourier_button.setIconSize(QSize(32, 32))
        self.fourier_button.setToolTip('Fourier Transform')

        self.ruler_button.setIcon(QIcon(':/icons/ruler'))
        self.ruler_button.setIconSize(QSize(32, 32))
        self.ruler_button.setToolTip('Ruler')

        self.rectangle_button.setIcon(QIcon(':/icons/rectangle'))
        self.rectangle_button.setIconSize(QSize(32, 32))
        self.rectangle_button.setToolTip('Rectangle')

        self.line_cut_button.setIcon(QIcon(':/icons/line-cut'))
        self.line_cut_button.setIconSize(QSize(32, 32))
        self.line_cut_button.setToolTip('Line-Cut Profile')

        layout = QGridLayout()
        layout.addWidget(self.home_button, 0, 0)
        layout.addWidget(self.save_button, 0, 1)
        layout.addWidget(self.move_button, 0, 2)

        if add_fourier_tool:
            layout.addWidget(self.fourier_button, 0, 3)

        layout.addWidget(self.ruler_button, 1, 0)
        layout.addWidget(self.rectangle_button, 1, 1)
        layout.addWidget(self.line_cut_button, 1, 2)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)


class ImageRendererGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Colorize', parent)
        self.renderer_combo_box = QComboBox()
        self.transformation_combo_box = QComboBox()
        self.variant_combo_box = QComboBox()

        self.renderer_combo_box.setToolTip('Array Component')
        self.transformation_combo_box.setToolTip('Transformation')
        self.variant_combo_box.setToolTip('Variant')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(self.renderer_combo_box)
        layout.addWidget(self.transformation_combo_box)
        layout.addWidget(self.variant_combo_box)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)


class ImageDataRangeGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Data Range', parent)
        self.display_range_slider = DecimalRangeSlider.create_instance(Qt.Orientation.Horizontal)
        self.auto_button = QPushButton('Auto')
        self.edit_button = QPushButton('Edit')
        self.color_legend_button = QPushButton('Color Legend')

        self.display_range_slider.setToolTip('Display Value Range')
        self.auto_button.setToolTip('Rescale to Data Range')
        self.edit_button.setToolTip('Rescale to Custom Range')
        self.color_legend_button.setToolTip('Toggle Color Legend Visibility')

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.auto_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.color_legend_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(self.display_range_slider)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)


class ImageRibbon(QWidget):
    def __init__(self, parent: QWidget | None = None, *, add_fourier_tool: bool = False) -> None:
        super().__init__(parent)
        self.image_tools_group_box = ImageToolsGroupBox(add_fourier_tool=add_fourier_tool)
        self.colormap_group_box = ImageRendererGroupBox()
        self.data_range_group_box = ImageDataRangeGroupBox()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_tools_group_box)
        layout.addWidget(self.colormap_group_box)
        layout.addWidget(self.data_range_group_box)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)


class ImageWidget(VisualizationView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color_legend_min_value = 0.0
        self._color_legend_max_value = 1.0
        self._color_legend_stop_points: list[tuple[float, QColor]] = [
            (0.0, QColor(Qt.GlobalColor.green)),
            (0.5, QColor(Qt.GlobalColor.yellow)),
            (1.0, QColor(Qt.GlobalColor.red)),
        ]
        self._color_legend_num_ticks = 5  # TODO
        self._is_color_legend_visible = False
        self._is_color_legend_cyclic = False

    def set_color_legend_colors(
        self, values: RealArrayType, rgba_array: RealArrayType, is_cyclic: bool
    ) -> None:
        color_legend_stop_points: list[tuple[float, QColor]] = list()
        self._color_legend_min_value = values.min()
        self._color_legend_max_value = values.max()

        value_range = self._color_legend_max_value - self._color_legend_min_value
        normalized_values = (
            (values - self._color_legend_min_value) / value_range
            if value_range > 0
            else numpy.full_like(values, 0.5)
        )

        for x, rgba in zip(normalized_values.clip(0, 1), rgba_array):
            color = QColor()
            color.setRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
            color_legend_stop_points.append((x, color))

        self._color_legend_stop_points = color_legend_stop_points
        self._is_color_legend_cyclic = is_cyclic
        scene = self.scene()

        if scene is not None:
            scene.update()

    def set_color_legend_visible(self, visible: bool) -> None:
        self._is_color_legend_visible = visible
        scene = self.scene()

        if scene is not None:
            scene.update()

    @property
    def _color_legend_ticks(self) -> Iterator[float]:
        for tick in range(self._color_legend_num_ticks):
            a = tick / (self._color_legend_num_ticks - 1)
            yield (1.0 - a) * self._color_legend_min_value + a * self._color_legend_max_value

    def drawForeground(self, painter: QPainter | None, rect: QRectF) -> None:  # noqa: N802
        if not self._is_color_legend_visible:
            return

        fg_painter = QPainter(self.viewport())

        pen = QPen()
        pen.setWidth(3)
        fg_painter.setPen(pen)

        font_metrics = fg_painter.fontMetrics()
        dx = font_metrics.horizontalAdvance('m')
        dy = font_metrics.lineSpacing()

        viewport = self.viewport()

        if viewport is None:
            return

        widget_rect = viewport.rect()

        if self._is_color_legend_cyclic:
            legend_diameter = 6 * dx
            legend_margin = 2 * dx

            legend_rect = QRectF(0.0, 0.0, legend_diameter, legend_diameter)
            legend_rect.moveRight(widget_rect.right() - legend_margin)
            legend_rect.moveBottom(widget_rect.height() - legend_margin)

            cgradient = QConicalGradient(legend_rect.center(), 90.0)
            cgradient.setStops(self._color_legend_stop_points)
            fg_painter.setBrush(cgradient)
            fg_painter.drawEllipse(legend_rect)
        else:
            tick_labels = [f'{tick:5g}' for tick in self._color_legend_ticks]
            tick_label_width = max(font_metrics.width(label) for label in tick_labels)

            legend_width = 2 * dx
            legend_height = (2 * len(tick_labels) - 1) * dy
            legend_margin = tick_label_width + 2 * dx

            legend_rect = QRectF(0.0, 0.0, legend_width, legend_height)
            legend_rect.moveRight(widget_rect.right() - legend_margin)
            legend_rect.moveTop((widget_rect.height() - legend_height) // 2)

            lgradient = QLinearGradient(legend_rect.bottomLeft(), legend_rect.topLeft())
            lgradient.setStops(self._color_legend_stop_points)
            fg_painter.setBrush(lgradient)
            fg_painter.drawRect(legend_rect)

            tick_x0 = legend_rect.right() + dx
            tick_y0 = legend_rect.bottom() + font_metrics.strikeOutPos()

            for tick_index, tick_label in enumerate(tick_labels):
                tick_dy = (tick_index * legend_rect.height()) // (len(tick_labels) - 1)
                viewport_point = QPointF(tick_x0, tick_y0 - tick_dy)
                fg_painter.drawText(viewport_point, tick_label)


class ImageView(QWidget):
    def __init__(self, parent: QWidget | None = None, *, add_fourier_tool: bool = False) -> None:
        super().__init__(parent)
        self.image_ribbon = ImageRibbon(add_fourier_tool=add_fourier_tool)
        self.image_widget = ImageWidget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(self.image_ribbon)
        layout.addWidget(self.image_widget)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
