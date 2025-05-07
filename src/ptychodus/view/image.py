from __future__ import annotations
from collections.abc import Iterator

import numpy

from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QSize
from PyQt5.QtGui import QColor, QConicalGradient, QIcon, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.typing import RealArrayType

from .visualization import VisualizationView
from .widgets import BottomTitledGroupBox, DecimalLineEdit, DecimalSlider


class ImageDisplayRangeDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.button_box = QDialogButtonBox()
        self.min_value_line_edit = DecimalLineEdit.create_instance()
        self.max_value_line_edit = DecimalLineEdit.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageDisplayRangeDialog:
        dialog = cls(parent)
        dialog.setWindowTitle('Set Display Range')
        dialog.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        dialog.button_box.accepted.connect(dialog.accept)
        dialog.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        dialog.button_box.rejected.connect(dialog.reject)

        layout = QFormLayout()
        layout.addRow('Minimum Displayed Value:', dialog.min_value_line_edit)
        layout.addRow('Maximum Displayed Value:', dialog.max_value_line_edit)
        layout.addRow(dialog.button_box)
        dialog.setLayout(layout)

        return dialog


class ImageToolsGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Tools', parent)
        self.home_button = QToolButton()
        self.save_button = QToolButton()
        self.move_button = QToolButton()
        self.ruler_button = QToolButton()
        self.rectangle_button = QToolButton()
        self.line_cut_button = QToolButton()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageToolsGroupBox:
        view = cls(parent)

        view.home_button.setIcon(QIcon(':/icons/home'))
        view.home_button.setIconSize(QSize(32, 32))
        view.home_button.setToolTip('Home')

        view.save_button.setIcon(QIcon(':/icons/save'))
        view.save_button.setIconSize(QSize(32, 32))
        view.save_button.setToolTip('Save Image')

        view.move_button.setIcon(QIcon(':/icons/move'))
        view.move_button.setIconSize(QSize(32, 32))
        view.move_button.setToolTip('Move')

        view.ruler_button.setIcon(QIcon(':/icons/ruler'))
        view.ruler_button.setIconSize(QSize(32, 32))
        view.ruler_button.setToolTip('Ruler')

        view.rectangle_button.setIcon(QIcon(':/icons/rectangle'))
        view.rectangle_button.setIconSize(QSize(32, 32))
        view.rectangle_button.setToolTip('Rectangle')

        view.line_cut_button.setIcon(QIcon(':/icons/line-cut'))
        view.line_cut_button.setIconSize(QSize(32, 32))
        view.line_cut_button.setToolTip('Line-Cut Profile')

        layout = QGridLayout()
        layout.addWidget(view.home_button, 0, 0)
        layout.addWidget(view.save_button, 0, 1)
        layout.addWidget(view.move_button, 0, 2)
        layout.addWidget(view.ruler_button, 1, 0)
        layout.addWidget(view.rectangle_button, 1, 1)
        layout.addWidget(view.line_cut_button, 1, 2)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        return view


class ImageRendererGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Colorize', parent)
        self.renderer_combo_box = QComboBox()
        self.transformation_combo_box = QComboBox()
        self.variant_combo_box = QComboBox()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageRendererGroupBox:
        view = cls(parent)

        view.renderer_combo_box.setToolTip('Array Component')
        view.transformation_combo_box.setToolTip('Transformation')
        view.variant_combo_box.setToolTip('Variant')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(view.renderer_combo_box)
        layout.addWidget(view.transformation_combo_box)
        layout.addWidget(view.variant_combo_box)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        return view


class ImageDataRangeGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Data Range', parent)
        self.min_display_value_slider = DecimalSlider.create_instance(Qt.Orientation.Horizontal)
        self.max_display_value_slider = DecimalSlider.create_instance(Qt.Orientation.Horizontal)
        self.auto_button = QPushButton('Auto')
        self.edit_button = QPushButton('Edit')
        self.color_legend_button = QPushButton('Color Legend')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageDataRangeGroupBox:
        view = cls(parent)

        view.min_display_value_slider.setToolTip('Minimum Display Value')
        view.max_display_value_slider.setToolTip('Maximum Display Value')
        view.auto_button.setToolTip('Rescale to Data Range')
        view.edit_button.setToolTip('Rescale to Custom Range')
        view.color_legend_button.setToolTip('Toggle Color Legend Visibility')

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(view.auto_button)
        button_layout.addWidget(view.edit_button)
        button_layout.addWidget(view.color_legend_button)

        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addRow('Min:', view.min_display_value_slider)
        layout.addRow('Max:', view.max_display_value_slider)
        layout.addRow(button_layout)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        return view


class ImageRibbon(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.image_tools_group_box = ImageToolsGroupBox.create_instance()
        self.colormap_group_box = ImageRendererGroupBox.create_instance()
        self.data_range_group_box = ImageDataRangeGroupBox.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageRibbon:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.image_tools_group_box)
        layout.addWidget(view.colormap_group_box)
        layout.addWidget(view.data_range_group_box)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        return view


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
        self.scene().update()

    def set_color_legend_visible(self, visible: bool) -> None:
        self._is_color_legend_visible = visible
        self.scene().update()

    @property
    def _color_legend_ticks(self) -> Iterator[float]:
        for tick in range(self._color_legend_num_ticks):
            a = tick / (self._color_legend_num_ticks - 1)
            yield (1.0 - a) * self._color_legend_min_value + a * self._color_legend_max_value

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        if not self._is_color_legend_visible:
            return

        fg_painter = QPainter(self.viewport())

        pen = QPen()
        pen.setWidth(3)
        fg_painter.setPen(pen)

        font_metrics = fg_painter.fontMetrics()
        dx = font_metrics.horizontalAdvance('m')
        dy = font_metrics.lineSpacing()

        widget_rect = self.viewport().rect()

        if self._is_color_legend_cyclic:
            legend_diameter = 6 * dx
            legend_margin = 2 * dx

            legend_rect = QRect(0, 0, legend_diameter, legend_diameter)
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

            legend_rect = QRect(0, 0, legend_width, legend_height)
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
                viewport_point = QPoint(tick_x0, tick_y0 - tick_dy)
                fg_painter.drawText(viewport_point, tick_label)


class ImageView(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.image_ribbon = ImageRibbon.create_instance()
        self.image_widget = ImageWidget()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> ImageView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(view.image_ribbon)
        layout.addWidget(view.image_widget)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        return view
