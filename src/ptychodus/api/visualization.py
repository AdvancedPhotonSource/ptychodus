from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from scipy.stats import gaussian_kde
import numpy

from .geometry import Box2D, Interval, Line2D, PixelGeometry
from .typing import NumberArrayType, RealArrayType


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: Sequence[float]

    def copy(self) -> PlotSeries:
        return PlotSeries(str(self.label), list(self.values))


@dataclass(frozen=True)
class PlotAxis:
    label: str
    series: Sequence[PlotSeries]

    @classmethod
    def create_null(cls) -> PlotAxis:
        return cls('', [])

    def copy(self) -> PlotAxis:
        return PlotAxis(str(self.label), [series.copy() for series in self.series])


@dataclass(frozen=True)
class Plot2D:
    axis_x: PlotAxis
    axis_y: PlotAxis

    @classmethod
    def create_null(cls) -> Plot2D:
        return cls(PlotAxis.create_null(), PlotAxis.create_null())

    def copy(self) -> Plot2D:
        return Plot2D(self.axis_x.copy(), self.axis_y.copy())


@dataclass(frozen=True)
class LineCut:
    distance_m: Sequence[float]
    value: Sequence[float | complex]


@dataclass(frozen=True)
class KernelDensityEstimate:
    value_lower: float
    value_upper: float
    kde: gaussian_kde


class VisualizationProduct:
    EPS: Final[float] = 1.0e-6

    def __init__(
        self,
        value_label: str,
        values: NumberArrayType,
        rgba: RealArrayType,
        pixel_geometry: PixelGeometry,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f'Values must be a 2-dimensional ndarray (actual={values.ndim}).')

        if rgba.ndim != 3:
            raise ValueError(f'RGBA must be a 3-dimensional ndarray (actual={rgba.ndim}).')

        if rgba.shape[2] != 4:
            raise ValueError(f'RGBA final dimension must have length=4 (actual={rgba.shape[2]}).')

        if values.shape[0] != rgba.shape[0] or values.shape[1] != rgba.shape[1]:
            raise ValueError(f'Shape mismatch (values={values.shape} and rgba={rgba.shape}).')

        self._value_label = value_label
        self._values = values
        self._rgba = rgba
        self._pixel_width_m = pixel_geometry.width_m
        self._pixel_height_m = pixel_geometry.height_m

    def get_value_label(self) -> str:
        return self._value_label

    def get_values(self) -> NumberArrayType:
        return self._values

    def get_image_rgba(self) -> RealArrayType:
        return self._rgba

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self._pixel_width_m,
            height_m=self._pixel_height_m,
        )

    @staticmethod
    def _intersect_bounding_box(begin: float, end: float, n: int) -> Interval[float]:
        length = end - begin

        if abs(length) < VisualizationProduct.EPS:
            return Interval[float](-numpy.inf, numpy.inf)
        else:
            return Interval[float].create_proper(
                (0 - begin) / length,
                (n - begin) / length,
            )

    @staticmethod
    def _intersect_grid_lines(
        begin: float, end: float, alpha_limits: Interval[float]
    ) -> Iterator[float]:
        ibegin = int(begin)
        iend = int(end)

        if iend < ibegin:
            ibegin, iend = iend, ibegin

        length = end - begin

        if abs(length) > VisualizationProduct.EPS:
            for idx in range(ibegin, iend + 1):
                alpha = (idx - begin) / length

                if alpha in alpha_limits:
                    yield alpha

    def _clip_to_bounding_box(self, line: Line2D) -> Interval[float]:
        alpha_x = self._intersect_bounding_box(line.begin.x, line.end.x, self._values.shape[-1])
        alpha_y = self._intersect_bounding_box(line.begin.y, line.end.y, self._values.shape[-2])

        return Interval[float].create_proper(
            max(0.0, max(alpha_x.lower, alpha_y.lower)),
            min(1.0, min(alpha_x.upper, alpha_y.upper)),
        )

    def _intersect_grid(self, line: Line2D) -> Sequence[float]:
        alpha_limits = self._clip_to_bounding_box(line)
        x_intersections = [
            x for x in self._intersect_grid_lines(line.begin.x, line.end.x, alpha_limits)
        ]
        y_intersections = [
            x for x in self._intersect_grid_lines(line.begin.y, line.end.y, alpha_limits)
        ]

        alpha = {alpha_limits.lower, alpha_limits.upper}
        alpha = alpha.union(x_intersections)
        alpha = alpha.union(y_intersections)
        return sorted(alpha)

    def get_info_text(self, x: float, y: float) -> str:
        ix = 0 if x < 0.0 else int(x)
        ix = min(ix, self._values.shape[-1])
        iy = 0 if y < 0.0 else int(y)
        iy = min(iy, self._values.shape[-2])
        value = self._values[iy, ix]

        if numpy.iscomplex(value):
            amplitude = numpy.absolute(value)
            phase = numpy.angle(value)
            return f'{x=:.1f} {y=:.1f} {amplitude=:6g} {phase=:6g}'

        return f'{x=:.1f} {y=:.1f} {value=:6g}'

    def get_line_cut(self, line: Line2D) -> LineCut:
        intersections = self._intersect_grid(line)

        dx = (line.end.x - line.begin.x) * self._pixel_width_m
        dy = (line.end.y - line.begin.y) * self._pixel_height_m
        line_length = numpy.hypot(dx, dy)

        distances: list[float] = list()
        values: list[float] = list()

        for alpha_l, alpha_r in zip(intersections[:-1], intersections[1:]):
            alpha = (alpha_l + alpha_r) / 2.0
            point = line.lerp(alpha)
            value = self._values[int(point.y), int(point.x)]

            distances.append(alpha * line_length)
            values.append(value)

        return LineCut(distances, values)

    def estimate_kernel_density(self, box: Box2D) -> KernelDensityEstimate:
        x_range = Interval[int](0, self._values.shape[-1])
        x_begin = x_range.clamp(int(box.x_begin))
        x_end = x_range.clamp(int(box.x_end) + 1)

        y_range = Interval[int](0, self._values.shape[-2])
        y_begin = y_range.clamp(int(box.y_begin))
        y_end = y_range.clamp(int(box.y_end) + 1)

        values = self._values[..., y_begin:y_end, x_begin:x_end]
        values = values.reshape(values.shape[-3], -1) if values.ndim > 2 else values.reshape(-1)

        if numpy.iscomplexobj(values):
            # TODO improve KDE for complex values
            values = numpy.absolute(values)

        return KernelDensityEstimate(values.min(), values.max(), gaussian_kde(values))
