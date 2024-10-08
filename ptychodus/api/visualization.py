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
    def createNull(cls) -> PlotAxis:
        return cls('', [])

    def copy(self) -> PlotAxis:
        return PlotAxis(str(self.label), [series.copy() for series in self.series])


@dataclass(frozen=True)
class Plot2D:
    axisX: PlotAxis
    axisY: PlotAxis

    @classmethod
    def createNull(cls) -> Plot2D:
        return cls(PlotAxis.createNull(), PlotAxis.createNull())

    def copy(self) -> Plot2D:
        return Plot2D(self.axisX.copy(), self.axisY.copy())


@dataclass(frozen=True)
class LineCut:
    distanceInMeters: Sequence[float]
    value: Sequence[float | complex]


@dataclass(frozen=True)
class KernelDensityEstimate:
    valueLower: float
    valueUpper: float
    kde: gaussian_kde


class VisualizationProduct:
    EPS: Final[float] = 1.0e-6

    def __init__(
        self,
        valueLabel: str,
        values: NumberArrayType,
        rgba: RealArrayType,
        pixelGeometry: PixelGeometry,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f'Values must be a 2-dimensional ndarray (actual={values.ndim}).')

        if rgba.ndim != 3:
            raise ValueError(f'RGBA must be a 3-dimensional ndarray (actual={rgba.ndim}).')

        if rgba.shape[2] != 4:
            raise ValueError(f'RGBA final dimension must have length=4 (actual={rgba.shape[2]}).')

        if values.shape[0] != rgba.shape[0] or values.shape[1] != rgba.shape[1]:
            raise ValueError(f'Shape mismatch (values={values.shape} and rgba={rgba.shape}).')

        self._valueLabel = valueLabel
        self._values = values
        self._rgba = rgba
        self._pixelWidthInMeters = pixelGeometry.widthInMeters
        self._pixelHeightInMeters = pixelGeometry.heightInMeters

    def getValueLabel(self) -> str:
        return self._valueLabel

    def getValues(self) -> NumberArrayType:
        return self._values

    def getImageRGBA(self) -> RealArrayType:
        return self._rgba

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self._pixelWidthInMeters,
            heightInMeters=self._pixelHeightInMeters,
        )

    @staticmethod
    def _intersectBoundingBox(begin: float, end: float, n: int) -> Interval[float]:
        length = end - begin

        if abs(length) < VisualizationProduct.EPS:
            return Interval[float](-numpy.inf, numpy.inf)
        else:
            return Interval[float].createProper(
                (0 - begin) / length,
                (n - begin) / length,
            )

    @staticmethod
    def _intersectGridLines(
        begin: float, end: float, alphaLimits: Interval[float]
    ) -> Iterator[float]:
        ibegin = int(begin)
        iend = int(end)

        if iend < ibegin:
            ibegin, iend = iend, ibegin

        length = end - begin

        if abs(length) > VisualizationProduct.EPS:
            for idx in range(ibegin, iend + 1):
                alpha = (idx - begin) / length

                if alpha in alphaLimits:
                    yield alpha

    def _clipToBoundingBox(self, line: Line2D) -> Interval[float]:
        alphaX = self._intersectBoundingBox(line.begin.x, line.end.x, self._values.shape[-1])
        alphaY = self._intersectBoundingBox(line.begin.y, line.end.y, self._values.shape[-2])

        return Interval[float].createProper(
            max(0.0, max(alphaX.lower, alphaY.lower)),
            min(1.0, min(alphaX.upper, alphaY.upper)),
        )

    def _intersectGrid(self, line: Line2D) -> Sequence[float]:
        alphaLimits = self._clipToBoundingBox(line)
        xIntersections = [
            x for x in self._intersectGridLines(line.begin.x, line.end.x, alphaLimits)
        ]
        yIntersections = [
            x for x in self._intersectGridLines(line.begin.y, line.end.y, alphaLimits)
        ]

        alpha = {alphaLimits.lower, alphaLimits.upper}
        alpha = alpha.union(xIntersections)
        alpha = alpha.union(yIntersections)
        return sorted(alpha)

    def getInfoText(self, x: float, y: float) -> str:
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

    def getLineCut(self, line: Line2D) -> LineCut:
        intersections = self._intersectGrid(line)

        dx = (line.end.x - line.begin.x) * self._pixelWidthInMeters
        dy = (line.end.y - line.begin.y) * self._pixelHeightInMeters
        lineLength = numpy.hypot(dx, dy)

        distances: list[float] = list()
        values: list[float] = list()

        for alphaL, alphaR in zip(intersections[:-1], intersections[1:]):
            alpha = (alphaL + alphaR) / 2.0
            point = line.lerp(alpha)
            value = self._values[int(point.y), int(point.x)]

            distances.append(alpha * lineLength)
            values.append(value)

        return LineCut(distances, values)

    def estimateKernelDensity(self, box: Box2D) -> KernelDensityEstimate:
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
