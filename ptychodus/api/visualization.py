from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

from scipy.stats import gaussian_kde
import numpy
import numpy.typing

from .geometry import Box2D, Interval, Line2D
from .patterns import PixelGeometry

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
NumberTypes: TypeAlias = numpy.integer[Any] | numpy.floating[Any] | numpy.complexfloating[Any, Any]
NumberArrayType: TypeAlias = numpy.typing.NDArray[NumberTypes]


class ScalarTransformation(ABC):  # FIXME remove
    '''interface for real-valued transformations of a real array'''

    @abstractmethod
    def decorateText(self, text: str) -> str:
        pass

    @abstractmethod
    def __call__(self, array: RealArrayType) -> RealArrayType:
        '''returns the transformed input array'''
        pass


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
    value: Sequence[float]
    valueLabel: str  # FIXME remove


@dataclass(frozen=True)
class KernelDensityEstimate:
    valueLower: float
    valueUpper: float
    kde: gaussian_kde


class VisualizationProduct:
    EPS: Final[float] = 1.e-6

    def __init__(self, valueLabel: str, values: NumberArrayType, rgba: RealArrayType,
                 pixelGeometry: PixelGeometry) -> None:
        if values.ndim != 2:
            raise ValueError('Values must be a 2-dimensional ndarray.')

        if rgba.ndim != 3:
            raise ValueError('RGBA must be a 3-dimensional ndarray.')

        # FIXME validate array shapes

        self._valueLabel = valueLabel
        self._values = numpy.stack((numpy.absolute(values), numpy.angle(values))) \
                if numpy.iscomplexobj(values) else values
        self._rgba = rgba
        self._pixelWidthInMeters = pixelGeometry.widthInMeters
        self._pixelHeightInMeters = pixelGeometry.heightInMeters

    def getValueLabel(self) -> str:
        return self._valueLabel

    def getImageRGBA(self) -> RealArrayType:
        return self._rgba

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
    def _intersectGridLines(begin: float, end: float,
                            alphaLimits: Interval[float]) -> Iterator[float]:
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
            max(0., max(alphaX.lower, alphaY.lower)),
            min(1., min(alphaX.upper, alphaY.upper)),
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
        ix = 0 if x < 0. else int(x)
        ix = min(ix, self._values.shape[-1])
        iy = 0 if y < 0. else int(y)
        iy = min(iy, self._values.shape[-2])

        if self._values.ndim == 3:
            amplitude = self._values[0, iy, ix]
            phase = self._values[1, iy, ix]
            return f'{x=:.1f} {y=:.1f} {amplitude=:6g} {phase=:6g}'

        value = self._values[iy, ix]
        return f'{x=:.1f} {y=:.1f} {value=:6g}'

    def getLineCut(self, line: Line2D) -> LineCut:
        intersections = self._intersectGrid(line)

        dx = (line.end.x - line.begin.x) * self._pixelWidthInMeters
        dy = (line.end.y - line.begin.y) * self._pixelHeightInMeters
        lineLength = numpy.hypot(dx, dy)

        distances: list[float] = list()
        values: list[float] = list()

        for alphaL, alphaR in zip(intersections[:-1], intersections[1:]):
            alpha = (alphaL + alphaR) / 2.
            point = line.lerp(alpha)
            value = self._values[int(point.y), int(point.x)]

            distances.append(alpha * lineLength)
            values.append(value)

        return LineCut(distances, values, self._valueLabel)

    def estimateKernelDensity(self, box: Box2D) -> KernelDensityEstimate:
        x_begin = int(box.x_begin)
        x_end = int(box.x_end) + 1
        y_begin = int(box.y_begin)
        y_end = int(box.y_end) + 1

        # FIXME clamp {xy}_begin, {xy}_end to data
        # FIXME check complex values, multimodal probes, multilayer objects

        # FIXME Datapoints to estimate from. In case of univariate data this is a 1-D array,
        # FIXME otherwise a 2-D array with shape (# of dims, # of data).
        values = self._values[..., y_begin:y_end, x_begin:x_end]
        values = values.reshape(values.shape[0], -1) if values.ndim == 3 \
                else values.reshape(-1)
        kde = gaussian_kde(values)
        return KernelDensityEstimate(values.min(), values.max(), kde)
