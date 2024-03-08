from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy
import numpy.typing

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


class ScalarTransformation(ABC):
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
class PlotUncertainSeries:
    label: str
    lo: Sequence[float]
    values: Sequence[float]
    hi: Sequence[float]

    def copy(self) -> PlotUncertainSeries:
        return PlotUncertainSeries(str(self.label), list(self.lo), list(self.values),
                                   list(self.hi))


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
class PlotUncertainAxis:
    label: str
    series: Sequence[PlotUncertainSeries]

    @classmethod
    def createNull(cls) -> PlotUncertainAxis:
        return cls('', [])

    def copy(self) -> PlotUncertainAxis:
        return PlotUncertainAxis(str(self.label), [series.copy() for series in self.series])


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
class PlotUncertain2D:
    axisX: PlotAxis
    axisY: PlotUncertainAxis

    @classmethod
    def createNull(cls) -> PlotUncertain2D:
        return cls(PlotAxis.createNull(), PlotUncertainAxis.createNull())

    def copy(self) -> PlotUncertain2D:
        return PlotUncertain2D(self.axisX.copy(), self.axisY.copy())


@dataclass(frozen=True)
class LineCut:
    distanceInMeters: Sequence[float]
    value: Sequence[float]
    valueLabel: str
