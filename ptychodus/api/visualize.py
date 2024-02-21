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


@dataclass(frozen=True)
class PlotUncertainSeries:
    label: str
    lo: Sequence[float]
    values: Sequence[float]
    hi: Sequence[float]


@dataclass(frozen=True)
class PlotAxis:
    label: str
    series: Sequence[PlotSeries]

    @classmethod
    def createNull(cls) -> PlotAxis:
        return cls('', [])


@dataclass(frozen=True)
class PlotUncertainAxis:
    label: str
    series: Sequence[PlotUncertainSeries]

    @classmethod
    def createNull(cls) -> PlotUncertainAxis:
        return cls('', [])


@dataclass(frozen=True)
class Plot2D:
    axisX: PlotAxis
    axisY: PlotAxis

    @classmethod
    def createNull(cls) -> Plot2D:
        return cls(PlotAxis.createNull(), PlotAxis.createNull())


@dataclass(frozen=True)
class PlotUncertain2D:
    axisX: PlotAxis
    axisY: PlotUncertainAxis

    @classmethod
    def createNull(cls) -> PlotUncertain2D:
        return cls(PlotAxis.createNull(), PlotUncertainAxis.createNull())


@dataclass(frozen=True)
class LineCut:
    distanceInMeters: Sequence[float]
    value: Sequence[float]
    valueLabel: str
