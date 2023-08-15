from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: Sequence[float]


@dataclass(frozen=True)
class PlotAxis:
    label: str
    series: Sequence[PlotSeries]

    @classmethod
    def createEmpty(cls) -> PlotAxis:
        return cls('', [])


@dataclass(frozen=True)
class Plot2D:
    axisX: PlotAxis
    axisY: PlotAxis

    @classmethod
    def createEmpty(cls) -> Plot2D:
        return cls(PlotAxis.createEmpty(), PlotAxis.createEmpty())
